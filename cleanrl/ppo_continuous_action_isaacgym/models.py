import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.losses.kl_divergence import kl_divergence_from_nn

from custom_layers import BayesianLinear


class Agent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        clipping_val: float,
        policy_layers: Sequence[int],
        value_layers: Sequence[int],
        entropy_cost: float,
        discounting: float,
        reward_scaling: float,
        device: str,
    ):
        super(Agent, self).__init__()

        policy = []
        for w1, w2 in zip(policy_layers, policy_layers[1:]):
            policy.append(
                nn.Linear(w1, w2),
            )
            policy.append(nn.SiLU())
        policy.pop()  # drop the final activation
        self.policy = nn.Sequential(*policy)

        value = []
        for w1, w2 in zip(value_layers, value_layers[1:]):
            value.append(
                nn.Linear(w1, w2),
            )
            value.append(nn.SiLU())
        value.pop()  # drop the final activation
        self.value = nn.Sequential(*value)

        self.num_steps = torch.zeros((), device=device)
        self.running_mean = torch.zeros(policy_layers[0], device=device)
        self.running_variance = torch.zeros(policy_layers[0], device=device)

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = clipping_val
        self.device = device

    @torch.jit.export
    def dist_create(self, logits):
        """Normal followed by tanh.

        torch.distribution doesn't work with torch.jit, so we roll our own."""
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + 0.001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(self, loc, scale):
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x):
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc, scale):
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(self, loc, scale, dist):
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation):
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance = self.running_variance + var_diff

    @torch.jit.export
    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    @torch.jit.export
    def get_logits_action(self, observation):
        observation = self.normalize(observation)
        logits = self.policy(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    @torch.jit.export
    def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = (
            reward + self.discounting * (1 - termination) * values_t_plus_1 - values
        )
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)

        for ti in range(truncation_mask.shape[0]):
            ti = truncation_mask.shape[0] - ti - 1
            acc = (
                deltas[ti]
                + self.discounting
                * (1 - termination[ti])
                * truncation_mask[ti]
                * self.lambda_
                * acc
            )
            vs_minus_v_xs[ti] = acc

        # Add V(x_s) to get v_s.
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
        advantages = (
            reward + self.discounting * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return vs, advantages

    @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor]):
        observation = self.normalize(td["observation"])
        policy_logits = self.policy(observation[:-1])
        baseline = self.value(observation)
        baseline = torch.squeeze(baseline, dim=-1)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = baseline[-1]
        baseline = baseline[:-1]
        reward = td["reward"] * self.reward_scaling
        termination = td["done"] * (1 - td["truncation"])

        loc, scale = self.dist_create(td["logits"])
        behaviour_action_log_probs = self.dist_log_prob(loc, scale, td["action"])
        loc, scale = self.dist_create(policy_logits)
        target_action_log_probs = self.dist_log_prob(loc, scale, td["action"])

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td["truncation"],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )

        rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = torch.mean(self.dist_entropy(loc, scale))
        entropy_loss = self.entropy_cost * -entropy
        kl_loss = torch.zeros_like(entropy_loss)

        return (
            policy_loss + v_loss + entropy_loss + kl_loss,
            policy_loss,
            v_loss,
            entropy_loss,
            kl_loss,
        )


class BayesianAgent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        clipping_val: float,
        number_of_cell_types: int,
        policy_layers: Sequence[int],
        value_layers: Sequence[int],
        entropy_cost: float,
        discounting: float,
        reward_scaling: float,
        device: str,
        complexity_cost: float,
    ):
        super(BayesianAgent, self).__init__()

        policy = []
        for w1, w2 in zip(policy_layers, policy_layers[1:]):
            policy.append(
                BayesianLinear(
                    in_features=w1,
                    out_features=w2,
                    neuron_types_in=number_of_cell_types,
                    neuron_types_out=number_of_cell_types,
                )
            )
            policy.append(nn.SiLU())
        policy.pop()  # drop the final activation
        self.policy = nn.Sequential(*policy)

        value = []
        for w1, w2 in zip(value_layers, value_layers[1:]):
            value.append(
                BayesianLinear(
                    in_features=w1,
                    out_features=w2,
                    neuron_types_in=number_of_cell_types,
                    neuron_types_out=number_of_cell_types,
                )
            )
            value.append(nn.SiLU())
        value.pop()  # drop the final activation
        self.value = nn.Sequential(*value)

        self.num_steps = torch.zeros((), device=device)
        self.running_mean = torch.zeros(policy_layers[0], device=device)
        self.running_variance = torch.zeros(policy_layers[0], device=device)

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = 0.3
        self.complexity_cost = complexity_cost
        self.device = device

    @torch.jit.export
    def dist_create(self, logits):
        """Normal followed by tanh.

        torch.distribution doesn't work with torch.jit, so we roll our own."""
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + 0.001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(self, loc, scale):
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x):
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc, scale):
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(self, loc, scale, dist):
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation):
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance = self.running_variance + var_diff

    @torch.jit.export
    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    @torch.jit.export
    def get_logits_action(self, observation):
        observation = self.normalize(observation)
        logits = self.policy(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    @torch.jit.export
    def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = (
            reward + self.discounting * (1 - termination) * values_t_plus_1 - values
        )
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)

        for ti in range(truncation_mask.shape[0]):
            ti = truncation_mask.shape[0] - ti - 1
            acc = (
                deltas[ti]
                + self.discounting
                * (1 - termination[ti])
                * truncation_mask[ti]
                * self.lambda_
                * acc
            )
            vs_minus_v_xs[ti] = acc

        # Add V(x_s) to get v_s.
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
        advantages = (
            reward + self.discounting * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return vs, advantages

    @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor]):
        observation = self.normalize(td["observation"])
        policy_logits = self.policy(observation[:-1])
        baseline = self.value(observation)
        baseline = torch.squeeze(baseline, dim=-1)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = baseline[-1]
        baseline = baseline[:-1]
        reward = td["reward"] * self.reward_scaling
        termination = td["done"] * (1 - td["truncation"])

        loc, scale = self.dist_create(td["logits"])
        behaviour_action_log_probs = self.dist_log_prob(loc, scale, td["action"])
        loc, scale = self.dist_create(policy_logits)
        target_action_log_probs = self.dist_log_prob(loc, scale, td["action"])

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td["truncation"],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )

        rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = torch.mean(self.dist_entropy(loc, scale))
        entropy_loss = self.entropy_cost * -entropy
        kl_loss = self.complexity_cost * (
            kl_divergence_from_nn(self.policy) + kl_divergence_from_nn(self.value)
        )

        return (
            policy_loss + v_loss + entropy_loss + kl_loss,
            policy_loss,
            v_loss,
            entropy_loss,
            kl_loss,
        )

    def sample_mean_agent(
        self, clipping_val: float, learning_rate: float, entropy_cost: float
    ):
        self.learning_rate = learning_rate
        self.entropy_cost = entropy_cost
        policy_layers = []
        value_layers = []
        for a_layer in self.policy:
            if type(a_layer) == BayesianLinear:
                policy_layers.append(
                    self.construct_vanilla_layer(
                        a_layer.weight_sampler.mu[0].T, a_layer.bias_sampler.mu
                    )
                )
                policy_layers.append(nn.SiLU())
        for a_layer in self.value:
            if type(a_layer) == BayesianLinear:
                value_layers.append(
                    self.construct_vanilla_layer(
                        a_layer.weight_sampler.mu[0].T, a_layer.bias_sampler.mu
                    )
                )
                value_layers.append(nn.SiLU())
        policy_layers.pop()
        value_layers.pop()
        policy_layer_widths = [
            layer.in_features for layer in policy_layers if type(layer) == nn.Linear
        ]
        value_layer_widths = [
            layer.in_features for layer in value_layers if type(layer) == nn.Linear
        ]
        vanilla_agent = Agent(
            clipping_val,
            policy_layer_widths,
            value_layer_widths,
            self.entropy_cost,
            self.discounting,
            self.reward_scaling,
            self.device,
        )
        vanilla_agent.policy = torch.nn.Sequential(*policy_layers)
        vanilla_agent.value = torch.nn.Sequential(*value_layers)
        vanilla_agent.running_mean = self.running_mean
        vanilla_agent.running_variance = self.running_variance
        vanilla_agent.num_steps = self.num_steps
        return vanilla_agent

    def sample_vanilla_agent(
        self, clipping_val: float, learning_rate: float, entropy_cost: float
    ):
        self.learning_rate = learning_rate
        self.entropy_cost = entropy_cost
        policy_layers = []
        value_layers = []
        for a_layer in self.policy:
            if type(a_layer) == BayesianLinear:
                policy_layers.append(
                    self.construct_vanilla_layer(
                        a_layer.weight_sampler.sample(),
                        a_layer.bias_sampler.sample(),
                    )
                )
                policy_layers.append(nn.SiLU())
        for a_layer in self.value:
            if type(a_layer) == BayesianLinear:
                value_layers.append(
                    self.construct_vanilla_layer(
                        a_layer.weight_sampler.sample(),
                        a_layer.bias_sampler.sample(),
                    )
                )
                value_layers.append(nn.SiLU())
        policy_layers.pop()
        value_layers.pop()
        policy_layer_widths = [
            layer.in_features for layer in policy_layers if type(layer) == nn.Linear
        ]
        value_layer_widths = [
            layer.in_features for layer in value_layers if type(layer) == nn.Linear
        ]
        vanilla_agent = Agent(
            clipping_val,
            policy_layer_widths,
            value_layer_widths,
            self.entropy_cost,
            self.discounting,
            self.reward_scaling,
            self.device,
        )
        vanilla_agent.policy = torch.nn.Sequential(*policy_layers)
        vanilla_agent.value = torch.nn.Sequential(*value_layers)
        vanilla_agent.running_mean = self.running_mean
        vanilla_agent.running_variance = self.running_variance
        vanilla_agent.num_steps = self.num_steps
        return vanilla_agent

    def construct_vanilla_layer(self, weights, biases):
        layer = nn.Linear(weights.shape[1], weights.shape[0], bias=True)
        layer.weight.data = weights
        layer.bias.data = biases
        return layer