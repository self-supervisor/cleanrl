import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from blitz.losses import kl_divergence_from_nn
from blitz.modules.base_bayesian_module import BayesianModule, BayesianRNN
from blitz.modules.weight_sampler import (
    PriorWeightDistribution,
    TrainableRandomDistribution,
)
from blitz.utils import variational_estimator

from utils import (
    find_num_neuron_per_type,
    gather2D,
    gather_nd_torch_dims_flat,
    get_indices,
)


class TrainableRandomDistribution_weight_share(nn.Module):
    def __init__(self, mu, rho, indices):
        super().__init__()

        self.weight_mu_share = mu
        self.weight_rho_share = rho
        self.indices = indices

        self.indices.flat = gather_nd_torch_dims_flat(
            self.weight_mu_share, self.indices, batch_dim=1
        )

        self.mu = gather2D(self.weight_mu_share, self.indices, self.indices.flat)
        self.rho = gather2D(self.weight_rho_share, self.indices, self.indices.flat)
        self.register_buffer("eps_w", torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is
        a function from a trainable parameter, and adding a mean
        sets those weights as the current ones
        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.mu = gather2D(self.weight_mu_share, self.indices, self.indices.flat)
        self.rho = gather2D(self.weight_rho_share, self.indices, self.indices.flat)

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w[0].permute(1, 0)

    def log_posterior(self, w=None):
        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost
        returns:
            torch.tensor with shape []
        """

        assert (
            self.w is not None
        ), "You can only have a log posterior for W if you've already sampled it"
        if w is None:
            w = self.w

        log_sqrt2pi = np.log(np.sqrt(2 * self.pi))
        log_posteriors = (
            -log_sqrt2pi
            - torch.log(self.sigma)
            - (((w - self.mu) ** 2) / (2 * self.sigma**2))
            - 0.5
        )
        return log_posteriors.sum()


class BayesianLinear(BayesianModule):
    """
    Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).
    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        neuron_types_in: int -> number of input type neurons
        neuron_types_out: int -> number of output type neurons
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """

    def __init__(
        self,
        in_features,
        out_features,
        neuron_types_in,
        neuron_types_out,
        WS=True,
        bias=True,
        prior_sigma_1=0.1,
        prior_sigma_2=0.4,
        prior_pi=1,
        posterior_mu_init=0,
        posterior_rho_init=-7.0,
        freeze=False,
        prior_dist=None,
    ):
        super().__init__()

        # our main parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        self.pnet_size = in_features * out_features
        self.gnet_size = neuron_types_in * neuron_types_out

        self.WS = WS
        # Variational weight parameters and sample
        if self.WS == False:
            self.weight_mu = nn.Parameter(
                torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1)
            )
            self.weight_rho = nn.Parameter(
                torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1)
            )
            self.weight_sampler = TrainableRandomDistribution(
                self.weight_mu, self.weight_rho
            )

            # Variational bias parameters and sample
            self.bias_mu = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_mu_init, 0.1)
            )
            self.bias_rho = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_rho_init, 0.1)
            )
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
        else:
            self.neuron_types_in = neuron_types_in
            self.neuron_types_out = neuron_types_out
            n_in_neurons_per_type = find_num_neuron_per_type(
                in_features, neuron_types_in
            )
            in_neuron_type_ids = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_in_neurons_per_type)], axis=0
            ).astype(np.int32)

            n_out_neurons_per_type = find_num_neuron_per_type(
                out_features, neuron_types_out
            )
            out_neuron_type_ids = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_out_neurons_per_type)], axis=0
            ).astype(np.int32)

            indices = get_indices(
                in_neuron_type_ids, in_features, out_neuron_type_ids, out_features
            )
            self.weight_mu_share = nn.Parameter(
                torch.Tensor(1, neuron_types_in, neuron_types_out).normal_(
                    posterior_mu_init, 0.1
                )
            )
            self.weight_rho_share = nn.Parameter(
                torch.Tensor(1, neuron_types_in, neuron_types_out).normal_(
                    posterior_rho_init, 0.1
                )
            )

            self.weight_sampler = TrainableRandomDistribution_weight_share(
                self.weight_mu_share, self.weight_rho_share, indices
            )

            self.bias_mu = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_mu_init, 0.1)
            )
            self.bias_rho = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_rho_init, 0.1)
            )
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.bias_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample the weights and forward it

        # if the model is frozen, return frozen
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features), device=x.device)
            b_log_posterior = 0
            b_log_prior = 0

        # Get the complexity cost
        self.log_variational_posterior = (
            self.weight_sampler.log_posterior() + b_log_posterior
        )
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.linear(x, w, b)

    def forward_frozen(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the feedforward operation with the expected value for weight and biases
        """
        if self.bias:
            return F.linear(x, self.weight_mu, self.bias_mu)
        else:
            return F.linear(x, self.weight_mu, torch.zeros(self.out_features))