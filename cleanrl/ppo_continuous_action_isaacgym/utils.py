from typing import List

import numpy as np
import torch


def find_num_neuron_per_type(num_features: int, num_types: int) -> np.ndarray:
    features = int(num_features)
    types = int(num_types)

    # Calculate the average value of each element
    avg_value = int(features // types)
    # Initialize the array with elements of value avg_value
    arr = np.ones(types) * avg_value

    # Distribute the remaining features across the elements
    remainder = features - sum(arr)
    arr[: int(remainder)] += 1

    return arr.astype(np.int64)


def make_legs_longer(length_adjustment: float = 1.5) -> None:
    import os

    import brax

    brax_path = os.path.dirname(brax.__file__)
    new_xml = ant_at_scale(length_adjustment)

    with open(f"{brax_path}/envs/assets/ant.xml", "w") as f:
        f.write(new_xml)

    print(f"writing at {brax_path}")


def get_indices(pre_unit_types, n_pre_neurons, post_unit_types, n_post_neurons):
    pre_type_indices = torch.tile(
        torch.tensor(pre_unit_types[None, :, None]), (1, 1, n_post_neurons)
    )

    post_type_indices = torch.tile(
        torch.tensor(post_unit_types[None, :, None]), (1, 1, n_pre_neurons)
    )

    indices = torch.stack(
        (pre_type_indices, post_type_indices.permute(0, 2, 1)), axis=-1
    )

    return indices


def to1d(x, y, z, num_inp, num_out):
    return (z * num_inp * num_out) + (y * num_out) + x


def to1d_CNN(a, x, y, z, out_channel, in_channel, num_out, num_inp):
    return (
        (a * num_inp * in_channel * num_out)
        + (x * num_inp * num_out)
        + (y * num_inp)
        + z
    )


def gather2D(tensor, indices, indices_flat):
    old_shape = indices.shape[:-1]
    tensor = gather_nd_torch(tensor, indices, indices_flat, batch_dim=1)
    return torch.reshape(tensor, old_shape)


def gather_nd_torch(params, indices, gather_dims_flat, batch_dim=0):
    expand = batch_dim == 0
    if expand:
        params = torch.unsqueeze(params, 0)
        indices = torch.unsqueeze(indices, 0)
        batch_dim = 1

    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size()[batch_dim:-1]  # x
    org_indices = indices
    n_indices = np.cumprod(list(n_indices))[-1]
    n_pos = indices.size(-1)  # m
    try:
        last_dim = params.size()[batch_dim + n_pos :][0]
    except:
        last_dim = 1

    gathered = torch.index_select(
        params.flatten(),
        0,
        torch.tensor(gather_dims_flat.flatten()).to(params.flatten().device),
    )

    gathered = gathered.reshape(
        *batch_dims, *org_indices.size()[batch_dim:-1], last_dim
    )

    gathered = torch.squeeze(gathered)

    return gathered


def gather_nd_torch_dims_flat(params, indices, batch_dim=0):
    expand = batch_dim == 0
    if expand:
        params = torch.unsqueeze(params, 0)
        indices = torch.unsqueeze(indices, 0)
        batch_dim = 1

    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size()[batch_dim:-1]  # x
    org_indices = indices
    n_indices = np.cumprod(list(n_indices))[-1]
    n_pos = indices.size(-1)  # m
    try:
        last_dim = params.size()[batch_dim + n_pos :][0]
    except:
        last_dim = 1
    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)
    idx_arr = [[] for _ in range(n_pos + 1)]
    for i in range(batch_size):
        idx_arr[0].append([i for _ in range(len(indices[i][:, 0]))])
        for j in range(n_pos):
            idx_arr[j + 1].append(indices[i][:, j].numpy())

    gather_dims = idx_arr

    try:
        gather_dims_flat = to1d_CNN(
            np.array(gather_dims[1]),
            np.array(gather_dims[2]),
            np.array(gather_dims[3]),
            np.array(gather_dims[4]),
            (gather_dims[1][0][-1] + 1),
            (gather_dims[2][0][-1] + 1),
            (gather_dims[3][0][-1] + 1),
            (gather_dims[4][0][-1] + 1),
        )
    except:
        gather_dims_flat = to1d(
            np.array(gather_dims[2]),
            np.array(gather_dims[1]),
            np.array(gather_dims[0]),
            (gather_dims[1][-1] + 1),
            (gather_dims[2][-1] + 1),
        )

    return gather_dims_flat


def get_indices_CNN(n_neurons_arr, unit_types_arr):
    n_pre_neurons = n_neurons_arr[3]
    n_post_neurons = n_neurons_arr[2]

    n_pre_channels = n_neurons_arr[1]
    n_post_channels = n_neurons_arr[0]

    pre_unit_types = unit_types_arr[3]
    post_unit_types = unit_types_arr[2]
    pre_channel_types = unit_types_arr[1]
    post_channel_types = unit_types_arr[0]

    pre_type_indices = torch.tile(
        torch.tensor(pre_unit_types[None, None, None, :]),
        (1, n_post_channels, n_pre_channels, n_post_neurons, 1),
    )

    post_type_indices = torch.tile(
        torch.tensor(post_unit_types[None, None, :, None]),
        (1, n_post_channels, n_pre_channels, 1, n_pre_neurons),
    )

    pre_channel_indices = torch.tile(
        torch.tensor(pre_channel_types[None, :, None, None]),
        (1, n_post_channels, 1, n_post_neurons, n_pre_neurons),
    )

    post_channel_indices = torch.tile(
        torch.tensor(post_channel_types[:, None, None, None]),
        (1, 1, n_pre_channels, n_post_neurons, n_pre_neurons),
    )

    indices = torch.stack(
        (
            post_channel_indices,
            pre_channel_indices,
            post_type_indices,
            pre_type_indices,
        ),
        axis=-1,
    )

    return indices


def calculate_parameters(mlp: List[int]) -> int:
    # The number of parameters between two layers is the product of the
    # number of neurons in these two layers, plus the number of neurons
    # in the current layer (for bias terms).
    # There are no parameters for the first layer, so we start from the second layer.
    return sum([mlp[i - 1] * mlp[i] + mlp[i] for i in range(1, len(mlp))])


def calculate_compression_ratio(
    env,
    vanilla_policy_layers: List[int],
    vanilla_value_layers: List[int],
    number_of_cell_types: int,
) -> float:
    number_of_vanilla_policy_parameters = calculate_parameters(vanilla_policy_layers)
    number_of_vanilla_value_parameters = calculate_parameters(vanilla_value_layers)
    total_vanilla_parameters = (
        number_of_vanilla_policy_parameters + number_of_vanilla_value_parameters
    )
    compressed_policy_layers = [
        env.observation_space.shape[-1],
        number_of_cell_types,
        number_of_cell_types,
        1,
    ]
    compressed_value_layers = [
        env.observation_space.shape[-1],
        number_of_cell_types,
        number_of_cell_types,
        1,
    ]
    number_of_compressed_policy_parameters = calculate_parameters(
        compressed_policy_layers
    )
    number_of_compressed_value_parameters = calculate_parameters(
        compressed_value_layers
    )
    total_compressed_parameters = (
        number_of_compressed_policy_parameters + number_of_compressed_value_parameters
    )
    return total_vanilla_parameters / total_compressed_parameters


def ant_at_scale(scale=1.0):
    return f"""<mujoco model="ant">
      <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
      <option timestep="0.01" iterations="4" />
      <custom>
        <!-- brax custom params -->
        <numeric data="0.0 0.0 {scale * 0.55} 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
        <numeric data="1000" name="constraint_limit_stiffness"/>
        <numeric data="4000" name="constraint_stiffness"/>
        <numeric data="10" name="constraint_ang_damping"/>
        <numeric data="20" name="constraint_vel_damping"/>
        <numeric data="0.5" name="joint_scale_pos"/>
        <numeric data="0.2" name="joint_scale_ang"/>
        <numeric data="0.0" name="ang_damping"/>
        <numeric data="1" name="spring_mass_scale"/>
        <numeric data="1" name="spring_inertia_scale"/>
        <numeric data="15" name="solver_maxls"/>
      </custom>
      <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom contype="0" conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5"/>
      </default>
      <asset>
        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
      </asset>
      <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" size="40 40 40" type="plane"/>
        <body name="torso" pos="0 0 0.75">
          <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
          <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere" mass="1.0" />
          <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
          <body name="front_left_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 {scale * 0.2} {scale * 0.2} 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
            <body name="aux_1" pos="{scale * 0.2} {scale * 0.2} 0">
              <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
              <geom fromto="0.0 0.0 0.0 {scale * 0.2} {scale * 0.2} 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
              <body pos="{scale * 0.2} {scale * 0.2} 0">
                <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 {scale * 0.4} {scale * 0.4} 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
                <geom name="left_foot_geom" contype="1" pos="{scale * 0.4} {scale * 0.4} 0" size="0.08" type="sphere" mass="0.0"/>
              </body>
            </body>
          </body>
          <body name="front_right_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 -{scale * 0.2} {scale * 0.2} 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
            <body name="aux_2" pos="-{scale * 0.2} {scale * 0.2} 0">
              <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
              <geom fromto="0.0 0.0 0.0 -{scale * 0.2} {scale * 0.2} 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
              <body pos="-{scale * 0.2} {scale * 0.2} 0">
                <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -{scale * 0.4} {scale * 0.4} 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
                <geom name="right_foot_geom" contype="1" pos="-{scale * 0.4} {scale * 0.4} 0" size="0.08" type="sphere" mass="0.0"/>
              </body>
            </body>
          </body>
          <body name="back_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 -{scale * 0.2} -{scale * 0.2} 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
            <body name="aux_3" pos="-{scale * 0.2} -{scale * 0.2} 0">
              <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
              <geom fromto="0.0 0.0 0.0 -{scale * 0.2} -{scale * 0.2} 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
              <body pos="-{scale * 0.2} -{scale * 0.2} 0">
                <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -{scale * 0.4} -{scale * 0.4} 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
                <geom name="third_foot_geom" contype="1" pos="-{scale * 0.4} -{scale * 0.4} 0" size="0.08" type="sphere" mass="0.0"/>
              </body>
            </body>
          </body>
          <body name="right_back_leg" pos="0 0 0">
            <geom fromto="0.0 0.0 0.0 {scale * 0.2} -{scale * 0.2} 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
            <body name="aux_4" pos="{scale * 0.2} -{scale * 0.2} 0">
              <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
              <geom fromto="0.0 0.0 0.0 {scale * 0.2} -{scale * 0.2} 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
              <body pos="{scale * 0.2} -{scale * 0.2} 0">
                <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 {scale * 0.4} -{scale * 0.4} 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
                <geom name="fourth_foot_geom" contype="1" pos="{scale * 0.4} -{scale * 0.4} 0" size="0.08" type="sphere" mass="0.0"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
      </actuator>
    </mujoco>"""