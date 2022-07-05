import numpy as np
import gym
from stable_baselines3 import PPO
# from stable_baselines.common import make_vec_env
from openDSSenv34 import openDSSenv34
# import json
# import datetime as dt
import torch
# from stable_baselines3.common.utils import set_random_seed
from feedforwardPolicy import *
from stable_baselines3 import A2C, PPO
from CustomPolicies import ActorCriticGCAPSPolicy
from DSS_Initialize import   *


class CustomNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        from torch import nn
        super(CustomNN, self).__init__(observation_space, features_dim)

        n_flatten = 1521
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):

        if len(observations["loss"].shape) == 1:
            data_loss = observations["loss"][:, None]
        else:
            data_loss = observations["loss"]

        statevec = np.concatenate((data_loss,
                                   observations['NodeFeat(BusVoltage)'].flatten(1, 2),
                                   observations['EdgeFeat(branchflow)'][:, :],
                                   observations['Adjacency'].flatten(1, 2)), axis=1)
        statevec = np.array(statevec)
        statevec = th.from_numpy(statevec)

        return self.linear(statevec)

def learning_rate_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:

        # return max((progress_remaining**2) * initial_value, 0.0002)

        return  initial_value
    return func

env = openDSSenv34()
rms_prop_eps = 1e-5
policy_kwargs = dict(
    features_extractor_class=CustomNN,
    features_extractor_kwargs=dict(features_dim=128),
    activation_fn=torch.nn.Tanh,
    net_arch=[dict(vf=[128,128])]
)




observations = env.test_func()

log_dir = "."
# model.save(log_dir + "r1_34_bus")
model = PPO.load(log_dir + "r1_34_bus_mlp_with_entropy_05_multi_env", env=env)

for i in range(10):
    observations['loss'] = np.float32(np.reshape(observations['loss'], (1,)))
    observations['TopologicalConstr'] = np.reshape(observations['TopologicalConstr'], (1,))
    observations['VoltageViolation'] = np.reshape(observations['VoltageViolation'], (1,))
    observations['FlowViolation'] = np.reshape(observations['FlowViolation'], (1,))
    observations['Convergence'] = np.reshape(observations['Convergence'], (1,))
    observations['EdgeFeat(branchflow)'] = np.float32(observations["EdgeFeat(branchflow)"])
    action, _states = model.predict(observations, deterministic=True)
    observations, rewards, dones, info = env.step(action)
    observations = env.test_func()

