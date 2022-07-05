# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:46:16 2022

@author: jxr180022
"""
import time
import gym.spaces
from stable_baselines3 import PPO
from state_action_reward import *
# from stable_baselines.common import make_vec_env
from openDSSenv34 import openDSSenv34
# import json
# import datetime as dt
import torch
# from stable_baselines3.common.utils import set_random_seed
#from feedforwardPolicy import *
from stable_baselines3 import A2C, PPO
#from CustomPolicies import ActorCriticGCAPSPolicy
from DSS_Initialize import   *
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

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

        if len(observations["Unserved Energy"].shape) == 1:
            data_UE = observations["Unserved Energy"][:, None]
        else:
            data_UE = observations["Unserved Energy"]
        #print(observations["Unserved Energy"])
        #data_UE.shape(0)
        #observations['Adjacency'].shape(0)
        
        statevec = np.concatenate((data_UE,
                                    observations['NodeFeat(BusVoltage)'].flatten(1,2),
                                    observations['EdgeFeat(branchflow)'][:,:],
                                    observations['Adjacency'].flatten(1,2)), axis=1)
        statevec = np.array(statevec)
        statevec = th.from_numpy(statevec)

        return self.linear(statevec)