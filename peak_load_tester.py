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

class CustomGNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 n_layers=2,
                 n_dim=256,
                 n_p=1,
                 node_dim=3,
                 n_K=1,
                 ):
        super(CustomGNN, self).__init__(observation_space, features_dim)
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = torch.nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = torch.nn.Linear(2, n_dim)

        self.W_L_1_G1 = torch.nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = torch.nn.BatchNorm1d(n_dim * n_p)

        self.W_F = torch.nn.Linear(n_dim * n_p, features_dim)
        self.full_context_nn = th.nn.Linear(41, features_dim)
        self.switch_encoder = torch.nn.Linear(2 * features_dim, features_dim)

        self.activ = torch.nn.Tanh()

    def forward(self, data):
        X = data['NodeFeat(BusVoltage)']
        # X = torch.cat((data['loc'], data['deadline']), -1)
        num_samples, num_locations, _ = X.size()
        # A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
        #    (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        # A[A != A] = 0
        A = data["Adjacency"]
        # print(A.shape)
        D = torch.mul(torch.eye(num_locations).expand((num_samples, num_locations, num_locations)),
                      (A.sum(-1))[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)

        # K = 3
        L = D - A

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :]
                                          ),
                                         -1))

        F1 = g_L1_1  # torch.cat((g_L1_1), -1)
        # F1 = self.activ(F1)

        F_final = self.W_F(F1)

        h = F_final  # torch.cat((init_depot_embed, F_final), 1)

        # return (
        #     h,  # (batch_size, graph_size, embed_dim)
        #     h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        # )
        switch_embeddings = self.switch_encoder(
            torch.cat((h[:, switch_bus_map[:, 0], :], h[:, switch_bus_map[:, 1], :]), -1))
        context = self.full_context_nn(th.cat((data["loss"], data["EdgeFeat(branchflow)"]), -1))
        return self.activ(switch_embeddings.mean(dim=1)) + context

env = openDSSenv34()
log_dir = "."
model = PPO.load(log_dir + "r1_34_bus_gcaps_with_entropy_05", env=env)

obs = env.test_func()
obs = {key: torch.as_tensor([_obs]) for (key, _obs) in obs.items()}
obs['loss'] = torch.as_tensor([[obs['loss']]])
obs['TopologicalConstr'] = torch.as_tensor([[obs['TopologicalConstr']]])
obs['VoltageViolation'] = torch.as_tensor([[obs['VoltageViolation']]])
obs['FlowViolation'] = torch.as_tensor([[obs['FlowViolation']]])

action, values, log_probs = model.policy.forward(obs)

new_obs = env.step(action[0,:])

ft = 0