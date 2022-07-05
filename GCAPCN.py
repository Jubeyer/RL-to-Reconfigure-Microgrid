"""
Author: Steve Paul 
Date: 11/22/21 """
import torch
import numpy as np
from torch import nn
import math
from DSS_Initialize import  *


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input



class GCAPCNFeatureExtractor(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=64,
                 n_p=1,
                 node_dim=3,
                 n_K=1
                 ):
        super(GCAPCNFeatureExtractor, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = nn.BatchNorm1d(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.switch_encoder = torch.nn.Linear(2 * n_dim, n_dim)

        self.activ = torch.nn.LeakyReLU()

    def forward(self, data, mask=None):
        #print(data["NodeFeat(BusVoltage)"].shape)
        #print("yoooooooooooooooo")
        X = data['NodeFeat(BusVoltage)']
        # X = torch.cat((data['loc'], data['deadline']), -1)
        num_samples, num_locations, _ = X.size()
        #A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
        #    (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
       # A[A != A] = 0
        A = data["Adjacency"]
        #print(A.shape)
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


        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        # F1 = self.activ(F1)


        F_final = self.W_F(F1)

        h = F_final #torch.cat((init_depot_embed, F_final), 1)
        switch_embeddings = self.switch_encoder(
            torch.cat((h[:, switch_bus_map[:, 0], :], h[:, switch_bus_map[:, 1], :]), -1))

        return (
            switch_embeddings,  # (batch_size, graph_size, embed_dim)
            F_final.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )