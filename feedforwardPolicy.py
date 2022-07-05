from stable_baselines3.common.policies import BasePolicy
import torch as th
import gym
import math
import numpy as np
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from typing import NamedTuple
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
#from GCAPCN import GCAPCNFeatureExtractor

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: th.Tensor
    context_node_projected: th.Tensor
    glimpse_key: th.Tensor
    glimpse_val: th.Tensor
    logit_key: th.Tensor

    def __getitem__(self, key):
        if th.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)

class feedforwardPolicy(BasePolicy):

    def __init__(self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[th.nn.Module] = th.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        super(feedforwardPolicy, self).__init__(observation_space,
            action_space,
            #features_extractor_class,
            #features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            #squash_output=squash_output
            )
        
        self.statesize = 320
        self.actionsize = 15

        self.value_net = th.nn.Linear(self.statesize, 1)
        #self.features_extractor = GCAPCNFeatureExtractor()
        #self.agent_decision_context = th.nn.Linear(2,128)
        #self.agent_context = th.nn.Linear(2,128)
        #self.full_context_nn = th.nn.Linear(256, 128)
        self.L1 = th.nn.Linear(self.statesize,self.statesize)
        self.L2 = th.nn.Linear(self.statesize,self.actionsize)
        self.activation = th.nn.ReLU()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        #self.action_net = th.nn.Linear(128,20)
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

        #self.project_fixed_context = th.nn.Linear(128, 128, bias=False)
        #self.project_node_embeddings = th.nn.Linear(128, 3 * 128, bias=False)
        #self.project_out = th.nn.Linear(128, 128, bias=False)
        #self.n_heads = 8
        #self.tanh_clipping = 10.
        #self.mask_logits = True
        #self.temp = 1.0



    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, values, log_prob = self.forward(observation)
        return th.tensor([actions])

    def _build(self):
        pass

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        # values = self.value_net(latent_vf)
        #distribution, values = self.get_distribution(obs)
        distribution = self.action_dist.proba_distribution(action_logits=actions)
        values = self.value_net
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def forward(self, obs,  *args, **kwargs):

        # latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        # values = self.value_net(latent_vf)

        #distribution, values = self.get_distribution(obs)
        statevec = np.concatenate((obs['loss'],
                                  obs['NodeFeat(BusVoltage)'].flatten(),
                                  obs['EdgeFeat(branchflow)'].flatten(),
                                  obs['Adjacency'].flatten()),axis=None)
        statevec = th.from_numpy(statevec)
        values = self.value_net(statevec)
        actions = self.activation(self.L2(self.L1(statevec)))
        # actions = distribution.get_actions(deterministic=True)
        #actions = distribution.distribution.logits[0][0].argmax()
        # a2 = distribution.distribution.sample()
        # actions = a2[0,0]
        actions = th.softmax(actions,dim=-1)
        distribution = self.action_dist.proba_distribution(action_logits=actions)
        log_prob = distribution.log_prob(actions)
        for i in range(self.actionsize):
            if actions[i] >= 0.5:
                actions[i] == 1
            else:
                actions[i] == 0  
        return th.tensor([actions]), values, log_prob
    
    
    
    # def get_distribution(self, obs):

    #     # features, graph_embed = self.extract_features(obs)
    #     # latent_pi, values = self.context_extractor(graph_embed, obs)
    #     mean_actions = self.decode_action_probabilites(latent_pi,graph_embed, features, obs)[:,:,:obs['mask'].shape[1] - 1]
    #     # # mean_actions = self.action_net(latent_pi)
    #     # latent_sde = latent_pi
    #     # # if self.sde_features_extractor is not None:
    #     # #     latent_sde = self.sde_features_extractor(features)
    
    #     if isinstance(self.action_dist, DiagGaussianDistribution):
    #         distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std)
    #     elif isinstance(self.action_dist, CategoricalDistribution):
    #         # Here mean_actions are the logits before the softmax
    #         distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, MultiCategoricalDistribution):
    #         # Here mean_actions are the flattened logits
    #         distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, BernoulliDistribution):
    #         # Here mean_actions are the logits (before rounding to get the binary actions)
    #         distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, StateDependentNoiseDistribution):
    #         distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
    #     else:
    #         raise ValueError("Invalid action distribution")
    
    #     # values = self.value_net(latent_vf)
    #     return distribution, values