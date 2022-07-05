"""
Author: Steve Paul 
Date: 11/22/21 """
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
from GCAPCN import GCAPCNFeatureExtractor

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

class ActorCriticGCAPSPolicy(BasePolicy):

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
        super(ActorCriticGCAPSPolicy, self).__init__(observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output)
        n_dim = 64
        # context_vals_layers = [th.nn.Linear(n_dim, n_dim), th.nn.Tanh(), th.nn.Linear(n_dim, n_dim), th.nn.Tanh()]
        context_vals_layers = [th.nn.Linear(n_dim, 2*n_dim),th.nn.Linear(2*n_dim, n_dim)]
        self.value_net_context = th.nn.Sequential(*context_vals_layers)
        self.value_net = th.nn.Linear(n_dim, 1)
        self.features_extractor = GCAPCNFeatureExtractor()
        context_layers = [th.nn.Linear(41, n_dim), th.nn.LeakyReLU()]
        self.full_context_nn = th.nn.Sequential(*context_layers)
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        dist_kwargs = None
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }


        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self.project_fixed_context = th.nn.Linear(n_dim, n_dim, bias=False)
        self.project_node_embeddings = th.nn.Linear(n_dim, 3 * n_dim, bias=False)
        self.project_out = th.nn.Linear(n_dim, n_dim, bias=False)
        self.n_heads = 8
        self.tanh_clipping = 20.
        self.mask_logits = True
        self.temp = 1.0
        self.switch_encoder = th.nn.Linear(2*n_dim,n_dim)

        # self.simple_decoder = th.nn.LogSoftmax(*th.nn.Sequential(*[ th.nn.Linear(2*n_dim,1), th.nn.Sigmoid()]))


    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, values, log_prob = self.forward(observation, deterministic)
        return actions

    def _build(self):
        pass

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        # values = self.value_net(latent_vf)
        distribution, values = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def forward(self, obs, deterministic=False, *args, **kwargs):

        # latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        # values = self.value_net(latent_vf)

        distribution, values = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        # actions = distribution.distribution.logits
        # actions=(distribution.distribution.mean > 0.25).to(th.float32)

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = th.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # if self.mask_inner:
        #     assert self.mask_logits, "Cannot mask inner without masking logits"
        #     compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = th.matmul(th.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = (th.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(
            final_Q.size(-1)))  ### steve has made a change here (normalizing)

        ## From the logits compute the probabilities by clipping, masking and softmax
        # if self.tanh_clipping > 0:
        #     logits = th.tanh(logits) * self.tanh_clipping
        # if self.mask_logits:
        #    logits[th.tensor(mask[:,1:,:].reshape(logits.shape), dtype=th.bool)] = -10e5

        return logits, glimpse.squeeze(-2)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _get_attention_node_data(self, fixed):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def decode_action_probabilites(self, latent_pi, graph_embed, features, obs, num_steps=1):
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(features[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        fixed = AttentionModelFixed(features, fixed_context, *fixed_attention_node_data)
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        query = fixed.context_node_projected + latent_pi
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K)

        # log_p = th.log_softmax(log_p / self.temp, dim=-1)

        return log_p

    def get_distribution(self, obs):

        switch_feats, graph_embed = self.extract_features(obs)
        # sw_ind = np.array([[5,6],[7,11],[10,13],[3,11]])
        # switch_feats = self.switch_encoder(th.cat((features[:, switch_bus_map[:,0],:], features[:, switch_bus_map[:,1],:]),-1))
        latent_pi, values_context = self.context_extractor(graph_embed, obs)
        mean_actions = self.decode_action_probabilites(latent_pi, graph_embed, switch_feats, obs)[:, 0, :]
        # mean_actions = self.action_net(latent_pi)
        latent_sde = latent_pi
        # if self.sde_features_extractor is not None:
        #     latent_sde = self.sde_features_extractor(features)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution = self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

        # values = self.value_net(latent_vf)
        return distribution, self.value_net(values_context)

    def context_extractor(self, graph_embed, observations):

        context = graph_embed[:, None, :] \
                  + \
                  self.full_context_nn(
                      th.cat((observations["loss"], observations["EdgeFeat(branchflow)"]), -1)[:, None, :])
        # statevec = np.concatenate((observations['loss'][0, :],
        #                            observations['NodeFeat(BusVoltage)'][0, :, :].flatten(),
        #                            observations['EdgeFeat(branchflow)'][0, :].flatten(),
        #                            observations['Adjacency'][0, :, :].flatten()), axis=None)
        # statevec = np.array([statevec])
        # statevec = th.from_numpy(statevec)
        # # print(context.shape)
        return context, self.value_net_context(context)
    # def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
    #     mean_actions = self.action_net(latent_pi)
    #
    #     if isinstance(self.action_dist, DiagGaussianDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std)
    #     elif isinstance(self.action_dist, CategoricalDistribution):
    #         # Here mean_actions are the logits before the softmax
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, MultiCategoricalDistribution):
    #         # Here mean_actions are the flattened logits
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, BernoulliDistribution):
    #         # Here mean_actions are the logits (before rounding to get the binary actions)
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, StateDependentNoiseDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
    #     else:
    #         raise ValueError("Invalid action distribution")