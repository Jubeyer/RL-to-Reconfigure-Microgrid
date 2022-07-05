import numpy as np
import gym
from stable_baselines3 import PPO
# from stable_baselines.common import make_vec_env
from openDSSenv import openDSSenv
#import json
#import datetime as dt
#import torch
#from stable_baselines3.common.utils import set_random_seed
from feedforwardPolicy import *
from stable_baselines3 import A2C

#from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


env = openDSSenv()


model = A2C(feedforwardPolicy, env).learn(total_timesteps=1000)
#model.learn(total_timesteps=2000)

obs = env.reset()

log_dir = "."
model.save(log_dir + "r1")
model = PPO.load(log_dir + "r1", env=env)

model.policy.mask_logits = True
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
# print(env.total_distance_travelled)
    # env.render()
# if __name__ == '__main__':
#
#     env = mTSPEnv(
#         n_locations = 21,
#         n_agents = 5
#     )
#     action_sequence = [3,4,1,10,2,8,6,9,7,5,11,14,12,16,13,15,20,19,17,18]
#     i = 0
#     while not env.done:
#         action =  action_sequence[i]
#         i += 1
#         env.step(action)