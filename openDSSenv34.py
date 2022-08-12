import gym
import random

import torch
import win32com.client
from gym import spaces
import numpy as np
import logging

from DSS_Initialize import *
from DSS_CircuitSetup import *
from state_action_reward import *

from gym.utils import seeding

#from valid_action_search import ACTION_VALID # for action_space shrinking

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.WARNING)


class openDSSenv34(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        print("Initializing Microgrid env with sectionalizing and tie switches")
        self.DSSCktObj, self.G_init, conv_flag = initialize()  # the DSSCircuit is set up and initialized
        # Set up action and observation space variables
        n_actions = len(sectional_swt) + len(tie_swt)  # the switching actions
        # This is for the normal operation action space shrinking technique       
        #self.action_space = spaces.Discrete(len(ACTION_VALID))
        # 
        
        self.action_space = spaces.MultiBinary(n_actions)
       
        
        
        ####################---------------------------------------------------#######################       
        self.observation_space = spaces.Dict({"loss": spaces.Box(low=0, high=600*mult_constant, shape=(1,), dtype=np.float64),
                                              "NodeFeat(BusVoltage)": spaces.Box(low=0, high=2*mult_constant,
                                                                                 shape=(len(self.G_init.nodes()), 3), dtype=np.float64),
                                              "EdgeFeat(branchflow)": spaces.Box(low=0, high=2*mult_constant,
                                                                                 shape=(len(self.G_init.edges()),), dtype=np.float64),
                                              "Adjacency": spaces.Box(low=0, high=1, shape=(
                                              len(self.G_init.nodes()), len(self.G_init.nodes()))),
                                              "TopologicalConstr": spaces.Box(low=0, high=200, shape=(1,)),
                                              "VoltageViolation": spaces.Box(low=0, high=200, shape=(1,)),
                                              "FlowViolation": spaces.Box(low=0, high=200, shape=(1,)),
                                              "Convergence": spaces.Box(low=0, high=200, shape=(1,))
                                              , "Unserved Energy": spaces.Box(low=0, high=200, shape=(1,))
                                              })
        print('Env initialized')

    def step(self, action):
        # Getting observation before action is executed
        observation = get_state(self.DSSCktObj, self.G_init)  # function to get state of the network
        # Executing the switching action
        self.DSSCktObj = take_action(self.DSSCktObj, action)  # function to implement the action
        # Getting observation after action is taken
        obs_post_action = get_state(self.DSSCktObj, self.G_init)
        reward = get_reward(obs_post_action)  # function to calculate reward
        #print(reward)
        done = True
        info = {"is_success": done,
                "episode": {
                    "r": reward,
                    "l": 1
                }
                }
        logging.info('Step success')

        obs_post_action["EdgeFeat(branchflow)"] = obs_post_action["EdgeFeat(branchflow)"]*mult_constant
        obs_post_action["NodeFeat(BusVoltage)"] = obs_post_action["NodeFeat(BusVoltage)"] * mult_constant
        obs_post_action["loss"] = obs_post_action["loss"] * mult_constant

        return obs_post_action, reward, done, info

    def reset(self):
        # In reset function I ensure that the beginning state at each episode has converged powerflow
        logging.info('resetting environment...')
        self.DSSCktObj, self.G_init, conv_flag = initialize()  # initial set up
        # Different Load Configurations
        umin = 0.1  # minimum load multiplication factor
        umax = 2.0
        
        # This section is for checking single line outage
        
        # For normal operation comment this out---------------------------
        
        L_OUT='L24' #considering one single outage at a time
       
        #--------------------------------------------------------------------------------------
        
        # Older version
        #Candidate_Lines=['L7','L9','L15','L16','L18','L19','L21','L22','L23','L24']
        
       

        conv_flag = 0  # Flag to indicate convergence (if 0 not converged, 1 converges)
        # Approach 2 -------------randomly set the all the loads in the network to a new value
        while conv_flag == 0:
            loadfactors = np.random.uniform(umin, umax, len(self.DSSCktObj.dssLoads.AllNames))
            i = self.DSSCktObj.dssLoads.First
            while i > 0:
                self.DSSCktObj.dssLoads.kW = round(self.DSSCktObj.dssLoads.kW * loadfactors[i - 1], 2)
                i = self.DSSCktObj.dssLoads.Next
                
                       
            self.DSSCktObj.dssText.command='open ' +'line.'+ L_OUT +' term=1'
            
            self.DSSCktObj.dssSolution.Solve()  # solving and setting the new load
            if self.DSSCktObj.dssSolution.Converged:
                conv_flag = 1
               
            else:
                conv_flag = 0
        
        
        
       

        logging.info("reset complete\n")
        obs = get_state(self.DSSCktObj, self.G_init)
        obs["EdgeFeat(branchflow)"] = obs["EdgeFeat(branchflow)"] * mult_constant
        obs["NodeFeat(BusVoltage)"] = obs["NodeFeat(BusVoltage)"] * mult_constant
        obs["loss"] = obs["loss"] * mult_constant
        return obs

    # # This function can be used for testing
    def test_func(self):
        # The default load multiplication factor is 1 use that for testing
        loadfactors = np.random.uniform(0.5,1.5, len(self.DSSCktObj.dssLoads.AllNames))
        self.DSSCktObj, self.G_init, conv_flag=initialize() #initial set up
        self.DSSCktObj.dssSolution.Solve() #solving and setting the new load
        i = self.DSSCktObj.dssLoads.First
        while i > 0:
            self.DSSCktObj.dssLoads.kW = round(self.DSSCktObj.dssLoads.kW*loadfactors[i-1],2)
            i = self.DSSCktObj.dssLoads.Next
        self.DSSCktObj.dssSolution.Solve()
        obs = get_state(self.DSSCktObj, self.G_init)
        obs["EdgeFeat(branchflow)"] = obs["EdgeFeat(branchflow)"] * mult_constant
        obs["NodeFeat(BusVoltage)"] = obs["NodeFeat(BusVoltage)"] * mult_constant
        obs["loss"] = obs["loss"] * mult_constant
        return obs

    def render(self, mode='human', close=False):
        pass
    
    ##########################################################
    def new_test_func(self):
        # The default load multiplication factor is 1 use that for testing
        self.DSSCktObj, self.G_init, conv_flag=test_initialize() #initial set up
        self.DSSCktObj.dssSolution.Solve() #solving and setting the new load
        obs = get_state(self.DSSCktObj, self.G_init)
        return obs, self.DSSCktObj, self.G_init
