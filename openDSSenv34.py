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

#from valid_action_search import ACTION_VALID

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.WARNING)

# GENS=DSSCktobj.dssCircuit.VSources.AllNames
# DERS=[i for i in GENS if i not in "source"]
# SWS=[j for j in DSSCktobj.dssCircuit.Lines.AllNames if "sw" in j]
class openDSSenv34(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        print("Initializing Microgrid env with sectionalizing and tie switches")
        self.DSSCktObj, self.G_init, conv_flag = initialize()  # the DSSCircuit is set up and initialized
        # Set up action and observation space variables
        #n_actions = len(sectional_swt) + len(tie_swt)  # the switching actions
        # Commented on 06/03/2022
        #GENS=DSSCktobj.dssCircuit.VSources.AllNames
        #DERS=[i for i in GENS if i not in "source"]
        
        n_actions=len(SWS+DERS)
        #print(n_actions)
        
        #self.action_space = spaces.Discrete(len(ACTION_VALID))
        # 
        # Jubeyer: Only thing I need to do here is to import a global variable which contains all the feasible actions
#--------------------------------------------------Important Notes----------------------------------
# For continuous action space they use Box, e.g., self.action_space = spaces.Box(low = action_range[0], high = action_range[1],dtype = np.float32)        
# We should also think of adding seed if we want to reproduce the output.
# The link: https://www.analyticsvidhya.com/blog/2021/08/creating-continuous-action-bot-using-deep-reinforcement-learning/
# But the main challenge is to have four continuous action space for four generators that we want to control, but how to combine them altogether is still unclear!
# There is an example of combining two action spaces but both of them were Box types. 
# link: https://docs.ray.io/en/latest/rllib/rllib-models.html
# The sources are confusing; some says it's not possible to mix these two types some suggested to use a dictionary like here: https://github.com/openai/gym/issues/1482
# So I have decided to move with this {"buy": Discrete(<number of different stocks), "amount": Box()}
# tentative script would be self.action_space=spaces.
        #self.action_space = spaces.MultiBinary(n_actions)
        # for dealing with all continous actions we take
        high = np.array([1] * n_actions) #need to replace the 21 with some generic format☻
        low = np.array([0] * n_actions)
        self.action_space = spaces.Box(low, high, dtype=np.float32) 
        
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
        
        # This section is modified for multiple line outages
        
        # For normal operation commented out---------------------------
        #Candidate_Lines=DSSCktobj.dssCircuit.Lines.AllNames
        ##L_OUT='L24'#random.choice(Candidate_Lines)#considering one single outage at a time
        #L_OUT=random.choices(Candidate_Lines,k=random.randint(1,len(Candidate_Lines)))
        #--------------------------------------------------------------------------------------
        
        # Older version
        #Candidate_Lines=['L7','L9','L15','L16','L18','L19','L21','L22','L23','L24']
        
        # Approach 1 -------------randomly set the multiplication factor(same for all loads)
        # loadmulti=round(random.uniform(umin,umax),2) #random generation of load multiplier
        # i=self.DSSCktObj.dssLoads.First
        # while i>0:
        #       self.DSSCktObj.dssLoads.kW = round(self.DSSCktObj.dssLoads.kW * loadmulti,2)
        #       i=self.DSSCktObj.dssLoads.Next
        # self.DSSCktObj.dssSolution.Solve() #solving and setting the new load

        conv_flag = 0  # Flag to indicate convergence (if 0 not converged, 1 converges)
        # Approach 2 -------------randomly set the all the loads in the network to a new value
        while conv_flag == 0:
            loadfactors = np.random.uniform(umin, umax, len(self.DSSCktObj.dssLoads.AllNames))
            i = self.DSSCktObj.dssLoads.First
            while i > 0:
                self.DSSCktObj.dssLoads.kW = round(self.DSSCktObj.dssLoads.kW * loadfactors[i - 1], 2)
                i = self.DSSCktObj.dssLoads.Next
                
            # This section is modified for allowing multiple line outages    
            #L_OUT=random.choice(Candidate_Lines)#considering one single outage at a time
            #-----------------------------------------------------------------------------
            # if len(L_OUT)>1:
            #     for i in L_OUT:
            #         DSSCktobj.dssText.command='open ' +'line.'+ i +' term=1'
            # else:
            #     DSSCktobj.dssText.command='open ' +'line.'+ L_OUT +' term=1'
           # -----------------------------------------------------------------------------
            
            
            
            #♥self.DSSCktObj.dssText.command='open ' +'line.'+ L_OUT +' term=1'
            
            self.DSSCktObj.dssSolution.Solve()  # solving and setting the new load
            if self.DSSCktObj.dssSolution.Converged:
                conv_flag = 1
                #print(conv_flag)
            else:
                conv_flag = 0
        
        
        
        # To observe if the load is changed
        # self.All_Loads={}
        # i=self.DSSCktObj.dssLoads.First
        # while i>0:
        #       self.All_Loads[self.DSSCktObj.dssLoads.Name]=self.DSSCktObj.dssLoads.kW
        #       i=self.DSSCktObj.dssLoads.Next
        # self.All_Loads

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
