import gym
import random
import win32com.client
from gym import spaces
import numpy as np
import logging

from DSS_Initialize import *
from DSS_CircuitSetup import *
from state_action_reward import *

from gym.utils import seeding

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.WARNING)


class openDSSenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print("Initializing 13-bus env with sectionalizing and tie switches")
        self.DSSCktObj, self.G_init = initialize()  # the DSSCircuit is set up and initialized
        # Set up action and observation space variables
        n_actions = len(sectional_swt) + len(tie_swt)  # the switching actions
        self.action_space = spaces.MultiBinary(n_actions)
        self.observation_space = spaces.Dict({"loss": spaces.Box(low=0, high=2, shape=(1,)),
                                              "NodeFeat(BusVoltage)": spaces.Box(low=0, high=2,
                                                                                 shape=(len(self.G_init.nodes()), 3)),
                                              "EdgeFeat(branchflow)": spaces.Box(low=0, high=2,
                                                                                 shape=(len(self.G_init.edges()),)),
                                              "Adjacency": spaces.Box(low=0, high=1,
                                                                      shape=(len(self.G_init.nodes()), len(self.G_init.nodes()))),
                                              "TopologicalConstr": spaces.Box(low=0, high=10000, shape=(1,)),
                                              "VoltageViolation": spaces.Box(low=0, high=1000, shape=(1,)),
                                              "FlowViolation": spaces.Box(low=0, high=1000, shape=(1,))
                                              })
        print('Env initialized')

    def step(self, action):
        # Getting observation before action is executed
        observation = get_state(self.DSSCktObj,self. G_init)  # function to get state of the network
        # Executing the switching action
        self.DSSCktObj = take_action(self.DSSCktObj, action)  # function to implement the action
        self.DSSCktObj.dssSolution.Solve()  # Solve Circuit
        # Getting observation after action is taken
        obs_post_action = get_state(self.DSSCktObj, self.G_init)
        reward = get_reward(obs_post_action)  # function to calculate reward
        done = True
        info = {}
        logging.info('Step success')
        return observation, reward, done, info

    def reset(self):
        logging.info('resetting environment...')
        self.DSSCktObj, self.G_init = initialize()  # initial set up
        self.DSSCktObj.dssSolution.Solve()  # Solve Circuit
        # Different Load Configurations
        umin = 0.1  # minimum load multiplication factor
        umax = 2.0

        # Approach 1 -------------randomly set the multiplication factor(same for all loads)
        # loadmulti=round(random.uniform(umin,umax),2) #random generation of load multiplier
        # i=self.DSSCktObj.dssLoads.First
        # while i>0:
        #       self.DSSCktObj.dssLoads.kW = round(self.DSSCktObj.dssLoads.kW * loadmulti,2)
        #       i=self.DSSCktObj.dssLoads.Next
        # self.DSSCktObj.dssSolution.Solve() #solving and setting the new load

        # Approach 2 -------------randomly set the all the loads in the network to a new value
        loadfactors = np.random.uniform(umin, umax, len(self.DSSCktObj.dssLoads.AllNames))
        i = self.DSSCktObj.dssLoads.First
        while i > 0:
            self.DSSCktObj.dssLoads.kW = round(self.DSSCktObj.dssLoads.kW * loadfactors[i - 1], 2)
            i = self.DSSCktObj.dssLoads.Next
        self.DSSCktObj.dssSolution.Solve()  # solving and setting the new load

        # To observe if the load is changed
        # self.All_Loads={}
        # i=self.DSSCktObj.dssLoads.First
        # while i>0:
        #       self.All_Loads[self.DSSCktObj.dssLoads.Name]=self.DSSCktObj.dssLoads.kW
        #       i=self.DSSCktObj.dssLoads.Next
        # self.All_Loads

        logging.info("reset complete\n")
        obs = get_state(self.DSSCktObj,  self.G_init)
        return obs

    # # This function can be used for testing
    def test_func(self):
        # The default load multiplication factor is 1 use that for testing
        self.DSSCktObj,G_init=initialize() #initial set up
        self.DSSCktObj.dssSolution.Solve() #solving and setting the new load
        obs = get_state(self.DSSCktObj, G_init)
        return obs

    def render(self, mode='human', close=False):
        pass
