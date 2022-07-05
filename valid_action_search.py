# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:27:33 2022

@author: jxr180022
"""

from  DSS_CircuitSetup import*
from DSS_Initialize import initialize, n_actions
#from state_action_reward import graphScenario
# def graphScenario(DSSCktObj,G_init):
#     #Input: The DSS Circuit object representing system state and initial circuit graph
#     #Output: The graph representation of the instance with action implemented
#     G_scenario=G_init.copy() #create copy of the initial circuit graph for scenario
#     Sw_dictlist= switchInfo(DSSCktObj,G_init)
#     for e in G_init.edges(data=True):
#         for s in Sw_dictlist:
#             if (s['edge name'].casefold()==e[2]['label'].casefold()) and (s['status']==0): # Switches with open status
#                  G_scenario.remove_edge(e[0],e[1])
#                  #remove open lines or switches form the graph to get the structure for the instance
#     return G_scenario
# def Topol_Constr(DSSCktObj,G_scenario):
#     #Input: The DSS Circuit object which contains system state and the corresponding graph scenario
#     #Output: The constraint violation penalty
#     M=200 # a large number assigned for penalty
#     no_nodes=len(G_scenario.nodes())
#     no_edges=len(G_scenario.edges())
#     connectn=nx.is_connected(G_scenario)
#     if (no_edges == no_nodes-1) and (connectn==True):
#         topol_viol=0
#     else:
#         topol_viol = M * (int(not connectn) + (abs(no_edges - no_nodes + 1)))#changed for trial
#         #topol_viol=M
#     return topol_viol

def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points


def all_actions(n_actions):
    a = [0,1]
    SW_CONFIG=cartesian_coord(*n_actions*[a])
    #total_actions=2**n_actions
    return SW_CONFIG #,total_actions

#def valid_actions(n_actions):
ACTION_VALID=[]
SW_CONFIG=all_actions(n_actions)

for s in SW_CONFIG:
    #count=0
    DSSCktObj, G_init, conv_flag=initialize()
    action=s
    i=DSSCktObj.dssCircuit.SwtControls.First #1
    while (i>0):
           Swobj=DSSCktObj.dssCircuit.SwtControls.SwitchedObj
           #print(Swobj)
           DSSCktObj.dssCircuit.SetActiveElement(Swobj)
           if action[i-1]==0: # i starts from 1 in DSS #if action is 0
               DSSCktObj.dssText.command='open ' + Swobj +' term=1'       #switching the line open
           else:
               DSSCktObj.dssText.command='close ' + Swobj +' term=1'      #switching the line close
           i=DSSCktObj.dssCircuit.SwtControls.Next
    #DSSCktObj.dssCircuit.Solution.Solve()
    # Then check how many loops are there
    #G_Scenario=graphScenario(DSSCktObj, G_init)
    #topol_viol=Topol_Constr(DSSCktObj, G_scenario)    
    DSSTopology=DSSCktObj.dssCircuit.Topology
    NumLoops=DSSTopology.NumLoops
    NumIsolatedLd = DSSTopology.NumIsolatedLoads
    Temp_Looped_Pair=DSSTopology.AllLoopedPairs
    # print("For the switcing combination of: ",s)
    # print("Total Number of Loops:", NumLoops)
    # print("Total Number of Isolated Loads:", NumIsolatedLd)
    # print(DSSTopology.AllLoopedPairs)
    #if NumIsolatedLd==False and NumLoops==3:
    if NumIsolatedLd==0 and NumLoops==3 and any("Line" in value for value in Temp_Looped_Pair)==False :    
    #if NumLoops==3:
        ACTION_VALID.append(s)
    #count=count+1
    #return ACTION_VALID
        
    
    
