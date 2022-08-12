"""
In this file the functions to evaluate the state, reward are defined and also the action is implemented
"""
import win32com.client
import os
import numpy as np
import networkx as nx
from  DSS_CircuitSetup import*
import matplotlib.pyplot as plt
#################---------------------------------------------------------------
# This section is for the action_space shrinking technique implementation
#from valid_action_search import ACTION_VALID

#####################-----------------------------------------------------------

sectional_swt=[]
tie_swt=[]
DER=[]
FolderName=os.path.dirname(os.path.realpath("__file__"))
DSSfile=r""+ FolderName+ "/ieee34Mod1.dss"
DSSCktobj=CktModSetup(DSSfile,sectional_swt,tie_swt,DER)
GENS=DSSCktobj.dssCircuit.Generators.AllNames
DERS=[i for i in GENS if i not in "source"]
SWS=[j for j in DSSCktobj.dssCircuit.Lines.AllNames if "sw" in j]
#########################-------------------------------------------------------

# Get the nodes of the switches from this function---includes all the details of switches
def switchInfo(DSSCktObj,G_init):
    #Input the DSSCircuitobject and initial circuit graph
    #Returns: A list of dictionaries which contains 
    #         The name of the switch.
    #         The associated line which is the edge label
    #         The from bus and to bus of the line housing the switch
    #         The status of the switch in the DSSCircuit object passed as system state
    
    AllSwitches=[]
    i=DSSCktObj.dssCircuit.SwtControls.First
    while i>0:
          name=DSSCktObj.dssCircuit.SwtControls.Name
          line=DSSCktObj.dssCircuit.SwtControls.SwitchedObj
          br_obj=Branch(DSSCktObj,line)
          from_bus=br_obj.bus_fr
          to_bus=br_obj.bus_to
          DSSCktObj.dssCircuit.SetActiveElement(line)
          if(DSSCktObj.dssCircuit.ActiveCktElement.IsOpen(1,0)):
             sw_status=0
          else:
             sw_status=1
          AllSwitches.append({'switch name':name,'edge name':line, 'from bus':from_bus.split('.')[0], 'to bus':to_bus.split('.')[0], 'status':sw_status})
          i=DSSCktObj.dssCircuit.SwtControls.Next
    return AllSwitches     

             
# To form the graph for the DSS circuit---remove open lines and consider only closed lines in topology     
def graphScenario(DSSCktObj,G_init):
    #Input: The DSS Circuit object representing system state and initial circuit graph
    #Output: The graph representation of the instance with action implemented
    G_scenario=G_init.copy() #create copy of the initial circuit graph for scenario
    Sw_dictlist= switchInfo(DSSCktObj,G_init)
    for e in G_init.edges(data=True):
        for s in Sw_dictlist:
            if (s['edge name'].casefold()==e[2]['label'].casefold()) and (s['status']==0): # Switches with open status
                 G_scenario.remove_edge(e[0],e[1])
                 #remove open lines or switches form the graph to get the structure for the instance
    return G_scenario

    
def get_state(DSSCktObj,G_init):
    #Input: object of type DSSObj.ActiveCircuit (COM interface for OpenDSS Circuit)
    #Returns: dictionary of circuit loss, bus voltage, branch powerflow,..
    G_scenario=graphScenario(DSSCktObj,G_init)
    node_list=list(G_init.nodes())
    Adj_mat=nx.adjacency_matrix(G_scenario,nodelist=node_list)
    # Extracting pu loss for the DSS circuit object
    DSSCktObj.dssTransformers.First
    KVA_base=DSSCktObj.dssTransformers.kva
    P_loss=(DSSCktObj.dssCircuit.Losses[0])/(1000*KVA_base)
    Q_loss=(DSSCktObj.dssCircuit.Losses[1])/(1000*KVA_base)
    # P_loss=(DSSCktObj.dssCircuit.Losses[0])/(1000)
    # #print(P_loss)
    # Q_loss=(DSSCktObj.dssCircuit.Losses[1])/(1000)
    # Extracting the pu node voltages at all buses
    Vmagpu=[]
    nodes_conn=[]
    for b in node_list:
        V=Bus(DSSCktObj,b).Vmag
        Vmagpu.append(V)
        nodes_conn.append(Bus(DSSCktObj,b).nodes)
    # Extracting the pu average branch currents(also includes the open branches)
    I_flow=[]
    for e in G_init.edges(data=True):
        branchname=e[2]['label']
        I=Branch(DSSCktObj, branchname).Cap
        I_flow.append(I)

    # The convergence test and violation penalty
    if DSSCktObj.dssSolution.Converged:
       conv_flag=1
       Conv_const=0
    else:
       conv_flag=0
       Conv_const=200  # NonConvergence penalty
    #print(Conv_flag)
    # The topological constraints
    topol_const=Topol_Constr(DSSCktObj,G_scenario)
    # The voltage violation
    V_viol=Volt_Constr(Vmagpu,nodes_conn)
    # The branch flow violation
    flow_viol=Flow_Constr(I_flow)

    ###########################################################################
    def getSumOdds(aList):
        result = 0
        for i in range(0,len(aList),2):
            result += aList[i]
        return result
    P_Consumed=[]
    P_Rated=[]
    for j in DSSCktObj.dssCircuit.Loads.AllNames:
            DSSCktObj.dssCircuit.Loads.Name=j
            #print(j)
            aList=DSSCktObj.dssCircuit.ActiveCktElement.Powers
            P_Consumed.append(getSumOdds(aList))
            P_Rated.append(DSSCktObj.dssCircuit.Loads.kw)
            

    UE=(sum(P_Rated)-sum(P_Consumed))/(1000*KVA_base)

    
    
    ###########################################################################

    return {"loss":P_loss,
            "NodeFeat(BusVoltage)":np.array(Vmagpu),
            "EdgeFeat(branchflow)":np.array(I_flow),
            "Adjacency":np.array(Adj_mat.todense()),
            "TopologicalConstr":topol_const,
            "VoltageViolation":V_viol,
            "FlowViolation":flow_viol,
            "Convergence":Conv_const
            ,"Unserved Energy":UE
            }


def take_action(DSSCktObj,action):
    # This is where I need to put the action in place for action_space shrinking technique (Normal Operation)
    #action=ACTION_VALID[action]
    
    
    #print(type(action)) # This is numpy.ndarray
    m=len(action)
    for j in range(0,m-len(DERS)):
        if action[j]<0.5:
            action[j]=0
        else:
            action[j]=1
        
    
    
    
    #print(action)
    #Input :object of type DSSObj.ActiveCircuit (COM interface for OpenDSS Circuit)
    #Input: action multi binary type. i.e., the status of each switch if it is 0 open and 1 close
    #Returns:the circuit object with action implemented
    DSSCircuit=DSSCktObj.dssCircuit
    i=DSSCircuit.SwtControls.First #1
    
    while (i>0): #and (i<m-len(DERS)):
           Swobj=DSSCktObj.dssCircuit.SwtControls.SwitchedObj
           DSSCircuit.SetActiveElement(Swobj)
           if action[i-1]==0: # i starts from 1 in DSS #if action is 0
               DSSCktObj.dssText.command='open ' + Swobj +' term=1'       #switching the line open
           else:
               DSSCktObj.dssText.command='close ' + Swobj +' term=1'      #switching the line close
           i=DSSCircuit.SwtControls.Next
    DSSCircuit.Solution.Solve() #solving the circuit to implement actions
           
    return DSSCktObj


# Assigns a very high penalty for Loop formation and Node Isolation
def Topol_Constr(DSSCktObj,G_scenario):
    #Input: The DSS Circuit object which contains system state and the corresponding graph scenario
    #Output: The constraint violation penalty
    M=200 # a large number assigned for penalty
    no_nodes=len(G_scenario.nodes())
    no_edges=len(G_scenario.edges())
    connectn=nx.is_connected(G_scenario)
    if (no_edges == no_nodes-1) and (connectn==True):
        topol_viol=0
    else:
        topol_viol = M * (int(not connectn) + (abs(no_edges - no_nodes + 1)))#changed for trial
        #topol_viol=M
    return topol_viol

# Constraint for voltage violation
# JR: modification performed to calculate the cumulative voltage violation on 03/25/22
def Volt_Constr(Vmagpu,nodes_conn):
    #Input: The pu magnitude of node voltages at all buses, node activated or node phase of all buses
    V_upper=1.10
    V_lower=0.90
    Vmin=100
    Vmax=0
    ##
    Vmin_viol=[]
    Vmax_viol=[]
    for i in range(len(nodes_conn)):
        for phase_co in nodes_conn[i]:
            if (Vmagpu[i][phase_co-1]<V_lower):
                Vmin_viol.append(V_lower-Vmagpu[i][phase_co-1])
            if (Vmagpu[i][phase_co-1]>V_upper):
                Vmax_viol.append(Vmagpu[i][phase_co-1]-V_upper)
    if not Vmin_viol or not Vmax_viol:
        V_viol=0
    else:
        V_viol=sum(Vmin_viol)+sum(Vmax_viol) # For the minimum and maximum voltage in the network(all nodes of all buses)
        #print(V_viol)
    
    return V_viol
    


# Constraint for branch flow violation
# JR: modified to calculate the summation of branch flow violation on 03/25/22
def Flow_Constr(I_flow):
    I_upper=2 #pu upper limit for average branch current
    I_lower=-2 # pu lower limit for average branch current
    flow_viol=0
    for i in I_flow:
        if (abs(i)>I_upper):
            flow_viol=flow_viol+(abs(i)-I_upper)#+abs(I_lower-i) #sum of all branch current violations
        else:
            flow_viol=0
    return flow_viol


def get_reward(observ_dict):
    #Input: A dictionary describing the state of the network
    #Output: reward
    
    reward=-(observ_dict['Unserved Energy']+observ_dict['FlowViolation']+2*observ_dict['VoltageViolation']+ observ_dict['Convergence'])
    
    return reward
