"""
In this file the sectionalizing and tie switch details are specified. 
Also the path for the DSS file containing the circuit information is specified.
The final DSS circuit which will be used by the environment is created.
"""

import os
import networkx as nx
import numpy as np
import random
from  DSS_CircuitSetup import*

"""
sectional_swt=[{'no':1,'line':'L25'},
               {'no':2,'line':'L17'},
               {'no':3,'line':'L30'},
               {'no':4,'line':'L14'},
               {'no':5,'line':'L13'}]

tie_swt=[{'no':1,'from node':'828','from conn':'.1.2.3', 'to node':'832','to conn':'.1.2.3', 'length':21,'code':'303', 'name':'L33'},
         {'no':2,'from node':'824','from conn':'.1.2.3','to node':'848','to conn':'.1.2.3','length':32,'code':'303', 'name':'L34'},
         {'no':3,'from node':'840','from conn':'.1.2.3','to node':'848','to conn':'.1.2.3','length':32,'code':'303', 'name':'L35'},
         {'no':4,'from node':'814','from conn':'.1.2.3','to node':'828','to conn':'.1.2.3','length':32,'code':'303', 'name':'L36'}]

# On 03/18/2022
# For DG placement
DER=[{'no':1,'bus':'814','name':'DG1','pf':1,'kW':50,'kVar':0,'model':3, 'kv':24.9}]
"""

sectional_swt=[]
tie_swt=[]
DER=[]


###############################################################################




###############################################################################
FolderName=os.path.dirname(os.path.realpath("__file__"))
DSSfile=r""+ FolderName+ "/ieee34Mod1.dss"
DSSCktobj=CktModSetup(DSSfile,sectional_swt,tie_swt,DER)
n_actions=len([j for j in DSSCktobj.dssCircuit.Lines.AllNames if "sw" in j])
mult_constant = 1.000000000
def initialize():
    FolderName=os.path.dirname(os.path.realpath("__file__"))
    DSSfile=r""+ FolderName+ "/ieee34Mod1.dss"
    #DSSCktobj=CktModSetup(DSSfile,sectional_swt,tie_swt) # initially the sectionalizing switches close and tie switches open
    
    # Commented out on 06/02/2022
    DSSCktobj=CktModSetup(DSSfile,sectional_swt,tie_swt,DER) # initially the sectionalizing switches close and tie switches open
    
    DSSCktobj.dssSolution.Solve() #solving snapshot power flows
    if DSSCktobj.dssSolution.Converged:
       conv_flag=1
    else:
       conv_flag=0
    G_init=graph_struct(DSSCktobj)
    return DSSCktobj,G_init,conv_flag

def test_initialize():       
    FolderName=os.path.dirname(os.path.realpath("__file__"))
    DSSfile=r""+ FolderName+ "/ieee34Mod1.dss"
    #######################----------Older one without DER #####################
    #DSSCktobj=CktModSetup(DSSfile,sectional_swt,tie_swt)
    ###########################New one with DER #############################
    # Commented out on 06/02/2022
    DSSCktobj=CktModSetup(DSSfile,sectional_swt,tie_swt,DER) # initially the sectionalizing switches close and tie switches open
    i = DSSCktobj.dssLoads.First
    while i > 0:
        DSSCktobj.dssLoads.kW = round(DSSCktobj.dssLoads.kW * 0.5, 2)
        i = DSSCktobj.dssLoads.Next
        
    ##################### New addition for single line outage testing ######################
    
    # commented out for now to check the normal operation
    # Candidate_Lines=DSSCktobj.dssCircuit.Lines.AllNames
    # #L_OUT='L24'#random.choice(Candidate_Lines)#considering one single outage at a time
    # L_OUT=random.choices(Candidate_Lines,k=random.randint(1,len(Candidate_Lines)))
    # print("The line/lines that is/are out :",L_OUT)
    # if len(L_OUT)>1:
    #     for i in L_OUT:
    #         DSSCktobj.dssText.command='open ' +'line.'+ i +' term=1'
    # else:
    #     DSSCktobj.dssText.command='open ' +'line.'+ L_OUT +' term=1'
        
    DSSCktobj.dssSolution.Solve() #solving snapshot power flows
    if DSSCktobj.dssSolution.Converged:
       conv_flag=1
    else:
       conv_flag=0
    G_init=graph_struct(DSSCktobj)
    return DSSCktobj,G_init,conv_flag

########################### Addition for shrinking the action space ###########
def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points

#n_actions = len(sectional_swt) + len(tie_swt)


       
