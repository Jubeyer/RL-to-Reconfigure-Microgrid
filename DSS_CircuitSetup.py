"""
In this file the DSS engine is set up.
The objects for circuit, bus, and branch are also set up for further use in the RL environment.
This file also includes function to modify the base DSS circuit with sectionalizing and tie switch information.
The translation of DSS Circuit into a graph structure is also defined here.
"""
import win32com.client
import numpy as np
import math
import networkx as nx

class DSS():  # to initialize the DSS circuit object and extract results
    def __init__(self,filename):

        self.filename=filename
        self.dssObj=win32com.client.Dispatch("OpenDSSEngine.DSS") #deploying DSS Engine

        if self.dssObj.Start(0)==False:
           print ("Problem with OpenDSS Engine initialization")

        else: # redeclaring variables defined under DSS object
            self.dssText=self.dssObj.Text
            self.dssCircuit=self.dssObj.ActiveCircuit
            self.dssSolution=self.dssCircuit.Solution
            self.dssTopology=self.dssCircuit.Topology
            self.dssBus = self.dssCircuit.ActiveBus
            self.dssCktElement=self.dssCircuit.ActiveCktElement
            self.dssLines=self.dssCircuit.Lines
            self.dssLoads=self.dssCircuit.Loads
            self.dssTransformers=self.dssCircuit.Transformers


    def version_dss(self):    # specifies the version of OpenDSS used
        return self.dssObj.Version

    def compile_ckt_dss(self): # Compiling the OpenDSS circuit
        self.dssObj.ClearAll()
        self.dssText.Command="compile [" + self.filename +"]"

    def get_cktname_dss(self): # The circuit name
        return self.dssCircuit.Name

    def get_ckt_base(self):
      self.dssTransformers.First
      self.KVA_base=self.dssTransformers.kva
      self.KV_base=self.dssTransformers.kv
      self.Z_base=((self.KV_base**2)*1000)/self.KVA_base
      return(self.KVA_base,self.KV_base,self.Z_base)

    def get_AllBuses(self):
        return self.dssCircuit.AllBusNames

    def get_AllLines(self):
        return self.dssLines.AllNames

    def get_AllPDElements(self):
        self.elem=[]
        i=self.dssCircuit.PDElements.First
        while i>0:
              self.elem.append(self.dssCircuit.PDElements.Name)
              i=self.dssCircuit.PDElements.Next
        return (self.elem)  #PD Elements include Lines, Transformers and Capacitors

    def get_BusLoads(self):
        self.Load_dict=[]
        i=self.dssCircuit.FirstPCElement() #set first power conversion element active
        while i>0:
              elname=self.dssCktElement.Name
              if elname.split('.')[0]=='Load':
                  name=elname.split('.')[1]
                  bus=self.dssCktElement.Busnames[0].split('.')[0]
                  self.dssLoads.Name=name
                  P_load=self.dssLoads.kW
                  Q_load=self.dssLoads.kvar
                  self.Load_dict.append({'Name':name, 'Bus':bus, 'Pload':P_load,'Qload':Q_load})
              i=self.dssCircuit.NextPCElement()
        return(self.Load_dict)

    def get_Sourcebus(self):
        self.sources=[]
        i=self.dssCircuit.Vsources.First
        while i>0:
             self.dssCircuit.SetActiveElement(self.dssCircuit.Vsources.Name)
             self.sources.append(self.dssCircuit.ActiveCktElement.BusNames[0].split('.')[0])
             i=self.dssCircuit.Vsources.Next
        return(self.sources)

    def solve_snapshot_dss(self,loadmultFac): #solving snapshot powerflow for particular load multiplication factor
        self.dssText.Command="Set Mode=SnapShot"
        self.dssText.Command="Set ControlMode=OFF"
        self.dssSolution.LoadMult=loadmultFac
        self.dssSolution.Solve()

    def get_results_dss(self): # total active and reactive power after power flow
        self.P= -1*(self.dssCircuit.Totalpower[0]) #active power in kW
        self.Q= -1*(self.dssCircuit.Totalpower[1]) #reactive power in kW
        losses=self.dssCircuit.Losses
        self.P_loss=(losses[0]/1000) #active power loss in kW
        self.Q_loss=(losses[1]/1000) #reactive power loss in kW
        return self.P,self.Q,self.P_loss,self.Q_loss

 # Bus class contains the bus object details
class Bus:
    def __init__(self,DSSCktobj,bus_name):
        """
        Inputs:
            circuit object
            bus name
        Contains:
            Vmag-  pu voltage magnitude at bus nodes (3 phase)
            Vang-  pu voltage angle at bus nodes (3 phase)
            nodes- node connection at bus
            Vmax- max pu voltage at bus
            Vmin- min pu voltage at bus
        """
        Vmag=np.zeros(3)
        #Vang=np.zeros(3)
        DSSCktobj.dssCircuit.SetActiveBus(bus_name)
        V=DSSCktobj.dssBus.puVmagAngle # pair of magnitude and angle of voltages in pu
        nodes=np.array(DSSCktobj.dssBus.Nodes) #Node connection
        for indx in range(len(nodes)):
            Vmag[nodes[indx]-1]=V[int(indx*2)] #assigning the voltages acc to node connection
            #Vang[nodes[indx]-1]=V[int(indx*2)+1]
        # Vmin=min(v for v in Vmag if v > 0)
        # Vmax=max(Vmag)
        self.Vmag = Vmag # 3 phase pu voltage at the buses
        # self.Vang = Vang
        self.nodes=nodes
        # self.Vmax=Vmax
        # self.Vmin=Vmin

class Branch:  # to extract properties of branch
    def __init__(self, DSSCktobj, branch_fullname):
        """
        Inputs:
            circuit object
            branch name
        Contains:
            bus_fr - from bus name
            bus_to - to bus name
            nphases - number of phases
            Cap - average current flow
        """
        # Calculating base current
        DSSCktobj.dssTransformers.First
        KVA_base=DSSCktobj.dssTransformers.kva # S base
        KV_base=DSSCktobj.dssTransformers.kv #L-L V base
        I_base=KVA_base/(math.sqrt(3)*KV_base)

        DSSCktobj.dssCircuit.SetActiveElement(branch_fullname)

        bus_connections=DSSCktobj.dssCktElement.BusNames
        bus1= bus_connections[0]
        bus2= bus_connections[1]

        i=np.array(DSSCktobj.dssCktElement.CurrentsMagAng)
        ctidx = 2 * np.array(range(0, min(int(i.size/ 4), 3)))
        I_mag = i[ctidx] #branch current in A
        #I_ang=i[ctidx + 1] #angle in deg
        #nphases=DSSCktobj.dssCktElement.NumPhases
        #MaxCap=DSSCktobj.dssCktElement.EmergAmps
        MaxCap=DSSCktobj.dssCktElement.NormalAmps
       # https://sourceforge.net/p/electricdss/discussion/861976/thread/8aa13830/
       # Problem is that Line.650632 already exceeds normal amps in Opendss=400 A and
       # Normal Amps in Kerstings book =530 A. So I will consider EmergAmps=600 A
        I_avg=np.average(I_mag)/I_base #average of all three phases in pu
        self.bus_fr=bus1
        self.bus_to=bus2
        #self.nphases=nphases
        self.Cap=I_avg
        self.MaxCap=MaxCap

def CktModSetup(DSSfile,sectional_swt,tie_swt,DER):
#def CktModSetup(DSSfile,sectional_swt,tie_swt): # give tie switches and sectionalizing switches as input
    DSSCktobj= DSS(DSSfile) #create a circuit object
    DSSCktobj.compile_ckt_dss() #compiling the circuit #compiling should only be done once in the beginning
    #Setting the iteration limits higher and disabling the warning message window
    DSSCktobj.dssText.command = "Set Maxiterations=5000"
    DSSCktobj.dssText.command = "Set maxcontroliter=5000"
    DSSCktobj.AllowForms=False
    
    return DSSCktobj

# For switches if Normal State= 1 it is open and if Normal State= 2  it is close in DSS

# ---------Graph formation and Adjacency matrix--------------#
def graph_struct(DSSCktobj):
    #Getting all elements
    All_Elems=[]
    i=DSSCktobj.dssCircuit.PDElements.First
    while i>0:
          All_Elems.append(DSSCktobj.dssCircuit.PDElements.Name)
          i=DSSCktobj.dssCircuit.PDElements.Next

    edges_dictlist=[]
    G_original=nx.Graph()
    for e in All_Elems:
        if ((e.split('.')[0]=='Line') or (e.split('.')[0]=='Transformer')): #only if its a line or transformer used in graph(capacitors avoided)
            branch_obj=Branch(DSSCktobj,e) #creating  a branch object instance with full name
            sr_node=branch_obj.bus_fr.split('.')[0] #extracting source bus of branch
            tar_node=branch_obj.bus_to.split('.')[0] #extracting target bus of branch
            name=e
            DSSCktobj.dssCircuit.SetActiveElement(e)
            G_original.add_edge(sr_node, tar_node, label=name)
    return G_original    
