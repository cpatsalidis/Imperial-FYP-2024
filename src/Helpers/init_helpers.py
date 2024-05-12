import src.Units.agentCloud as agentc
from mesa.time import SimultaneousActivation
from mesa import DataCollector  
import random
import math
import networkx as nx

# generate Small World Scale Free network
def klemm_eguilez_network(N,m,miu):
  G = nx.Graph()
  central = []
  notcentral = []
  #create fully connected netwrok of m nodes
  for i in range(m):
    G.add_node(i)
    central.append(i) #add node in list of active
    for j in G:
      if i!=j:
        G.add_edge(i, j)
  G.add_nodes_from(range(m, N)) # N-m nodes in the list
  for i in range(m,N):
    for j in central: #m edges to other nodes
      chance = random.uniform(0, 1)
      if (miu>chance) or (notcentral == []):
        G.add_edge(j, i)
      else:
        connected = 0
        while (connected==0):
          node2connect = random.choice(notcentral)
          chance2 = random.uniform(0, 1)
          E = 0
          k_j = G.degree[node2connect]
          for k in notcentral:
            E = E + G.degree[k]
          if (k_j/E)>chance2: #more likely to choose high degrees from inactive (attach to high degrees)
            G.add_edge(node2connect,i)
            connected = 1
    central.append(i)
    #remove central
    j_found = 0
    while (j_found == 0):
        j = random.choice(central)
        k_j = G.degree[j]
        E = 0
        for k in central:
          E = E + 1/G.degree[k]
        p_d = (1/k_j) / E #more likely to choose low degrees from active (remain active if high degree)
        chance3 = random.uniform(0, 1)
        if (p_d > chance3): 
          j_found = 1
          central.remove(j)
          notcentral.append(j)
  # return (G,central,notcentral)
  return G

#generate list of neighbors - based on 6 degrees of separation
def find_neighbors(self):
  agent = self.unique_id
  G = self.model.G
  neighbors = [n for n in G.neighbors(agent)]
  nei2 = []
  #second
  for sec_nei in neighbors:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei2 = nei2 + neighbors_of_nei
  nei2 = list(dict.fromkeys(nei2)) #remove duplicates
  for i in nei2:
    if (i in neighbors) or (i == agent):
      nei2.remove(i)
  #third
  nei3 = []
  for sec_nei in nei2:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei3 = nei3 + neighbors_of_nei
  nei3 = list(dict.fromkeys(nei3)) #remove duplicates
  for i in nei3:
    if (i in neighbors or i in nei2 or (i == agent)):
      nei3.remove(i)
  nei4 = []
  for sec_nei in nei3:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei4 = nei4 + neighbors_of_nei
  nei4 = list(dict.fromkeys(nei4)) #remove duplicates
  for i in nei4:
    if (i in neighbors or i in nei2 or i in nei3 or (i == agent)):
      nei4.remove(i)
  nei5 = []
  for sec_nei in nei4:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei5 = nei5 + neighbors_of_nei
  nei5 = list(dict.fromkeys(nei5)) #remove duplicates
  for i in nei5:
    if (i in neighbors or i in nei2 or i in nei3 or i in nei4 or (i == agent)):
      nei5.remove(i)
  nei6 = []
  for sec_nei in nei5:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei6 = nei6 + neighbors_of_nei
  nei6 = list(dict.fromkeys(nei6)) #remove duplicates
  for i in nei6:
    if (i in neighbors or i in nei2 or i in nei3 or i in nei4 or i in nei5 or (i == agent)):
      nei6.remove(i)
  all_neighbors = {1:neighbors,2:nei2,3:nei3,4:nei4,5:nei5,6:nei6} #not used anymore
  hopp6_neighbors = neighbors + nei2 + nei3 + nei4 + nei5 + nei6
  #remove duplicates
  hopp6_neighbors = list(dict.fromkeys(hopp6_neighbors))
  if (self.unique_id in hopp6_neighbors):
    hopp6_neighbors.remove(self.unique_id)
  return neighbors, hopp6_neighbors

# initialise collectors for visualisations
def init_collectors_cloud(self):
  self.datacollector = DataCollector(    
    model_reporters={"rounds":"rounds","meanNoise": "meanNoise","stdDNoise":"stdDNoise"},#,"agents":"G"},
    agent_reporters={"rounds":"myrounds","iN":"iN","inN":"inN","fN":"fN","intN":"intN",'xN':"xN"})
  self.datacollector2 = DataCollector(    
    model_reporters={"rounds":"rounds","thoryvos": "thoryvos","indThoryvos":"indThoryvos","netThoryvos":"netThoryvos","itThoryvos":"itThoryvos","expThoryvos":"expThoryvos","stdii":"stdii","stdbi":"stdbi","stdfi":"stdfi","stdei":"stdei"},
    agent_reporters={"Loudness":"fN","Voice":"Ns"})
  self.datacollector3= DataCollector(     
    model_reporters={"rounds":"rounds","reward":"reward","NagentsIn":"NagentsIn","thoryvos": "thoryvos"})
  self.datacollector4= DataCollector(    
    model_reporters={"rounds":"rounds","stdii": "stdii","stdfi": "stdfi","stdbi": "stdbi","stdei": "stdei","stdsel": "stdsel","Eii": "Eii","Efi": "Efi","Ebi": "Ebi","Eei": "Eei","Esel": "Esel"})
  self.datacollector5= DataCollector(    
    model_reporters={"rounds":"rounds","aIV": "aIV","aEV":"aEV","aFN":"aFN","aBN":"aBN"})
  self.datacollector6= DataCollector(    
    model_reporters={"rounds":"rounds","reward":"reward","NagentsIn": "NagentsIn","thoryvos": "thoryvos","aIV": "aIV","aEV":"aEV","aFN":"aFN","aBN":"aBN","stdii": "stdii","stdfi": "stdfi","stdbi": "stdbi","stdei": "stdei","stdsel": "stdsel","Eii": "Eii","Efi": "Efi","Ebi": "Ebi","Eei": "Eei","Esel": "Esel"})    
  self.datacollector7= DataCollector(    
    model_reporters={"rounds":"rounds","amountasked":"amountasked","commonsources":"commonsources","difsources": "difsources"},
    agent_reporters={"rounds":"myrounds","amasked":"amasked","expert":"expert"}) 

# initialise variables of model
def initVarCloud(self):
    self.envFactor = 1
    self.stdii = 0
    self.stdfi = 0
    self.stdbi = 0
    self.stdei = 0
    self.stdsel = 0
    self.Eii = 0
    self.Efi = 0
    self.Ebi = 0
    self.Eei = 0
    self.Esel = 0  
    self.vectorIn = {}
    self.timesasked ={}
    self.amountasked = {}
    self.creditsources = []
    self.allnoiseselection = {}
    self.allJobs = {}
    self.allPrios = {}
    self.allUrg = {}
    self.allimPat = {}
    self.allDelay = {}
    self.allOrder = {}
    self.allsumOrder = {}
    self.allCumDelay = {}
    self.allRoundsAlive = {}
    self.avgRoundDelay = {}
    self.allavgDis = {}
    self.allCumDis = {}
    self.avgimPatience = 0
    self.stdimPatience = 0
    self.meanDis = 0
    self.CumDis = 0
    self.meanCumDis = 0
    self.allCumNoise = {}
    self.allavgNoise = {}
    self.inavgNoise = 0
    self.outavgNoise =0
    self.meanNoise = 0
    self.stdNoise = 0
    self.stdDis = 0
    self.TCO = {} #total cost
    self.QoS = {} #total quality (numofjobs/delay)
    self.iQoS = {} #individual quality
    self.iTCO = {} # individual cost
    self.maxiQoS = {}
    self.miniTCO = {}
    self.agentOrder = {}
    self.iN = {}
    self.xN = {}
    self.iNprev = {}
    self.thprev = 0
    self.expprev = 0
    self.inN = {}
    self.longiN = {}
    self.intN = {}
    self.fN = {}
    self.fintN = {}
    self.orderJobs = {}
    self.thoryvos = 0
    self.indThoryvos = 0
    self.netThoryvos = 0
    self.itThoryvos = 0
    self.expThoryvos = 0
    self.longThoryvos = 0
    self.pred = 0
    self.reward = 0
    self.cumreward = 0
    self.jobsize = 0
    self.NagentsIn = random.randint(1,self.n_agents)
    self.agentsIn = []
    self.jobsIn = {}
    self.fixedcost = 500
    self.avgIndTh = 0
    self.commonsources = 0
    self.difsources = 0
    self.totalAgInC= dict.fromkeys(range(1,self.n_agents), 0)
    self.AgInTimes = dict.fromkeys(range(1,self.n_agents), 0)
    self.avgAgInC = dict.fromkeys(range(1,self.n_agents), 0)
    self.totalAgInQ= dict.fromkeys(range(1,self.n_agents), 0)
    self.avgAgInQ = dict.fromkeys(range(1,self.n_agents), 0)
    self.expertsIn = random.choices(self.activeAgents, k = math.ceil(math.sqrt(self.n_agents)))
    self.expNoise = 0
    self.expNoiseUrg = 0
    self.aIV = 0
    self.aEV = 0
    self.aFN = 0
    self.aBN = 0
    self.schedule = SimultaneousActivation(self)
    for i in range(self.n_agents):
        a = agentc.Agent(i, self)
        self.schedule.add(a)
