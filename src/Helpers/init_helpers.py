import src.Units.agentCloud as agentc
from mesa.time import SimultaneousActivation
from mesa import DataCollector  
import random
import math
import networkx as nx

# Generate Small World Scale Free network
def klemm_eguilez_network(N,m,miu):
  # Initialize a new graph
  G = nx.Graph()

  # Lists to keep track of central (active) and non-central (inactive) nodes
  central = []
  notcentral = []

  # Create a fully connected network of m nodes (initial active nodes)
  for i in range(m):
    G.add_node(i)
    central.append(i) # add node to list of active nodes
    for j in G:
      if i!=j:
        G.add_edge(i, j) # create edge to every other node

  # Add remaining N-m nodes to the graph     
  G.add_nodes_from(range(m, N)) 

  # Connect new nodes to the network
  for i in range(m,N):
    for j in central: 
      chance = random.uniform(0, 1)
      # Connect new node to the network based on chance and miu parameter
      if (miu>chance) or (notcentral == []):
        G.add_edge(j, i)
      else:
        connected = 0
        while (connected==0):
          # Select a random node from non-central nodes to connect to
          node2connect = random.choice(notcentral)
          chance2 = random.uniform(0, 1)
          E = 0
          k_j = G.degree[node2connect]
          # Calculate total degree of non-central nodes
          for k in notcentral:
            E = E + G.degree[k]
          # Connect based on node degree; more likely to choose high degree nodes
          if (k_j/E)>chance2: 
            G.add_edge(node2connect,i)
            connected = 1

    # Randomly remove central nodes based on their degree, moving them to non-central list
    central.append(i)
    j_found = 0
    while (j_found == 0):
        j = random.choice(central)
        k_j = G.degree[j]
        E = 0
        for k in central:
          E = E + 1/G.degree[k]
        p_d = (1/k_j) / E # Probability of choosing node based on degree
        chance3 = random.uniform(0, 1)
        if (p_d > chance3): 
          j_found = 1
          central.remove(j)
          notcentral.append(j)
  # return (G,central,notcentral)
  return G

# Generate list of neighbors - based on 6 degrees of separation
def find_neighbors(self):
  # Retrieve unique identifier for the current agent
  agent = self.unique_id
  # Access the graph structure from the model
  G = self.model.G
  # Collect first-degree neighbors of the agent
  neighbors = [n for n in G.neighbors(agent)]
  nei2 = []

  # SECOND DEGREE: Collect second-degree neighbors
  for sec_nei in neighbors:
    # For each first-degree neighbor, get their neighbors
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei2 = nei2 + neighbors_of_nei
  # Remove duplicates in the second-degree neighbor list
  nei2 = list(dict.fromkeys(nei2)) 
  # Remove the original agent and their first-degree neighbors from the list
  for i in nei2:
    if (i in neighbors) or (i == agent):
      nei2.remove(i)

  # THIRD DEGREE: Collect third-degree neighbors
  nei3 = []
  for sec_nei in nei2:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei3 = nei3 + neighbors_of_nei
  nei3 = list(dict.fromkeys(nei3))
  # Exclude all first, second-degree neighbors, and the agent itself
  for i in nei3:
    if (i in neighbors or i in nei2 or (i == agent)):
      nei3.remove(i)

  # FOURTH DEGREE: Collect fourth-degree neighbors
  nei4 = []
  for sec_nei in nei3:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei4 = nei4 + neighbors_of_nei
  nei4 = list(dict.fromkeys(nei4)) 
  # Exclude first, second, third-degree neighbors, and the agent
  for i in nei4:
    if (i in neighbors or i in nei2 or i in nei3 or (i == agent)):
      nei4.remove(i)

  # FIFTH DEGREE: Collect fifth-degree neighbors
  nei5 = []
  for sec_nei in nei4:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei5 = nei5 + neighbors_of_nei
  nei5 = list(dict.fromkeys(nei5)) 
  # Exclude first through fourth-degree neighbors, and the agent
  for i in nei5:
    if (i in neighbors or i in nei2 or i in nei3 or i in nei4 or (i == agent)):
      nei5.remove(i)

  # SIXTH DEGREE: Collect sixth-degree neighbors
  nei6 = []
  for sec_nei in nei5:
    neighbors_of_nei = [n for n in G.neighbors(sec_nei)] 
    nei6 = nei6 + neighbors_of_nei
  nei6 = list(dict.fromkeys(nei6))
  # Exclude neighbors from all previous degrees and the agent
  for i in nei6:
    if (i in neighbors or i in nei2 or i in nei3 or i in nei4 or i in nei5 or (i == agent)):
      nei6.remove(i)

  # all_neighbors = {1:neighbors,2:nei2,3:nei3,4:nei4,5:nei5,6:nei6} #not used anymore

  # Combine all degrees of neighbors and ensure no duplicates
  hopp6_neighbors = neighbors + nei2 + nei3 + nei4 + nei5 + nei6
  
  # Ensure the agent itself is not included in its list of neighbors
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
    model_reporters={"epoch_mean_error":"epoch_mean_error","epoch_std_error":"epoch_std_error","rounds":"rounds","reward":"reward","NagentsIn": "NagentsIn","thoryvos": "thoryvos","aIV": "aIV","aEV":"aEV","aFN":"aFN","aBN":"aBN","stdii": "stdii","stdfi": "stdfi","stdbi": "stdbi","stdei": "stdei","stdsel": "stdsel","Eii": "Eii","Efi": "Efi","Ebi": "Ebi","Eei": "Eei","Esel": "Esel"})    
  self.datacollector7= DataCollector(    
    model_reporters={"rounds":"rounds","amountasked":"amountasked","commonsources":"commonsources","difsources": "difsources"},
    agent_reporters={"rounds":"myrounds","amasked":"amasked","expert":"expert"}) 

# initialise variables of model
def initVarCloud(self):
    self.envFactor = 1
    self.stdii = 0 # Standard deviation of individual noise
    self.stdfi = 0 # Standard deviation of foreground noise
    self.stdbi = 0 # Standard deviation of background noise
    self.stdei = 0 # Standard deviation of expert noise
    self.stdsel = 0 # Standard deviation of noise attended to
    self.Eii = 0 # Mean ...
    self.Efi = 0 #
    self.Ebi = 0 #
    self.Eei = 0 #
    self.Esel = 0 # 
    self.vectorIn = {} 
    self.timesasked ={} # Number of times each agent has asked for their opinion 
    self.amountasked = {} # Normalized number of times each agent has been asked
    self.creditsources = [] # Top 'n' agent significant sources of opinion
    self.allnoiseselection = {} # The number of times the collective payed attention to each noise source (0 = own, 2 = expert, 1 = foreground, -1 = background)
    self.allJobs = {} # All agent job sizes
    self.allPrios = {} # All agent job priorities
    self.allUrg = {} # All agent job urgencies
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
    self.maxiQoS = {} # Maximum individual quality of servicev
    self.miniTCO = {}
    self.agentOrder = {}
    self.voi = {}
    self.suggestion = {} # suggestion for the next agent to ask
    self.pwtp = {}
    self.iN = {} # Individual noise
    self.xN = {} # Expert noise TOP
    self.iNprev = {}
    self.thprev = 0
    self.expprev = 0
    self.inN = {} # Neighbour noise (Foreground noise)
    self.longiN = {}
    self.intN = {} # Background noise
    self.fN = {} # Noise attended to
    self.fintN = {}
    self.orderJobs = {}
    self.thoryvos = 0 # Total noise
    self.indThoryvos = 0 # Total individual noise
    self.netThoryvos = 0 # Total foreground noise
    self.itThoryvos = 0 # Total background noise
    self.expThoryvos = 0 # Total expert noise
    self.longThoryvos = 0 # Not used
    self.pred = 0
    self.reward = 0 
    self.cumreward = 0
    self.jobsize = 0
    self.NagentsIn = random.randint(1,self.n_agents)
    self.agentsIn = []
    self.jobsIn = {}
    self.fixedcost = 500 # Fixed cost tha contributes to the total cost 
    self.avgIndTh = 0 # Average individual noise
    self.commonsources = 0
    self.difsources = 0
    self.totalAgInC= dict.fromkeys(range(1,self.n_agents), 0)
    self.AgInTimes = dict.fromkeys(range(1,self.n_agents), 0)
    self.avgAgInC = dict.fromkeys(range(1,self.n_agents), 0)
    self.totalAgInQ= dict.fromkeys(range(1,self.n_agents), 0)
    self.avgAgInQ = dict.fromkeys(range(1,self.n_agents), 0)
    self.expertsIn = random.choices(self.activeAgents, k = math.ceil(math.sqrt(self.n_agents))) # Randomly select experts from active agents
    self.expNoise = 0
    self.expNoiseUrg = 0 # Expert noise
    self.aIV = 0 # Number of times the collective payed attention to individual noise
    self.aEV = 0 # Number of times the collective payed attention to expert noise
    self.aFN = 0 # Number of times the collective payed attention to foreground noise
    self.aBN = 0 # Number of times the collective payed attention to background noise
    self.agent_interactions = {i: {j: 0 for j in range(self.n_agents) if i != j} for i in range(self.n_agents)}
    self.sqrt_errors = {i: {j: 0 for j in range(self.n_agents) if i != j} for i in range(self.n_agents)}
    self.noise_variance = 10
    self.epoch_mean_error = 0
    self.epoch_std_error = 0


    # Create a scheduler that activates all agents at the same time each step.
    self.schedule = SimultaneousActivation(self) 
    # Create n, number of agents with unique IDs and reference to the current model
    for i in range(self.n_agents): 
        a = agentc.Agent(i, self) 
        self.schedule.add(a) # Add the newly created agent to the scheduler
