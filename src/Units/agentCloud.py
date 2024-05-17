from mesa import Agent
import random
import src.Helpers.init_helpers as ih

class Agent(Agent):
  def __init__(self, unique_id, model):
    super().__init__(unique_id, model)
    self.unique_id = unique_id
    if self.model.rounds == 0:
      self.oneneighbors = [] # first degree neighbors
      self.neighbors = [] # Up to 6th degree neighbors
    self.start = 1
    self.myrounds = self.model.rounds
    self.selfconfidence = 0.5
    self.ucoefficient = 0.1
    self.trust = {}
    self.timesasked = {}
    self.trustNoise = 0.5 # Trust in background noise
    self.trustFN = 0.5 # Trust in foreground noise
    self.trustExp =0.5 # Trust in expert noise
    self.selfconfidenceit = 0.5 # Trust in individual noise
    self.last_asked = 0 # Last agent asked for their opinion
    self.last_asked_it = 0 # not used
    self.voi = 1 # Value of information
    self.pwtp = 0.5 # Price willing to pay for observer information
    self.c = 0.001 
    self.averagedemand = random.randint(1,self.model.maxJobSize) # Initial average demand affects each agents job size in 'genJobs' function
    self.averagepriority = random.uniform(self.averagedemand,self.model.avgDelay) #random.uniform(self.model.avgDelay/2,self.model.avgDelay+self.model.maxJobSize/2) #priority defines amount of delay accepted
    self.averageurgency = random.randint(1,self.model.maxJobSize) # Initial average urgency is random between 1 and maxJobSize
    self.jobSize = self.averagedemand  # Initial job size is the average demand
    self.jobPriority = self.averagepriority # Initial job priority is the average priority
    self.jobUrgency =self.averageurgency # Initial job urgency is the average urgency
    self.totalJobS = self.jobSize # Total job size is the job size
    self.totalJobP = self.jobPriority # Total job priority is the job priority
    self.iN = 0 # Individual noise
    self.inN = 0 # Neighbour noise (Foreground noise)
    self.intN = 0
    self.fN = 0
    self.fintN = 0
    self.longiN = 0
    self.xN = 0 # Exprert noise
    self.Ns = random.choice([-1,0,1,2])
    self.suggestion = 0
    model.allJobs[self.unique_id] = self.jobSize 
    model.allPrios[self.unique_id] = self.jobPriority
    model.allUrg[self.unique_id] = self.jobUrgency
    model.allimPat[self.unique_id] = 0
    model.allDelay[self.unique_id] = 0
    model.allOrder[self.unique_id] = 0
    model.allsumOrder[self.unique_id] = 0
    model.allCumDelay[self.unique_id] = 0
    model.allRoundsAlive[self.unique_id] = 0
    model.allavgDis[self.unique_id] = 0
    model.allCumNoise[self.unique_id] = 0
    model.allavgNoise[self.unique_id] = 0
    model.allnoiseselection[self.unique_id] = 0 #0 =own, 1=foreground, -1 background
    model.timesasked[self.unique_id] = 0
    model.amountasked[self.unique_id] = 0
    self.amasked = 0
    self.expert = 0 # Identify if agent is an expert

  def update_urgency(self, amount):
        self.urgency = max(0, self.jobUrgency + amount)  # Ensure urgency doesn't go negative
  
  def update_trust(self, trust_updates):
        for agent_id, multiplier in trust_updates.items():
            if agent_id in self.trust:
                #print(agent_id, " : ",multiplier)
                self.trust[agent_id] = max(0, self.trust[agent_id] * multiplier)  # Ensure trust doesn't go negative
                

    
  # Initialise credence to neighbours 
  def initialise_trust(self):
    node_list = self.oneneighbors
    init_trust = {}
    for i in node_list:
      init_trust[i] = 0.5    
    return init_trust

  # Reset agent to initial state (always called)
  def step(self):
    if (self.start == 1):
      self.start = 0
      self.oneneighbors,self.neighbors = ih.find_neighbors(self)
      self.trust = self.initialise_trust() 
    self.myrounds +=1
    
