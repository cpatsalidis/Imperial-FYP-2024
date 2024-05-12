from mesa import Agent
import random
import src.Helpers.init_helpers as ih

class Agent(Agent):
  def __init__(self, unique_id, model):
    super().__init__(unique_id, model)
    self.unique_id = unique_id
    if self.model.rounds == 0:
      self.oneneighbors = []
      self.neighbors = []
    self.start = 1
    self.myrounds = self.model.rounds
    self.selfconfidence = 0.5
    self.ucoefficient = 0.1
    self.trust = {}
    self.trustNoise = 0.5
    self.trustFN = 0.5
    self.trustExp =0.5
    self.selfconfidenceit = 0.5
    self.last_asked = 0
    self.last_asked_it = 0
    self.c = 0.001
    self.averagedemand = random.randint(1,self.model.maxJobSize)
    self.averagepriority = random.uniform(self.averagedemand,self.model.avgDelay)#random.uniform(self.model.avgDelay/2,self.model.avgDelay+self.model.maxJobSize/2) #priority defines amount of delay accepted
    self.averageurgency = random.randint(1,self.model.maxJobSize) 
    self.jobSize = self.averagedemand
    self.jobPriority = self.averagepriority
    self.jobUrgency =self.averageurgency
    self.totalJobS = self.jobSize
    self.totalJobP = self.jobPriority
    self.iN = 0
    self.inN = 0
    self.intN = 0
    self.fN = 0
    self.fintN = 0
    self.longiN = 0
    self.xN = 0
    self.Ns = random.choice([-1,0,1,2])
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
    self.expert = 0
    
    #initialise credence to neighbours 
  def initialise_trust(self):
    node_list = self.oneneighbors
    init_trust = {}
    for i in node_list:
      init_trust[i] = 0.5    
    return init_trust

  def step(self):
    if (self.start == 1):
      self.start = 0
      self.oneneighbors,self.neighbors = ih.find_neighbors(self)
      self.trust = self.initialise_trust() 
    self.myrounds +=1
    
