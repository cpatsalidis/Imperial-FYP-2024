import random
import src.Helpers.init_helpers as ih
import math
import src.Helpers.updates as up
import statistics
import numpy as np

#initialise credence to neighbours (helper function)
def initialise_trust(node_list):
    init_trust = {}
    for i in node_list:
      init_trust[i] = 0.5    
    return init_trust
  
#initialise credence to neighbours
def init_trust(self):
  for agent in self.schedule.agents:
    agent.oneneighbors,agent.neighbors = ih.find_neighbors(agent)
    agent.trust =  initialise_trust(agent.oneneighbors) 
    agent.trustNoise = 0.5
    agent.selfconfidence = 0.5

#initialise network properties in every round ()
def initialiseRound(self):
  # self.rounds += 1
  self.activeAgents = []
  for agent in self.schedule.agents:
    self.activeAgents.append(agent.unique_id)
    self.allRoundsAlive[agent.unique_id] = self.allRoundsAlive.get(agent.unique_id,0) + 1 
    agent.oneneighbors,agent.neighbors = ih.find_neighbors(agent)
    agent.trust =  up.add_trust(agent) 

# generate jobs randomly with size from given range
def genJobs(self):
  self.allJobs = {}
  self.allPrios = {}
  self.allUrg = {}
  for agent in self.schedule.agents:
    agent.jobSize = random.choice([1,agent.averagedemand-1,agent.averagedemand,agent.averagedemand+1])
    agent.jobPriority = random.choice([0,agent.averagepriority-1,agent.averagepriority,agent.averagepriority+1])
    agent.jobUrgency = random.randint(0,self.maxJobSize)
    agent.totalJobS = agent.totalJobS + agent.jobSize
    agent.totalJobP = agent.totalJobP + agent.jobPriority
    self.allJobs[agent.unique_id] = agent.jobSize
    self.allPrios[agent.unique_id] = agent.jobPriority
    self.allUrg[agent.unique_id] = agent.jobUrgency
  self.jobsize = sum(self.allJobs.values())

# order jobs based on urgency + size
def orderNchooseJobs(self): #returns updated the jobsIn
  jobsIncr = dict(sorted(self.allJobs.items(), key=lambda item: item[1],reverse=False))
  for ag, js in jobsIncr.items():
    jobsIncr[ag] = self.allUrg.get(ag,0)
  urgord = dict(sorted(jobsIncr.items(), key=lambda item: item[1],reverse=True))
  smallurg = {}
  for agent in urgord.keys():
    smallurg[agent] = self.allJobs.get(agent,0)
  self.allJobs = smallurg.copy()
  n = int(self.NagentsIn)
  self.jobsIn = {}
  for ag, job in smallurg.items():
    if (n>0):
      self.jobsIn[ag] = job
      n = n -1
  self.agentsIn = list(self.jobsIn.keys())

#process jobs based on decided order (smallest first, urgent first etc.)
def processJobsOrder(self):
  self.iQoS = {}
  self.iTCO = {}
  self.vectorIn = {}
  totalCompute = 0
  totalDelay = 0
  totalNumJobs = 0 
  self.maxiQoS = {}
  tcextra = 0
  for ag,js in self.allJobs.items():
    if ag not in self.agentsIn:
      self.vectorIn[ag] = 0
      self.iQoS[ag] = 0
      self.iTCO[ag] = 0
      tcextra += js
      self.maxiQoS[ag] = tcextra
    else: 
      self.vectorIn[ag] = 1
      totalCompute += js
      tcextra += js 
      self.maxiQoS[ag] = tcextra
      self.iQoS[ag] = totalCompute
      totalDelay = totalDelay + totalCompute
      totalNumJobs += 1
  #reconsider TQoS, TCO denominator 
  self.QoS = sum(self.iQoS.values())/max(1,self.NagentsIn)
  self.TCO = totalCompute + self.fixedcost
  for agent in self.activeAgents:
    if agent in self.agentsIn:
      self.iTCO[agent] = self.TCO/max(1,self.NagentsIn)
    self.miniTCO[agent] = (tcextra + self.fixedcost)/max(1,len(self.activeAgents))
    
# generate individual noise based on quality of service (delay) and cost 
def iNSr(self):
  self.iN = {}
  urgin = 0
  urgout = 0
  urgall = 0 
  self.outavgNoise = 0
  self.inavgNoise = 0
  for ag, urg in self.allUrg.items():
    urgall += urg
    if ag in self.agentsIn:
      urgin += urg
    else: 
      urgout += urg 
  urgi = ((urgin-urgout)/urgall)
  avgU = urgall/max(1,len(self.activeAgents))
  avgQ = sum(self.iQoS.values())/max(1,self.NagentsIn)
  avgC = self.TCO/max(1,self.NagentsIn)
  maxQ = sum(self.maxiQoS.values())/max(1,len(self.activeAgents))
  self.expNoise = - (maxQ-avgQ)/avgC
  self.expNoiseUrg = (self.expNoise - urgi*10)/10
  for agent in self.schedule.agents: 
    if agent.unique_id in self.expertsIn:
      Si = self.expNoiseUrg
    else: 
      U = self.allUrg.get(agent.unique_id,0)
      if agent.unique_id not in self.agentsIn:
        Si = (U/max(1,avgU))
        self.outavgNoise = Si + self.outavgNoise
      else: 
        Q = self.iQoS.get(agent.unique_id,0)
        C = self.iTCO.get(agent.unique_id,0)
        noise = - (maxQ-Q)/C - (U/max(1,avgU))*10
        Si = noise/10#noise+self.expNoiseUrg)/2
        self.inavgNoise = Si + self.inavgNoise
    agent.iN = Si #(-1+2/(1+math.exp(-(Si))))  
    agent.longiN = agent.longiN+agent.iN
    agent.xN = self.expNoiseUrg
    self.iN[agent.unique_id] = agent.iN
    self.xN[agent.unique_id] = self.expNoiseUrg
    self.longiN[agent.unique_id] = agent.longiN/max(1,self.rounds)
  self.inavgNoise =  self.inavgNoise/max(1,self.NagentsIn)
  self.outavgNoise = self.outavgNoise/max(1,len(self.activeAgents)-self.NagentsIn)
  # print(self.NagentsIn,max(self.iN.values()),self.expNoiseUrg,min(self.iN.values()))         

# generate individual noise based on quality of service (delay) and cost 
def iNoiseTest(self):
  self.iN = {}
  self.expNoise = ((self.n_agents*self.maxJobSize)/2)/((self.n_agents*self.maxJobSize/2) + self.fixedcost)-self.NagentsIn*self.maxJobSize/(self.NagentsIn*self.maxJobSize + self.fixedcost)
  for agent in self.schedule.agents: 
    if agent.unique_id in self.expertsIn:
      Si = self.expNoise
    else: 
      if agent.unique_id not in self.agentsIn:
        Si = 0
      else: 
        Q = self.iQoS.get(agent.unique_id,0)
        C = self.iTCO.get(agent.unique_id,0)
        noise = Q/C
        Si = noise
    agent.iN = Si #(-1+2/(1+math.exp(-(Si))))  
    agent.xN = self.expNoise
    self.iN[agent.unique_id] = agent.iN

# listen to network noise (foreground noise)
def netNoise(self):
  self.inN = {}
  for agent in self.schedule.agents: 
    # rand = random.randint(0,1)
    if agent.oneneighbors == []: 
      agentchosen = random.choice(self.activeAgents)
      self.inN[agent.unique_id] = self.iN.get(agentchosen,0)
      agent.last_asked = agentchosen
    else:
      flag = 0
      agent.trust =  {k: v for k, v in sorted(agent.trust.items(), key=lambda item: item[1], reverse=True)}
      trust_internal = agent.trust.copy()
      max_value = max(agent.trust.values())
      max_keys = [k for k, v in agent.trust.items() if v == max_value] # getting all keys containing the `maximum`
      if (type(max_keys) == list):
        l = len(max_keys)
        for i in range(0,l):
          if flag == 0:
            key = random.choice(max_keys)
            max_keys.remove(key)
            trust_internal.pop(key)
            p = random.uniform(0, 1)
            if (p > 0.5 and self.iN.get(key,0) !=0 and key in agent.oneneighbors):
              self.inN[agent.unique_id] = self.iN.get(key,0)
              agent.last_asked = key
              self.timesasked[key] = self.timesasked.get(key,0) + 1
              flag = 1
        #you did not find it in max -> look for it in the rest
        if flag == 0:
          for key, value in trust_internal.items():
            p = random.uniform(0, 1)
            if (p > 0.5 and flag == 0):
              if (key in agent.oneneighbors and self.iN.get(key,0)!=0):
                self.inN[agent.unique_id] = self.iN.get(key,0)
                agent.last_asked = key
                self.timesasked[key] = self.timesasked.get(key,0) + 1
                #if not evaluated yet
      #iterate over agents and randomly select whom to ask in the order of trust
      else:
        for key, value in agent.trust.items():
          if flag == 0:
            p = random.uniform(0, 1)
            if (p > 0.5):
              if (key in agent.oneneighbors and self.iN.get(key,0)!=0):
                self.inN[agent.unique_id] = self.iN.get(key,0)
                agent.last_asked = key
                self.timesasked[key] = self.timesasked.get(key,0) + 1
                flag = 1
      agent.inN = self.inN.get(agent.unique_id,0)

# listen to background noise     
def backNoise(self): 
  self.intN = {}
  for agent in self.schedule.agents:
    randomsample = random.choices(self.activeAgents,k=math.ceil(len(self.activeAgents)/5))
    anoise = {}
    for agent2 in randomsample:
      anoise[agent2] = self.iN.get(agent2,0)
    avg = sum(anoise.values())/max(1,len(randomsample))
    agent.intN = avg
    self.intN[agent.unique_id] = agent.intN

# based on value of attention, select which voice to pay attention to
def attendNoiseFBIEandR(self):
  self.fN = {}
  self.allnoiseselection = {}
  for agent in self.schedule.agents:
    fore = agent.trustFN
    back = agent.trustNoise
    own = agent.selfconfidence
    exp = agent.trustExp
    p = random.uniform(0,1)
    if p > 0.3: 
      if max(fore,back,own,exp) == exp:
        self.fN[agent.unique_id] = self.expNoiseUrg
        self.allnoiseselection[agent.unique_id] = 2
        agent.Ns = 2
      elif max(fore,back,own,exp) == fore:
        self.fN[agent.unique_id] = self.inN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = 1
        agent.Ns = 1
      elif max(fore,back,own,exp) == back:
        self.fN[agent.unique_id] = self.intN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = -1
        agent.Ns = -1
      else: 
        self.fN[agent.unique_id] = self.iN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = 0
        agent.Ns = 0
    else:
      choose = random.choice(['f','b','i','x'])
      if choose == 'x':
        self.fN[agent.unique_id] = self.expNoiseUrg
        self.allnoiseselection[agent.unique_id] = 2
        agent.Ns = 2
      elif choose == 'f':
        self.fN[agent.unique_id] = self.inN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = 1
        agent.Ns = 1
      elif choose == 'b':
        self.fN[agent.unique_id] = self.intN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = -1
        agent.Ns = -1
      else: 
        self.fN[agent.unique_id] = self.iN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = 0    
        agent.Ns = 0  
    agent.fN = self.fN.get(agent.unique_id,0)
    self.allCumNoise[agent.unique_id] = self.allCumNoise.get(agent.unique_id,0) + self.fN.get(agent.unique_id,0)
    self.allavgNoise[agent.unique_id] = self.allCumNoise.get(agent.unique_id,0)/max(1,self.allRoundsAlive.get(agent.unique_id,0))
    # print(self.allnoiseselection[agent.unique_id])
  self.meanNoise = statistics.mean(list(self.allavgNoise.values()))
  self.stdDNoise = statistics.stdev(list(self.allavgNoise.values()))
  return self.fN

# if objective is random, select to pay attention to noise randomly
def attendRandom(self):
  self.fN = {}
  self.allnoiseselection = {}
  for agent in self.schedule.agents:
    fore = agent.trustFN
    back = agent.trustNoise
    own = agent.selfconfidence
    exp = agent.trustExp
    choose = random.choice(['f','b','i','x'])
    if choose == 'x':
      self.fN[agent.unique_id] = self.expNoiseUrg
      self.allnoiseselection[agent.unique_id] = 2
    elif choose == 'f':
      self.fN[agent.unique_id] = self.inN.get(agent.unique_id,0)
      self.allnoiseselection[agent.unique_id] = 1
    elif choose == 'b':
      self.fN[agent.unique_id] = self.intN.get(agent.unique_id,0)
      self.allnoiseselection[agent.unique_id] = -1
    else: 
      self.fN[agent.unique_id] = self.iN.get(agent.unique_id,0)
      self.allnoiseselection[agent.unique_id] = 0      
    agent.fN = self.fN.get(agent.unique_id,0)
    self.allCumNoise[agent.unique_id] = self.allCumNoise.get(agent.unique_id,0) + self.fN.get(agent.unique_id,0)
    self.allavgNoise[agent.unique_id] = self.allCumNoise.get(agent.unique_id,0)/max(1,self.allRoundsAlive.get(agent.unique_id,0))
    # print(self.allnoiseselection[agent.unique_id])
  self.meanNoise = statistics.mean(list(self.allavgNoise.values()))
  self.stdDNoise = statistics.stdev(list(self.allavgNoise.values()))
  return self.fN

# identify sources of opinion (i.e. influencers)
def community_sources(self):
  self.timesasked = dict(sorted(self.timesasked.items(), key=lambda item: item[1],reverse=True))
  n = math.ceil(math.sqrt(self.n_agents))
  sources = []
  for key,value in self.timesasked.items():
    self.amountasked[key] = value/max(1,self.rounds*self.n_agents)
    if n > 0:
     sources.append(key)
     n = n-1 
  self.creditsources = sources
  common = 0
  different = 0 
  for agent in sources: 
    if agent in self.expertsIn:
      common += 1
    else:
      different += 1
  self.commonsources = common*10
  self.difsources = different*10
  for agent in self.schedule.agents: 
    agent.amasked = self.amountasked.get(agent.unique_id,0)
    if agent.unique_id in self.expertsIn:
      agent.expert = 1
 
# helper action for visualisation of attention 
def attAlignment(self):
  self.aIV = 0
  self.aEV = 0
  self.aFN = 0
  self.aBN = 0
  for value in self.allnoiseselection.values():
    if value == 0:
      self.aIV += 1
    elif value == 2:
      self.aEV += 1
    elif value == 1:
      self.aFN += 1
    else: 
      self.aBN +=1
  self.aIV = 100*self.aIV/max(1,len(self.activeAgents))
  self.aEV = 100*self.aEV/max(1,len(self.activeAgents))
  self.aFN = 100*self.aFN/max(1,len(self.activeAgents))
  self.aBN = 100*self.aBN/max(1,len(self.activeAgents))  

#  aggregate noise produced by agents
def computeThoryvos(self):
  # randomsample = self.agentsIn
  totalnoise = 0
  totalindnoise = 0
  totalnoiseit = 0
  totallongnoise = 0
  totalnetnoise = 0
  totalexp = 0
  for agent, noise in self.fN.items():
    totalnoise = totalnoise + noise
    totalexp = totalexp + self.expNoiseUrg
  for agent, indnoise in self.iN.items():
    totalindnoise = totalindnoise + indnoise
  for agent, noise in self.inN.items():
    totalnetnoise = totalindnoise + noise 
  for agent, noise in self.intN.items():
    totalnoiseit = totalnoiseit + noise
  self.thoryvos = totalnoise#/len(randomsample)
  self.indThoryvos = totalindnoise#/max(1,len(randomsample))
  self.avgIndTh = self.indThoryvos/max(1,len(self.activeAgents))
  self.itThoryvos = totalnoiseit#/len(randomsample)
  self.netThoryvos = totalnetnoise
  self.expThoryvos = totalexp
  self.longThoryvos = totallongnoise
  if isinstance(self.NagentsIn, np.ndarray):
    nag = self.NagentsIn.item(0)
  else:
    nag = self.NagentsIn
  # print(self.NagentsIn)
  self.totalAgInC[nag] = self.totalAgInC.get(nag,0) + self.TCO/max(1,nag)
  self.totalAgInQ[nag] = self.totalAgInQ.get(nag,0) + self.QoS
  self.AgInTimes[nag] = self.AgInTimes.get(nag,0) + 1
  self.avgAgInC[nag] = self.totalAgInC.get(nag,0)/max(1,self.AgInTimes.get(nag,0))
  self.avgAgInQ[nag] = self.totalAgInQ.get(nag,0)/max(1,self.AgInTimes.get(nag,0))
  self.stdii = statistics.stdev(list(self.iN.values()))
  self.stdfi = statistics.stdev(list(self.inN.values()))
  self.stdbi = statistics.stdev(list(self.intN.values()))
  self.stdei = statistics.stdev(list(self.xN.values()))
  self.stdsel = statistics.stdev(list(self.fN.values()))
  self.Eii = statistics.mean(list(self.iN.values()))
  self.Efi = statistics.mean(list(self.inN.values()))
  self.Ebi = statistics.mean(list(self.intN.values()))
  self.Eei = statistics.mean(list(self.xN.values()))
  self.Esel = statistics.mean(list(self.fN.values()))
