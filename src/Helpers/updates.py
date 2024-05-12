# update attention based on individual approach (exp.param: update = 'ind')
def updAttNInd(self):
  # the update is only positive
  for agent in self.schedule.agents:
    # done = self.allnoiseselection.get(agent.unique_id,0)
    #good action
    dif = self.iN.get(agent.unique_id,0) - self.iNprev.get(agent.unique_id,0) 
    if self.allnoiseselection.get(agent.unique_id,0) == 0:
      if dif > 0: 
        agent.selfconfidence = max(0,agent.selfconfidence - agent.selfconfidence*agent.c)
      elif dif < 0:
        agent.selfconfidence = min(1,agent.selfconfidence + agent.selfconfidence*agent.c)
    elif self.allnoiseselection.get(agent.unique_id,0) == 1:
      if dif > 0: 
        agent.trustFN = max(0,agent.trustFN-agent.trustFN*agent.c)
      elif dif < 0:   
        agent.trustFN = min(1,agent.trustFN+agent.trustFN*agent.c)
    elif self.allnoiseselection.get(agent.unique_id,0) == 2: 
      if dif > 0:
        agent.trustExp = max(0,agent.trustExp - agent.trustExp*agent.c)
      elif dif < 0:
        agent.trustExp = min(1,agent.trustExp + agent.trustExp*agent.c)
    else: 
      if dif > 0:
        agent.trustNoise = max(0,agent.trustNoise - agent.trustNoise*agent.c)
      elif dif < 0:
        agent.trustNoise = min(1,agent.trustNoise + agent.trustNoise*agent.c)
    #update for network only
    if dif > 0: 
      agent.trust[agent.last_asked] = max(0,agent.trust.get(agent.last_asked,0)-agent.trust.get(agent.last_asked,0)*agent.c)
    elif dif < 0:   
      agent.trust[agent.last_asked] = min(1,agent.trust.get(agent.last_asked,0)+agent.trust.get(agent.last_asked,0)*agent.c)

# update attention based on collective approach (exp.param: update = 'col')
def updAttNCom(self):
  # the update is only positive
  for agent in self.schedule.agents:
    # done = self.allnoiseselection.get(agent.unique_id,0)
    #good action
    difown = abs((self.indThoryvos/len(self.activeAgents)) - self.iN.get(agent.unique_id,0))
    diffore = abs((self.indThoryvos/len(self.activeAgents)) - self.inN.get(agent.unique_id,0))    
    difback = abs((self.indThoryvos/len(self.activeAgents)) - self.intN.get(agent.unique_id,0))
    difexp = abs((self.indThoryvos/len(self.activeAgents)) - self.expNoiseUrg)
    if max(difown,diffore,difback,difexp) == difown:
      agent.selfconfidence = max(0,agent.selfconfidence - agent.selfconfidence*agent.c)
    elif max(difown,diffore,difback,difexp) == diffore:
      agent.trustFN = max(0,agent.trustFN-agent.trustFN*agent.c)
    elif max(difown,diffore,difback,difexp) == difback:
      agent.trustNoise = max(0,agent.trustNoise - agent.trustNoise*agent.c)
    elif max(difown,diffore,difback,difexp) == difexp:
      agent.trustExp = max(0,agent.trustExp - agent.trustExp*agent.c)
    if min(difown,diffore,difback,difexp) == difown:
       agent.selfconfidence = min(1,agent.selfconfidence + agent.selfconfidence*agent.c)
    elif min(difown,diffore,difback,difexp) == diffore:
      agent.trustFN = min(1,agent.trustFN+agent.trustFN*agent.c)
    elif min(difown,diffore,difback,difexp) == difback:
      agent.trustNoise = min(1,agent.trustNoise + agent.trustNoise*agent.c)
    elif min(difown,diffore,difback,difexp) == difexp:
      agent.trustExp = min(1,agent.trustExp + agent.trustExp*agent.c)
    if diffore > ((self.indThoryvos/len(self.activeAgents))/10): 
      agent.trust[agent.last_asked] = max(0,agent.trust.get(agent.last_asked,0)-agent.trust.get(agent.last_asked,0)*agent.c)
    else:   
      agent.trust[agent.last_asked] = min(1,agent.trust.get(agent.last_asked,0)+agent.trust.get(agent.last_asked,0)*agent.c)

def updAttNComNew(self):
  # the update is only positive
  for agent in self.schedule.agents:
    # done = self.allnoiseselection.get(agent.unique_id,0)
    #good action
    difown = abs((self.thoryvos/len(self.activeAgents)) - self.iN.get(agent.unique_id,0))
    diffore = abs((self.thoryvos/len(self.activeAgents)) - self.inN.get(agent.unique_id,0))    
    difback = abs((self.thoryvos/len(self.activeAgents)) - self.intN.get(agent.unique_id,0))
    difexp = abs((self.thoryvos/len(self.activeAgents)) - self.expNoiseUrg)
    if max(difown,diffore,difback,difexp) == difown:
      agent.selfconfidence = max(0,agent.selfconfidence - agent.selfconfidence*agent.c)
    elif max(difown,diffore,difback,difexp) == diffore:
      agent.trustFN = max(0,agent.trustFN-agent.trustFN*agent.c)
    elif max(difown,diffore,difback,difexp) == difback:
      agent.trustNoise = max(0,agent.trustNoise - agent.trustNoise*agent.c)
    elif max(difown,diffore,difback,difexp) == difexp:
      agent.trustExp = max(0,agent.trustExp - agent.trustExp*agent.c)
    if min(difown,diffore,difback,difexp) == difown:
       agent.selfconfidence = min(1,agent.selfconfidence + agent.selfconfidence*agent.c)
    elif min(difown,diffore,difback,difexp) == diffore:
      agent.trustFN = min(1,agent.trustFN+agent.trustFN*agent.c)
    elif min(difown,diffore,difback,difexp) == difback:
      agent.trustNoise = min(1,agent.trustNoise + agent.trustNoise*agent.c)
    elif min(difown,diffore,difback,difexp) == difexp:
      agent.trustExp = min(1,agent.trustExp + agent.trustExp*agent.c)
    if diffore > ((self.thoryvos/len(self.activeAgents))/10): 
      agent.trust[agent.last_asked] = max(0,agent.trust.get(agent.last_asked,0)-agent.trust.get(agent.last_asked,0)*agent.c)
    else:   
      agent.trust[agent.last_asked] = min(1,agent.trust.get(agent.last_asked,0)+agent.trust.get(agent.last_asked,0)*agent.c)
 
# update attention based on expert approach (exp.param: update = 'exp')
def updAttNExp(self):
  # the update is only positive
  for agent in self.schedule.agents:
    # done = self.allnoiseselection.get(agent.unique_id,0)
    #good action
    difown = abs(self.expNoiseUrg - self.iN.get(agent.unique_id,0))
    diffore = abs(self.expNoiseUrg - self.inN.get(agent.unique_id,0))    
    difback = abs(self.expNoiseUrg - self.intN.get(agent.unique_id,0))
    difexp = abs(self.expNoiseUrg - self.expNoiseUrg)
    if max(difown,diffore,difback,difexp) == difown:
      agent.selfconfidence = max(0,agent.selfconfidence - agent.selfconfidence*agent.c)
    elif max(difown,diffore,difback,difexp) == diffore:
      agent.trustFN = max(0,agent.trustFN-agent.trustFN*agent.c)
    elif max(difown,diffore,difback,difexp) == difback:
      agent.trustNoise = max(0,agent.trustNoise - agent.trustNoise*agent.c)
    elif max(difown,diffore,difback,difexp) == difexp:
      agent.trustExp = max(0,agent.trustExp - agent.trustExp*agent.c)
    if min(difown,diffore,difback,difexp) == difown:
       agent.selfconfidence = min(1,agent.selfconfidence + agent.selfconfidence*agent.c)
    elif min(difown,diffore,difback,difexp) == diffore:
      agent.trustFN = min(1,agent.trustFN+agent.trustFN*agent.c)
    elif min(difown,diffore,difback,difexp) == difback:
      agent.trustNoise = min(1,agent.trustNoise + agent.trustNoise*agent.c)
    elif min(difown,diffore,difback,difexp) == difexp:
      agent.trustExp = min(1,agent.trustExp + agent.trustExp*agent.c)
    if diffore > 0: 
      agent.trust[agent.last_asked] = max(0,agent.trust.get(agent.last_asked,0)-agent.trust.get(agent.last_asked,0)*agent.c)
    else:   
      agent.trust[agent.last_asked] = min(1,agent.trust.get(agent.last_asked,0)+agent.trust.get(agent.last_asked,0)*agent.c)

def add_trust(self):
  for agent in self.oneneighbors:
    if agent not in self.trust.keys():
      self.trust[agent] = 0.5
  return self.trust
