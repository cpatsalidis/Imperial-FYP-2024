# Update attention based on individual approach (update = 'ind')
def updAttNInd(self):
  # Iterate through all agents managed by the scheduler
  for agent in self.schedule.agents:
    # done = self.allnoiseselection.get(agent.unique_id,0)
    
    # Calculate the difference between current and previous individual noise 
    dif = self.iN.get(agent.unique_id,0) - self.iNprev.get(agent.unique_id,0) 

    # Determine the type of noise the agent was attending to based on their selection (0 = own, 1 = foreground, 2 = expert, -1 = background)
    # Based on the performance of the noise that the agent attended to during the last round, we update the trust of the agent
    # to the noise type they were attending to
    if self.allnoiseselection.get(agent.unique_id,0) == 0:
      # If the difference in noise is positive (Noise increase), reduce SELF-CONFIDENCE proportionally; otherwise, increase it
      if dif > 0: 
        agent.selfconfidence = max(0,agent.selfconfidence - agent.selfconfidence*agent.c)
      elif dif < 0:
        agent.selfconfidence = min(1,agent.selfconfidence + agent.selfconfidence*agent.c)
    # If the difference in noise is positive (Noise increase), reduce FOREGROUND TRUST proportionally; otherwise, increase it
    elif self.allnoiseselection.get(agent.unique_id,0) == 1:
      if dif > 0: 
        agent.trustFN = max(0,agent.trustFN-agent.trustFN*agent.c)
        agent.pwtp = max(0,agent.pwtp+agent.pwtp*0.2)
      elif dif < 0:   
        agent.trustFN = min(1,agent.trustFN+agent.trustFN*agent.c)
        agent.pwtp = min(1,agent.pwtp-dif*0.2)
        self.pwtp[agent.unique_id] = agent.pwtp
    # If the difference in noise is positive (Noise increase), reduce EXPERT TRUST proportionally; otherwise, increase it
    elif self.allnoiseselection.get(agent.unique_id,0) == 2: 
      if dif > 0:
        agent.trustExp = max(0,agent.trustExp - agent.trustExp*agent.c)
      elif dif < 0:
        agent.trustExp = min(1,agent.trustExp + agent.trustExp*agent.c)
    # If the difference in noise is positive (Noise increase), reduce BACKGROUND TRUST proportionally; otherwise, increase it
    else: 
      if dif > 0:
        agent.trustNoise = max(0,agent.trustNoise - agent.trustNoise*agent.c)
      elif dif < 0:
        agent.trustNoise = min(1,agent.trustNoise + agent.trustNoise*agent.c)

    # If the noise difference is positive, decrease trust in the last agent asked; otherwise, increase it
    # This part updates the trust based solely on the last interaction in the network context
    if dif > 0: 
      agent.trust[agent.last_asked] = max(0,agent.trust.get(agent.last_asked,0)-agent.trust.get(agent.last_asked,0)*agent.c)
    elif dif < 0:   
      agent.trust[agent.last_asked] = min(1,agent.trust.get(agent.last_asked,0)+agent.trust.get(agent.last_asked,0)*agent.c)

# Update attention based on collective approach (update = 'col')
def updAttNCom(self):
  for agent in self.schedule.agents:
    # done = self.allnoiseselection.get(agent.unique_id,0)
    
    # Calculate the absolute differences between average expression of the agents, and each noise type.
    # the trust to the voice that deviates less from the average expression of the agents is increased
    # and the one that deviates the most is decreased.
    difown = abs((self.indThoryvos/len(self.activeAgents)) - self.iN.get(agent.unique_id,0))
    diffore = abs((self.indThoryvos/len(self.activeAgents)) - self.inN.get(agent.unique_id,0))    
    difback = abs((self.indThoryvos/len(self.activeAgents)) - self.intN.get(agent.unique_id,0))
    difexp = abs((self.indThoryvos/len(self.activeAgents)) - self.expNoiseUrg)

    # Identify the noise type that deviates the most and the least from the average expression of the agents
    if max(difown,diffore,difback,difexp) == difown:
      agent.selfconfidence = max(0,agent.selfconfidence - agent.selfconfidence*agent.c)
    elif max(difown,diffore,difback,difexp) == diffore:
      agent.trustFN = max(0,agent.trustFN-agent.trustFN*agent.c)
      agent.pwtp = min(1,agent.pwtp-diffore*0.2)
      self.pwtp[agent.unique_id] = agent.pwtp
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
    
    # Adjust the trust for the last asked agent based on how their forward noise deviation compares to the average expression/10
    if diffore > ((self.indThoryvos/len(self.activeAgents))/10): 
      agent.trust[agent.last_asked] = max(0,agent.trust.get(agent.last_asked,0)-agent.trust.get(agent.last_asked,0)*agent.c)
    else:   
      agent.trust[agent.last_asked] = min(1,agent.trust.get(agent.last_asked,0)+agent.trust.get(agent.last_asked,0)*agent.c)

# def updAttNComNew(self):
#   # the update is only positive
#   for agent in self.schedule.agents:
#     # done = self.allnoiseselection.get(agent.unique_id,0)
#     #good action
#     difown = abs((self.thoryvos/len(self.activeAgents)) - self.iN.get(agent.unique_id,0))
#     diffore = abs((self.thoryvos/len(self.activeAgents)) - self.inN.get(agent.unique_id,0))    
#     difback = abs((self.thoryvos/len(self.activeAgents)) - self.intN.get(agent.unique_id,0))
#     difexp = abs((self.thoryvos/len(self.activeAgents)) - self.expNoiseUrg)
#     if max(difown,diffore,difback,difexp) == difown:
#       agent.selfconfidence = max(0,agent.selfconfidence - agent.selfconfidence*agent.c)
#     elif max(difown,diffore,difback,difexp) == diffore:
#       agent.trustFN = max(0,agent.trustFN-agent.trustFN*agent.c)
#     elif max(difown,diffore,difback,difexp) == difback:
#       agent.trustNoise = max(0,agent.trustNoise - agent.trustNoise*agent.c)
#     elif max(difown,diffore,difback,difexp) == difexp:
#       agent.trustExp = max(0,agent.trustExp - agent.trustExp*agent.c)
#     if min(difown,diffore,difback,difexp) == difown:
#        agent.selfconfidence = min(1,agent.selfconfidence + agent.selfconfidence*agent.c)
#     elif min(difown,diffore,difback,difexp) == diffore:
#       agent.trustFN = min(1,agent.trustFN+agent.trustFN*agent.c)
#     elif min(difown,diffore,difback,difexp) == difback:
#       agent.trustNoise = min(1,agent.trustNoise + agent.trustNoise*agent.c)
#     elif min(difown,diffore,difback,difexp) == difexp:
#       agent.trustExp = min(1,agent.trustExp + agent.trustExp*agent.c)
#     if diffore > ((self.thoryvos/len(self.activeAgents))/10): 
#       agent.trust[agent.last_asked] = max(0,agent.trust.get(agent.last_asked,0)-agent.trust.get(agent.last_asked,0)*agent.c)
#     else:   
#       agent.trust[agent.last_asked] = min(1,agent.trust.get(agent.last_asked,0)+agent.trust.get(agent.last_asked,0)*agent.c)
 
# Update attention based on expert approach (update = 'exp')
def updAttNExp(self):
  for agent in self.schedule.agents:
    # done = self.allnoiseselection.get(agent.unique_id,0)
   
    # Calculate the absolute differences between expert noise, and each noise type.
    # the trust to the voice that deviates less from the expert noise is increased
    # and the one that deviates the most is decreased.   
    difown = abs(self.expNoiseUrg - self.iN.get(agent.unique_id,0))
    diffore = abs(self.expNoiseUrg - self.inN.get(agent.unique_id,0))    
    difback = abs(self.expNoiseUrg - self.intN.get(agent.unique_id,0))
    difexp = abs(self.expNoiseUrg - self.expNoiseUrg)

    # Identify the noise type that deviates the most and the least from the expert noise
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

    # Adjust the trust for the last asked agent based on how their foreground noise deviation compares to zero (Increase if agent trusted is an expert)
    if diffore > 0: 
      agent.trust[agent.last_asked] = max(0,agent.trust.get(agent.last_asked,0)-agent.trust.get(agent.last_asked,0)*agent.c)
    else:  
      agent.trust[agent.last_asked] = min(1,agent.trust.get(agent.last_asked,0)+agent.trust.get(agent.last_asked,0)*agent.c)

# Update attention based on collective approach (update = 'col')
def updAttNLie(self):
  for agent in self.schedule.agents:
    # done = self.allnoiseselection.get(agent.unique_id,0)
    
    # Calculate the absolute differences between average expression of the agents, and each noise type.
    # the trust to the voice that deviates less from the average expression of the agents is increased
    # and the one that deviates the most is decreased.
    difown = abs((self.realThoryvos/len(self.activeAgents)) - self.iN.get(agent.unique_id,0))
    diffore = abs((self.realThoryvos/len(self.activeAgents)) - self.inN.get(agent.unique_id,0))    
    difback = abs((self.realThoryvos/len(self.activeAgents)) - self.intN.get(agent.unique_id,0))
    difexp = abs((self.realThoryvos/len(self.activeAgents)) - self.expNoiseUrg)



    # Identify the noise type that deviates the most and the least from the average expression of the agents
    if max(difown,diffore,difback,difexp) == difown:
      agent.selfconfidence = max(0,agent.selfconfidence - agent.selfconfidence*agent.c)
    elif max(difown,diffore,difback,difexp) == diffore:
      agent.trustFN = max(0,agent.trustFN-agent.trustFN*agent.c)
      agent.pwtp = min(1,agent.pwtp-diffore*0.2)
      self.pwtp[agent.unique_id] = agent.pwtp
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
    


def add_trust(self):
  # add trust to all neighbors
  for agent in self.oneneighbors: 
    if agent not in self.trust.keys(): 
      self.trust[agent] = 0.5
  return self.trust
