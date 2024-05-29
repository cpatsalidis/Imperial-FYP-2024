import random
import src.Helpers.init_helpers as ih
import math
import src.Helpers.updates as up
import statistics
from src.Units.observer import Observer
import numpy as np
from scipy.stats import uniform


# This function initializes a dictionary of trust values for a list of neighboring nodes.
# Each node is assigned an initial trust value of 0.5.
def initialise_trust(node_list):
    init_trust = {}
    # Set initial trust value for each node in the list to 0.5
    for i in node_list:
      init_trust[i] = 0.5    
    return init_trust
  
# This method sets up initial trust and confidence values for each agent in the simulation.
def init_trust(self):
  for agent in self.schedule.agents:
    agent.oneneighbors,agent.neighbors = ih.find_neighbors(agent) # Find and assign the first/6th-degree neighbors of the agent
    agent.trust =  initialise_trust(agent.oneneighbors) # Initialize trust for the agent's one-degree neighbors (0.5 for all)

    # Set initial trust towards noise and self-confidence (0.5 for all)
    agent.trustNoise = 0.5
    agent.selfconfidence = 0.5

def initialize_priors(self):
  for agent in self.schedule.agents:
    for other_agent in self.schedule.agents:
        if other_agent.unique_id != agent.unique_id:
            mean = np.random.uniform(0.4, 0.6)  # Initial mean for prior belief with slight variation
            variance = np.random.uniform(0.8, 1.2)  # Initial variance for prior belief with slight variation
            agent.priors[other_agent.unique_id] = (mean, variance)
            agent.beliefs[other_agent.unique_id] = mean
            agent.kalman_states[other_agent.unique_id] = (mean, variance)  # Initialize Kalman filter state


def update_beliefs(self):
    for agent in self.schedule.agents:
        for other_agent_id in self.iN:
            if agent.unique_id != other_agent_id:
                agent.observed_noises[other_agent_id] = self.iN.get(other_agent_id, 0) + np.random.normal(0, self.noise_variance)  # Add noise to observed noise
            else:
                agent.observed_noises[other_agent_id] = self.iN.get(other_agent_id, 0)

    for agent in self.schedule.agents:
        for other_agent_id, observed_noise in agent.observed_noises.items():
            if agent.unique_id != other_agent_id:
                
                previous_belief = agent.beliefs.get(other_agent_id, 0.5)
                agent.beliefs[other_agent_id] += 0.5 * (observed_noise - previous_belief)
                # prior_mean, prior_variance = agent.kalman_states.get(other_agent_id, (0.5, 1.0))

                # kalman_gain = prior_variance / (prior_variance + self.noise_variance)
                # posterior_mean = prior_mean + kalman_gain * (observed_noise - prior_mean)
                # posterior_variance = (1 - kalman_gain) * prior_variance

                # # Update agent's beliefs and Kalman filter state
                # agent.beliefs[other_agent_id] = posterior_mean
                # agent.kalman_states[other_agent_id] = (posterior_mean, posterior_variance)

                error = (agent.beliefs[other_agent_id] - self.iN.get(other_agent_id, 0)) ** 2
                self.sqrt_errors[agent.unique_id][other_agent_id] = error

            else:
                self.sqrt_errors[agent.unique_id][other_agent_id] = 0


def compute_epoch_statistics(self):
        mean_errors = []
        for agent_id, errors in self.sqrt_errors.items():
            if errors:
                rmse = np.sqrt(np.sum(list(errors.values())) / len(errors))
                mean_errors.append(rmse)
        
        # Compute mean of the mean errors for this epoch
        self.epoch_mean_error = np.mean(mean_errors)
        
        # Compute standard deviation of the mean errors for this epoch
        self.epoch_std_error = np.std(mean_errors)


# Initialise network properties in every round
def initialiseRound(self):
  # self.rounds += 1
  self.activeAgents = []
  for agent in self.schedule.agents:
    self.activeAgents.append(agent.unique_id) # Append each agent's unique identifier to the activeAgents list

    # Update the count of rounds each agent has been alive
    self.allRoundsAlive[agent.unique_id] = self.allRoundsAlive.get(agent.unique_id,0) + 1 
    agent.oneneighbors,agent.neighbors = ih.find_neighbors(agent) # Find and assign the first/6th-degree neighbors of the agent
    agent.trust = up.add_trust(agent) 

# Generate jobs randomly with size from given range 
def genJobs(self):
  self.allJobs = {}
  self.allPrios = {}
  self.allUrg = {}
  for agent in self.schedule.agents: 
    agent.jobSize = random.choice([1,agent.averagedemand-1,agent.averagedemand,agent.averagedemand+1]) # Set agent job size around average demand
    agent.jobPriority = random.choice([0,agent.averagepriority-1,agent.averagepriority,agent.averagepriority+1]) # Set agent job priority around average priority
    agent.jobUrgency = random.randint(0,self.maxJobSize) # Set agent job urgency to a random value between 0 and maxJobSize
    agent.totalJobS = agent.totalJobS + agent.jobSize # Update total job size for the agent
    agent.totalJobP = agent.totalJobP + agent.jobPriority # Update total job priority for the agent
    self.allJobs[agent.unique_id] = agent.jobSize # Add agent job size to a dictionary that holds all agent job sizes
    self.allPrios[agent.unique_id] = agent.jobPriority # Add agent job priority to a dictionary that holds all agent job priorities
    self.allUrg[agent.unique_id] = agent.jobUrgency # Add agent job urgency to a dictionary that holds all agent job urgencies
  self.jobsize = sum(self.allJobs.values()) # Calculate total job size for all agents

# Order jobs based on size (ascending) and urgency (descending) 
# and select the smallest urgent jobs based on the number of agents to be included
def orderNchooseJobs(self): 
  jobsIncr = dict(sorted(self.allJobs.items(), key=lambda item: item[1],reverse=False)) # Sort jobs in ascending order of size
  for ag, js in jobsIncr.items():
    jobsIncr[ag] = self.allUrg.get(ag,0) # Update job size to urgency
  urgord = dict(sorted(jobsIncr.items(), key=lambda item: item[1],reverse=True)) # Sort jobs in descending order of urgency
  smallurg = {} # Dictionary to hold the smallest urgent jobs
  for agent in urgord.keys(): 
    smallurg[agent] = self.allJobs.get(agent,0) # Add agent to dictionary of smallest urgent jobs
  self.allJobs = smallurg.copy() # Update allJobs with the smallest urgent jobs
  n = int(self.NagentsIn) # Number of agents to be included in the jobsIn list
  self.jobsIn = {}
  for ag, job in smallurg.items(): 
    # Add agents to jobsIn list based on the number of agents to be included
    if (n>0): 
      self.jobsIn[ag] = job 
      n = n -1
  self.agentsIn = list(self.jobsIn.keys()) # List of agents included in the jobsIn list

# Process jobs based on decided order (smallest first, urgent first etc.)
def processJobsOrder(self):
  self.iQoS = {} # Individual quality of service
  self.iTCO = {} # Individual total cost
  self.vectorIn = {} # Satus of each agent (included or not)
  totalCompute = 0 # Total compute time
  totalDelay = 0 # Total delay
  totalNumJobs = 0 # Total number of jobs
  self.maxiQoS = {} # Maximum individual quality of service
  tcextra = 0 # Extra compute time
  
  # Iterate through all jobs
  for ag,js in self.allJobs.items(): 
    # For agents NOT included in the list of agents to process jobs
    if ag not in self.agentsIn: 
      self.vectorIn[ag] = 0
      self.iQoS[ag] = 0
      self.iTCO[ag] = 0
      tcextra += js 
      self.maxiQoS[ag] = tcextra 
    # For agents included in the list of agents to process jobs
    else: 
      self.vectorIn[ag] = 1 # Set status of agent to included
      totalCompute += js
      tcextra += js 
      self.maxiQoS[ag] = tcextra
      self.iQoS[ag] = totalCompute
      totalDelay = totalDelay + totalCompute # Update total delay
      totalNumJobs += 1 # Update total number of jobs

  # reconsider TQoS, TCO denominator 
  self.QoS = sum(self.iQoS.values())/max(1,self.NagentsIn) # Average quality of service
  self.TCO = totalCompute + self.fixedcost # Total cost of operation
  # Update individual cost for each agent
  for agent in self.activeAgents: 
    if agent in self.agentsIn:
      self.iTCO[agent] = self.TCO/max(1,self.NagentsIn)
    self.miniTCO[agent] = (tcextra + self.fixedcost)/max(1,len(self.activeAgents)) # Minimum individual cost, irrespective of inclusion
    
# Generate individual noise and Expert noise based on quality of service (delay) and cost 
def iNSr(self):
  self.iN = {} # Individual noise
  urgin = 0
  urgout = 0
  urgall = 0 
  self.outavgNoise = 0 # Average noise for agents NOT included in the list of agents to process jobs
  self.inavgNoise = 0 # Average noise for agents included in the list of agents to process jobs

  # Calculate total, in, and out urgency
  for ag, urg in self.allUrg.items():
    urgall += urg
    if ag in self.agentsIn:
      urgin += urg
    else: 
      urgout += urg 
  urgi = ((urgin-urgout)/urgall) # Urgency index
  avgU = urgall/max(1,len(self.activeAgents)) # Average urgency
  avgQ = sum(self.iQoS.values())/max(1,self.NagentsIn) # Average quality of service
  avgC = self.TCO/max(1,self.NagentsIn) # Average total cost of operation
  maxQ = sum(self.maxiQoS.values())/max(1,len(self.activeAgents)) # Maximum individual quality of service

  # Calculate Expert Noise
  self.expNoise = - (maxQ-avgQ)/avgC 
  self.expNoiseUrg = (self.expNoise - urgi*10)/10

  for agent in self.schedule.agents: 
    if agent.unique_id in self.expertsIn:
      Si = self.expNoiseUrg # Expert agents' individual noise is set to the expert noise
    else: 
      U = self.allUrg.get(agent.unique_id,0) # Get urgency of agent
      if agent.unique_id not in self.agentsIn:
        Si = (U/max(1,avgU)) # Set individual noise for agents NOT included in the list of agents to process jobs
        self.outavgNoise = Si + self.outavgNoise # Update average noise for agents NOT included in the list of agents to process jobs
      else: 
        Q = self.iQoS.get(agent.unique_id,0) # Get individual quality of service
        C = self.iTCO.get(agent.unique_id,0) # Get individual cost of operation
        noise = - (maxQ-Q)/C - (U/max(1,avgU))*10 # Calculate individual noise for agents included in the list of agents to process jobs
        Si = noise/10#noise+self.expNoiseUrg)/2
        self.inavgNoise = Si + self.inavgNoise # Update average noise for agents included in the list of agents to process jobs

    # agent.longiN = agent.longiN+agent.iN
    # Set Individual and Expert noise for each agent
    agent.iN = Si #(-1+2/(1+math.exp(-(Si))))   
    agent.xN = self.expNoiseUrg
    self.iN[agent.unique_id] = agent.iN
    self.xN[agent.unique_id] = self.expNoiseUrg

    # NOT USED
    self.longiN[agent.unique_id] = agent.longiN/max(1,self.rounds)
  self.inavgNoise =  self.inavgNoise/max(1,self.NagentsIn)
  self.outavgNoise = self.outavgNoise/max(1,len(self.activeAgents)-self.NagentsIn)
  # print(self.NagentsIn,max(self.iN.values()),self.expNoiseUrg,min(self.iN.values()))         

# generate individual noise based on quality of service (delay) and cost 
# def iNoiseTest(self):
#   self.iN = {}
#   self.expNoise = ((self.n_agents*self.maxJobSize)/2)/((self.n_agents*self.maxJobSize/2) + self.fixedcost)-self.NagentsIn*self.maxJobSize/(self.NagentsIn*self.maxJobSize + self.fixedcost)
#   for agent in self.schedule.agents: 
#     if agent.unique_id in self.expertsIn:
#       Si = self.expNoise
#     else: 
#       if agent.unique_id not in self.agentsIn:
#         Si = 0
#       else: 
#         Q = self.iQoS.get(agent.unique_id,0)
#         C = self.iTCO.get(agent.unique_id,0)
#         noise = Q/C
#         Si = noise
#     agent.iN = Si #(-1+2/(1+math.exp(-(Si))))  
#     agent.xN = self.expNoise
#     self.iN[agent.unique_id] = agent.iN 

# Foreground noise - Which agent to ask for opinion, FROM NEIGHBOURS (their individual noise, iN)
def netNoise(self):
  self.inN = {} # neighbour noise
  for agent in self.schedule.agents: 
    # rand = random.randint(0,1)
    # If the agent has no neighbors, randomly select an agent to ask for opinion
    if agent.oneneighbors == []: 
      agentchosen = random.choice(self.activeAgents)
      self.inN[agent.unique_id] = agent.observed_noises.get(agentchosen,0)
      agent.last_asked = agentchosen # Set the last agent asked for opinion to the randomly selected agent
      self.agent_interactions[agent.unique_id][agentchosen] += 1  # Update observer's dictionary
    # If the agent has neighbors, select an agent to ask for opinion based on trust values
    else:
      flag = 0 # A flag to indicate if a suitable agent has been found
      agent.trust =  {k: v for k, v in sorted(agent.trust.items(), key=lambda item: item[1], reverse=True)} # Sort the agent's trust dictionary by trust values in descending order
      trust_internal = agent.trust.copy()
      max_value = max(agent.trust.values()) # Find the maximum trust value
      max_keys = [k for k, v in agent.trust.items() if v == max_value] # Find all agents with the maximum trust value
      obs_voi = agent.voi # Value of information


      # Check if multiple agents have the highest trust value
      if (type(max_keys) == list):
        l = len(max_keys)
        for i in range(0,l):
          if flag == 0:
            # Choose a random agent from those with highest trust, and remove him
            key = random.choice(max_keys)
            max_keys.remove(key)
            trust_internal.pop(key)
            p = random.uniform(0, 1) # Generate a random probability
            # If random probability > 0.5 and the chosen agent has non-zero individual noise and is a neighbor, ask for opinion
            if (p > 0.5 and self.iN.get(key,0) !=0 and key in agent.oneneighbors):
              # if (agent.pwtp >= obs_voi):
              #   trusted_sec_n = second_degree(self, self.schedule.agents[key]) # Find the trusted secondary neighbor
              #   self.inN[agent.unique_id] = self.iN.get(trusted_sec_n,0)
              #   agent.last_asked = trusted_sec_n
              #   agent.voi = min(1, agent.voi + (abs(agent.pwtp - obs_voi))/2)
              #   self.voi[agent.unique_id] = agent.voi
              #   agent.suggestion += 1
              #   self.suggestion[agent.unique_id] = agent.suggestion

              agent.voi = max(0, agent.voi - (abs(agent.pwtp - obs_voi))/2)
              self.voi[agent.unique_id] = agent.voi
              self.inN[agent.unique_id] = agent.observed_noises.get(key,0)
              agent.last_asked = key # Set the last agent asked for opinion to the chosen agent
              self.timesasked[key] = self.timesasked.get(key,0) + 1 # Increment the number of times the chosen agent has been asked for opinion
              self.agent_interactions[agent.unique_id][key] += 1  # Update observer's dictionary
              flag = 1 # Indicate a suitable agent has been found

        # If no suitable agent is found in those with maximum trust, search among the rest
        if flag == 0:
          for key, value in trust_internal.items():
            p = random.uniform(0, 1)
            # Similar conditions as above, checking remaining agents
            if (p > 0.5 and flag == 0):
              if (key in agent.oneneighbors and self.iN.get(key,0)!=0):
                self.inN[agent.unique_id] = agent.observed_noises.get(key,0)
                agent.last_asked = key
                self.timesasked[key] = self.timesasked.get(key,0) + 1
                self.agent_interactions[agent.unique_id][key] += 1  # Update observer's dictionary
                #if not evaluated yet

      # If nothing else, iterate over agents and randomly select whom to ask in the order of trust
      else:
        for key, value in agent.trust.items():
          if flag == 0:
            p = random.uniform(0, 1)
            if (p > 0.5):
              if (key in agent.oneneighbors and self.iN.get(key,0)!=0):
                self.inN[agent.unique_id] = agent.observed_noises.get(key,0)
                agent.last_asked = key
                self.timesasked[key] = self.timesasked.get(key,0) + 1
                self.agent_interactions[agent.unique_id][key] += 1  # Update observer's dictionary
                flag = 1

      agent.inN = self.inN.get(agent.unique_id,0) # Set the agent's foreground noise to the selected agent's individual noise

# Background Noise - Which agent to ask for opinion, FROM ALL AGENTS (their individual noise, iN)    
def backNoise(self): 
  self.intN = {} # Background noise
  for agent in self.schedule.agents:
    randomsample = random.choices(self.activeAgents,k=math.ceil(len(self.activeAgents)/5)) # Randomly sample a number of agents (5)
    anoise = {} 
    for agent2 in randomsample:
      anoise[agent2] = agent.observed_noises.get(agent2,0) # Get individual noise of the randomly sampled agents
    avg = sum(anoise.values())/max(1,len(randomsample)) # Calculate the average individual noise of the randomly sampled agents
    agent.intN = avg
    self.intN[agent.unique_id] = agent.intN # Set the agent's background noise to the average individual noise of the randomly sampled agents

# Define a function to determine which noise source an agent should pay attention to based on different trust levels.
def attendNoiseFBIEandR(self):
  self.fN = {}
  self.allnoiseselection = {} # Track the type of noise each agent selects

  for agent in self.schedule.agents:
    # Retrieve trust values for different sources of noise
    fore = agent.trustFN
    back = agent.trustNoise
    own = agent.selfconfidence
    exp = agent.trustExp

    p = random.uniform(0,1)
    # If the probability is greater than 0.3, choose the noise based on maximum trust value
    if p > 0.3: 
      # Attend to Expert noise
      if max(fore,back,own,exp) == exp:
        self.fN[agent.unique_id] = self.expNoiseUrg
        self.allnoiseselection[agent.unique_id] = 2
        agent.Ns = 2
      # Attend to Foreground noise
      elif max(fore,back,own,exp) == fore:
        self.fN[agent.unique_id] = self.inN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = 1
        agent.Ns = 1
      # Attend to Background noise
      elif max(fore,back,own,exp) == back:
        self.fN[agent.unique_id] = self.intN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = -1
        agent.Ns = -1
      # Attend to Individual noise
      else: 
        self.fN[agent.unique_id] = self.iN.get(agent.unique_id,0)
        self.allnoiseselection[agent.unique_id] = 0
        agent.Ns = 0

    # If the random number is below or equal to 0.3, randomly attend to a noise
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
    agent.fN = self.fN.get(agent.unique_id,0) # Set the agent's current noise to the selected noise attention
    self.allCumNoise[agent.unique_id] = self.allCumNoise.get(agent.unique_id,0) + self.fN.get(agent.unique_id,0)
    self.allavgNoise[agent.unique_id] = self.allCumNoise.get(agent.unique_id,0)/max(1,self.allRoundsAlive.get(agent.unique_id,0))
    # print(self.allnoiseselection[agent.unique_id])

  # Calculate the mean and standard deviation of the average noise values across all agents
  self.meanNoise = statistics.mean(list(self.allavgNoise.values()))
  self.stdDNoise = statistics.stdev(list(self.allavgNoise.values()))
  return self.fN

# If the objective is random, select to pay attention to a noise randomly
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

# Identify sources of opinion (i.e. influencers)
def community_sources(self):
  # Sort agents by the number of times they have been asked for their opinion, in descending order of frequency
  self.timesasked = dict(sorted(self.timesasked.items(), key=lambda item: item[1],reverse=True))
  
  # Calculate the number of top sources to consider as significant influencers, based on the square root of the number of agents
  n = math.ceil(math.sqrt(self.n_agents)) 
  sources = []

  for key,value in self.timesasked.items():
    # Normalize the number of times each agent has been asked by the total number of possible interactions (rounds * number of agents)
    self.amountasked[key] = value/max(1,self.rounds*self.n_agents)
    # Collect only the top 'n' agents as significant sources
    if n > 0:
     sources.append(key)
     n = n-1 
  self.creditsources = sources
  common = 0
  different = 0 

  # Count how many of the identified top sources are considered experts and how many are not
  for agent in sources: 
    if agent in self.expertsIn:
      common += 1
    else:
      different += 1
  self.commonsources = common*10
  self.difsources = different*10
  # Update each agent's record with the normalized amount they have been asked for their opinion
  for agent in self.schedule.agents: 
    agent.amasked = self.amountasked.get(agent.unique_id,0) # 
    if agent.unique_id in self.expertsIn:
      agent.expert = 1 # Flag the agent as an expert
 
# Helper action for visualisation of attention 
def attAlignment(self):
  self.aIV = 0
  self.aEV = 0
  self.aFN = 0
  self.aBN = 0
  # Count the number of times the collective payed attention to each noise source
  for value in self.allnoiseselection.values(): 
    if value == 0:
      self.aIV += 1
    elif value == 2:
      self.aEV += 1
    elif value == 1:
      self.aFN += 1
    else: 
      self.aBN +=1

  # Calculate the percentage of attention paid to each noise source
  self.aIV = 100*self.aIV/max(1,len(self.activeAgents)) 
  self.aEV = 100*self.aEV/max(1,len(self.activeAgents))
  self.aFN = 100*self.aFN/max(1,len(self.activeAgents))
  self.aBN = 100*self.aBN/max(1,len(self.activeAgents))  

# Function to calculate and aggregate different types of noise generated by agents.
def computeThoryvos(self):
  # randomsample = self.agentsIn
  totalnoise = 0
  totalindnoise = 0
  totalnoiseit = 0
  totallongnoise = 0
  totalnetnoise = 0
  totalexp = 0

  # Aggregate expert and total noise
  for agent, noise in self.fN.items():
    totalnoise = totalnoise + noise
    totalexp = totalexp + self.expNoiseUrg
  # Aggregate individual noise  
  for agent, indnoise in self.iN.items():
    totalindnoise = totalindnoise + indnoise
  # Aggregate foreground noise
  for agent, noise in self.inN.items():
    totalnetnoise = totalindnoise + noise 
  # Aggregate background noise
  for agent, noise in self.intN.items():
    totalnoiseit = totalnoiseit + noise 

  # Compute overall noise and related statistics  
  self.thoryvos = totalnoise#/len(randomsample)
  self.indThoryvos = totalindnoise#/max(1,len(randomsample))
  self.avgIndTh = self.indThoryvos/max(1,len(self.activeAgents))
  self.itThoryvos = totalnoiseit#/len(randomsample)
  self.netThoryvos = totalnetnoise
  self.expThoryvos = totalexp
  self.longThoryvos = totallongnoise

  # If the number of agents in the network is stored as a numpy array, handle it appropriately.
  if isinstance(self.NagentsIn, np.ndarray):
    nag = self.NagentsIn.item(0)
  else:
    nag = self.NagentsIn
  # print(self.NagentsIn)
  self.totalAgInC[nag] = self.totalAgInC.get(nag,0) + self.TCO/max(1,nag) # Aggregate total cost of operation per agent
  self.totalAgInQ[nag] = self.totalAgInQ.get(nag,0) + self.QoS # Aggregate quality of service per agent 
  self.AgInTimes[nag] = self.AgInTimes.get(nag,0) + 1 # Increment the number of times the agent has been included in the list of agents to process jobs
  self.avgAgInC[nag] = self.totalAgInC.get(nag,0)/max(1,self.AgInTimes.get(nag,0)) # Calculate average total cost of operation per agent
  self.avgAgInQ[nag] = self.totalAgInQ.get(nag,0)/max(1,self.AgInTimes.get(nag,0)) # Calculate average quality of service per agent

  # Calculate the mean and standard deviation of each noise type and attention to noise
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

def second_degree(self, agent):
  if agent.oneneighbors == []: 
    chosen_key = random.choice(self.activeAgents)
  else:
    total = sum(agent.trust.values())
    probabilities = {key: value / total for key, value in agent.trust.items()}

    # Choose a key based on the computed probabilities.
    chosen_key = random.choices(list(probabilities.keys()), weights=probabilities.values(), k=1)[0]
    return chosen_key
    
