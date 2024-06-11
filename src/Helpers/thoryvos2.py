import gym
import numpy as np
import src.Helpers.processesCloud as procC
import src.Helpers.updates as up
import src.Helpers.init_helpers as ih
import src.Units.observer as obs
import gym.spaces
import random
import statistics
import math 

# Update the state of the environment with the jobs, urgency and selection status of each agent
def state_update_dyn(state,model):
  for agent,th in model.allJobs.items(): # Update state with the jobs of each agent
    state[agent][0] = th
  for agent,urg in model.allUrg.items(): # Update state with the urgency of each agent
    state[agent][1] = urg
  for agent,n in model.vectorIn.items(): # Update state with the selection status of each agents' job (0 or 1)
    state[agent][2] = n
  return state


# Update the number of agents to be active in the next iteration based on the original Baseline parameter 
def action_update(action,model):
  myaction = 'rl'
  if model.baseline == 0:
    myaction = 'random'
  elif model.baseline == -1:
    myaction = 'same'
  if myaction == 'random':
    model.NagentsIn = random.randint(1,model.n_agents) # Randomly select the number of agents to be active in the next iteration
  elif myaction == 'same':
    model.NagentsIn = model.NagentsIn # Keep the number of agents the same as the previous iteration
  else: 
    model.NagentsIn = action # Use the action from the RL model

  # Ensure that the number of agents is not 0
  if model.NagentsIn == 0: 
    model.NagentsIn = 1
  # model.NagentsIn = min(model.rounds,99)
  return 

# Reset the model to its initial state
def reset_mamodel(self):
  self.G = ih.klemm_eguilez_network(self.n_agents,self.m,self.miu) 
  # self.rounds = 0
  procC.init_trust(self) # initialise trust to neighbours, noise and self-confidence
  procC.initialize_priors(self) # initialise priors
  self.allJobs = {} # dictionary of jobs for each agent
  self.allPrios = {} # dictionary of priorities for each agent
  self.allUrg = {} # dictionary of urgencies for each agent
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
  self.suggestion = {} # suggestion for the next agent to ask
  self.meanNoise = 0
  self.stdNoise = 0
  self.stdDis = 0
  self.TCO = {} #total cost
  self.QoS = {} #total quality (numofjobs/delay)
  self.iQoS = {} #individual quality
  self.iTCO = {} # individual cost
  self.maxiQoS = {}
  self.agentOrder = {}
  self.iN = {}
  self.iNprev = {}
  self.longiN = {}
  self.intN = {}
  self.fN = {}
  self.fintN = {}
  self.orderJobs = {}
  self.thoryvos = 0
  self.indThoryvos = 0
  self.netThoryvos = 0
  self.itThoryvos = 0
  self.longThoryvos = 0
  self.pred = 0
  self.reward = 0
  self.cumreward = 0
  self.jobsize = 0
  self.agentsIn = []
  self.jobsIn = {}
  self.agent_interactions = {i: {j: 0 for j in range(self.n_agents) if i != j} for i in range(self.n_agents)}
 
# Main look of the game, generates jobs of agents, calls all the functions related to the voices, 
# selects the voice based on the experimental parameters, forms collective expression,
# calls the functions for the updates of the attention of each agent to the voices and stores the
# variables for next iteration and visualisations
def workDynB(state,model,action):
  model.schedule.step() # Activates the agent and stages any necessary changes, but does not apply them yet
  action_update(action,model)
  model.iNprev = model.iN # Store the previous individual noise
  procC.initialiseRound(model) # Update active agents, neighbours and trust
  procC.genJobs(model) # Generate jobs, priority and urgency for each agent
  procC.orderNchooseJobs(model) # Select the agents whose jobs are to be processed, based on the number of agents to be active
  procC.processJobsOrder(model) # Process the jobs of the selected/not selected agents (Delay, cost)
  state = state_update_dyn(state,model) # Update the state of the environment (Jobs, urgency and selection status of each agent)
  procC.iNSr(model) # Compute individual and expert noise
  procC.update_beliefs(model) 
  procC.netNoise(model) # Compute foreground noise
  procC.backNoise(model) # Compute background noise

  # Select a voice to attend to based on the update parameter
  # if setting is to attend to random noise, choose noise randomly
  if model.update == 'ran': 
    model.fN = procC.attendRandom(model)
  else: 
    # otherwise select noise based on the parameter of update
    model.fN = procC.attendNoiseFBIEandR(model)

  procC.computeThoryvos(model) # Compute noise levels and statistical metrics (mean and standard deviation of noise types)
  # update attention to noise based on the update parameter
  if model.update == 'ind': 
    up.updAttNInd(model)
  elif model.update == 'com': 
    up.updAttNCom(model)
  else: 
    up.updAttNExp(model)
  procC.compute_epoch_statistics(model)
  procC.community_sources(model) # Identify top 'n' sources of influence
  procC.attAlignment(model) # Visualise attention
  model.datacollector.collect(model)
  model.datacollector2.collect(model)
  model.datacollector3.collect(model)
  model.datacollector4.collect(model)
  model.datacollector5.collect(model)
  model.datacollector6.collect(model)
  model.datacollector7.collect(model)
  return state

# Reward for regulator based on the initial reward parameter 
def rewardDyn(model):
  if model.rewardinp == 'exp':
    model.reward = -model.expNoiseUrg#/max(1,model.NagentsIn) # Reward based on the expert noise
  elif model.rewardinp == 'com':
    model.reward = -model.indThoryvos # Reward based on the total individual noise
  elif model.rewardinp == 'uni':
    model.reward = -model.thoryvos # Reward based on the total noise
  elif model.rewardinp == 'fore':
    model.reward = -model.netThoryvos # Reward based on the foreground noise
  else: 
    model.reward = -statistics.stdev(list(model.iN)) # Reward based on the standard deviation of the individual noise

#MAIN CLASS OF THE ENVIRONMENT - Custom environment based on gym library
class SoSPole(gym.Env):
  def __init__(self, model,max_steps=2000): 
    self.observation_shape = (model.n_agents,3) # Define the shape of the observation space based on the number of agents and each having 3 features (job, urgency, selection status)
    # Define the observation space as a Box space, which is suitable for the continuous state space where each agent has 3 observable features
    self.observation_space = gym.spaces.Box(low = np.zeros(self.observation_shape), high = np.full(self.observation_shape,model.maxJobSize),dtype = np.float16)
    # Define the action space. Each action corresponds to the number of active agents in the next iteration
    self.actions = list(range(1,model.n_agents))
    self.action_space = gym.spaces.Discrete(len(self.actions))
    self.log = ''
    self.MAmodel = model
    self.max_steps = max_steps # Set the maximum number of steps the environment can take before resetting
    #self.observer = obs.Observer(model)

  def reset(self):
    # Reset the environment to an initial state
    # Initialize the state of the environment as zeros
    self.state = np.zeros(self.observation_shape)
    reset_mamodel(self.MAmodel)
    self.steps_left = self.max_steps
    action = self.MAmodel.NagentsIn # Get the initial action based on the number of agents in the model
    self.state = workDynB(self.state,self.MAmodel,action) # Update the state of the environment
    return self.state
  
  def step(self, action):
    # Step function to move the environment to the next state based on an action
    # Increment the rounds in the model
    self.MAmodel.rounds = self.MAmodel.rounds+1
    #self.observer.step()
    self.state = workDynB(self.state,self.MAmodel,action) # Update the state of the environment
    # Compute the reward based on the initial reward parameter
    rewardDyn(self.MAmodel) 
    new_score = self.MAmodel.reward
    reward = new_score 
    # Determine if the environment should reset based on the number of steps left
    self.steps_left -= 1
    done = (self.steps_left <= 0)
    # Return the new state, reward, done status, and an empty dictionary for extra info
    return self.state, reward, done, {}
  