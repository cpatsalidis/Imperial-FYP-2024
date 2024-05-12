import gym
import numpy as np
import src.Helpers.processesCloud as procC
import src.Helpers.updates as up
import src.Helpers.init_helpers as ih
import gym.spaces
import random
import statistics
import math 

def state_update_dyn(state,model):
  for agent,th in model.allJobs.items():
    state[agent][0] = th
  for agent,urg in model.allUrg.items():
    state[agent][1] = urg
  for agent,n in model.vectorIn.items():
    state[agent][2] = n
  return state

def action_update(action,model):
  myaction = 'rl'
  if model.baseline == 0:
    myaction = 'random'
  elif model.baseline == -1:
    myaction = 'same'
  if myaction == 'random':
    model.NagentsIn = random.randint(1,model.n_agents)
  elif myaction == 'same':
    model.NagentsIn = model.NagentsIn
  else: 
    model.NagentsIn = action
  if model.NagentsIn == 0:
    model.NagentsIn = 1
  # model.NagentsIn = min(model.rounds,99)
  return 

def reset_mamodel(self):
  self.G = ih.klemm_eguilez_network(self.n_agents,self.m,self.miu) 
  # self.rounds = 0
  procC.init_trust(self)
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
 
#main look of the game, generates jobs of agents, calls all the functions related to the voices, 
# selects the voice based on the experimental parameters, forms collective expression,
# calls the functions for the updates of the attention of each agent to the voices and stores the
# variables for next iteration and visualisations
def workDynB(state,model,action):
  model.schedule.step()
  action_update(action,model)
  model.iNprev = model.iN
  procC.initialiseRound(model)
  procC.genJobs(model)
  procC.orderNchooseJobs(model)
  procC.processJobsOrder(model)
  state = state_update_dyn(state,model)
  procC.iNSr(model)
  procC.netNoise(model)
  procC.backNoise(model)
  # if setting is to attend to random noise, choose noise randomly
  if model.update == 'ran': 
    model.fN = procC.attendRandom(model)
  else: 
    # otherwise select noise based on the parameter of update
    model.fN = procC.attendNoiseFBIEandR(model)
  procC.computeThoryvos(model) 
  # update attention to noise based on expternally determined objective
  if model.update == 'ind': 
    up.updAttNInd(model)
  elif model.update == 'com': 
    up.updAttNCom(model)
  else: 
    up.updAttNExp(model)
  procC.community_sources(model)
  procC.attAlignment(model)
  model.datacollector.collect(model)
  model.datacollector2.collect(model)
  model.datacollector3.collect(model)
  model.datacollector4.collect(model)
  model.datacollector5.collect(model)
  model.datacollector6.collect(model)
  model.datacollector7.collect(model)
  return state

#reward based on the initial params 
def rewardDyn(model):
  if model.rewardinp == 'exp':
    model.reward = -model.expNoiseUrg#/max(1,model.NagentsIn)
  elif model.rewardinp == 'com':
    model.reward = -model.indThoryvos
  elif model.rewardinp == 'uni':
    model.reward = -model.thoryvos
  else: 
    model.reward = -statistics.stdev(list(model.iN))

#MAIN CLASS OF THE ENVIRONMENT - Custom environment based on gym library
class SoSPole(gym.Env):
  def __init__(self, model,max_steps=2000): # agent
    self.observation_shape = (model.n_agents,3)
    self.observation_space = gym.spaces.Box(low = np.zeros(self.observation_shape), high = np.full(self.observation_shape,model.maxJobSize),dtype = np.float16)
    self.actions = list(range(1,model.n_agents))
    self.action_space = gym.spaces.Discrete(len(self.actions))
    self.log = ''
    self.MAmodel = model
    self.max_steps = max_steps

  def reset(self):
    self.state = np.zeros(self.observation_shape)
    reset_mamodel(self.MAmodel)
    self.steps_left = self.max_steps
    action = self.MAmodel.NagentsIn
    self.state = workDynB(self.state,self.MAmodel,action)
    return self.state
  
  # this is the main loop
  def step(self, action):
    self.MAmodel.rounds = self.MAmodel.rounds+1
    self.state = workDynB(self.state,self.MAmodel,action)
    rewardDyn(self.MAmodel)
    new_score = self.MAmodel.reward
    reward = new_score 
    self.steps_left -= 1
    done = (self.steps_left <= 0)
    return self.state, reward, done, {}
  