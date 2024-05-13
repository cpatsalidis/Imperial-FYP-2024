from mesa import Model
import src.Helpers.init_helpers as ip
import  src.Helpers.thoryvos2 as th
from stable_baselines3 import A2C
import random

class Environment(Model):
  def __init__(self,n_agents,n_thoryvos,m,miu,totalrounds,m_t,maxJobSize,totallearn,baseline,update,reward):
    random.seed(42)
    self.n_agents = n_agents
    self.miu = miu
    self.m = m
    self.n_thoryvos = n_thoryvos #svm classifier
    self.start_state = 1
    self.rounds = 1
    self.m_t = m_t
    self.baseline = baseline
    self.totalrounds = totalrounds
    self.totallearn = totallearn
    self.maxJobSize = maxJobSize
    self.avgDelay = self.n_agents*(self.maxJobSize-1)/2#(self.maxJobSize/2)*(self.n_agents+1)/2 #from arithmetic series
    self.remaining = self.totalrounds - self.rounds
    self.existingagents = list(range(0,n_agents-1))
    self.activeAgents = list(range(self.n_agents))
    self.update = update
    self.rewardinp = reward
    self.G = ip.klemm_eguilez_network(self.n_agents,self.m,self.miu) #one unique network
    self.nextunique = self.n_agents
    ip.initVarCloud(self)
    ip.init_collectors_cloud(self)
    
  def step(self):
    env = th.SoSPole(self)
    observation = env.reset()
    modeldqn = A2C("MlpPolicy", env, verbose=0)
    modeldqn.learn(total_timesteps=self.totallearn)
    observation = env.reset()
    
    for _ in range(self.totalrounds):
      action, _states = modeldqn.predict(observation, deterministic=True)
      observation, reward, done, info = env.step(action)
    