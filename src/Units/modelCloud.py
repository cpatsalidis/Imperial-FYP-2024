from mesa import Model
import src.Helpers.init_helpers as ip
import  src.Helpers.thoryvos2 as th
import src.Units.observer as ob
from stable_baselines3 import DQN
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
    self.lying_fraction = random.uniform(0.2, 0.3)
    ip.initVarCloud(self)
    ip.init_collectors_cloud(self)
    ip.convert_agents_to_liars(self)
    
  def step(self):
    env = th.SoSPole(self) # Create an instance of the SoSPole environment with the current model configuration.
    observation = env.reset() # Reset the environment to its initial state and get the initial observation.
    # Initialize a DQN model.
    modeldqn = DQN("MlpPolicy", env, verbose=0, exploration_fraction=0.1, exploration_final_eps=0.1, exploration_initial_eps=1.0, learning_starts=1000)
    modeldqn.learn(total_timesteps=self.totallearn) # Start the learning process over the specified number of total timesteps.
    observation = env.reset() # Reset the environment again to start a new series of steps or episodes.
      # observation, reward, done, info = env.step(action)
    for _ in range(self.totalrounds):
      action, _states = modeldqn.predict(observation, deterministic=True)
      observation, reward, done, info = env.step(action)