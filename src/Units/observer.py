# observer.py

import random
import numpy as np

class Observer:
    def __init__(self, model):
        self.model = model
        self.additional_info = self.gather_additional_info()
        self.historical_data = []  # Store historical values
        self.learning_rate = 0.02  # Learning rate for adjustments
        self.influence_factor_urgency = 5  # Factor to adjust influence strength
        self.success_rates = {}  # Success rates of interactions
        
    def gather_additional_info(self):
        # Gather additional information about the system
        additional_info = {}
        additional_info['average_job_size'] = np.mean([agent.jobSize for agent in self.model.schedule.agents])
        additional_info['average_urgency'] = np.mean([agent.jobUrgency for agent in self.model.schedule.agents])
        return additional_info

    def update_historical_data(self):
        # Update historical data with current values
        current_data = {
            'noise_selection': [self.model.allnoiseselection.get(agent.unique_id,0) for agent in self.model.schedule.agents],
        }
        self.historical_data.append(current_data)
        if len(self.historical_data) > 10:  # Limit historical data length
            self.historical_data.pop(0)

    def influence_system(self):
        # Influence the regulated units
        self.influence_regulated_units()

    def calculate_percentiles(self):
        lower_percentile=25
        upper_percentile=75
        k = list(self.success_rates.values())
        lower_limit = np.percentile(k, lower_percentile)
        upper_limit = np.percentile(k, upper_percentile)
        return lower_limit, upper_limit
    
    def find_top(self, other_id):
        interactions = self.model.agent_interactions[other_id]
        most_interacted_agent = max(interactions, key=interactions.get)
        self.past_selection[most_interacted_agent] = self.allnoiseselection.get(most_interacted_agent,0)

    def find_past_selection(self):
        # Adjust trust levels based on historical performance and community sources
        for agent in self.model.schedule.agents:
            agent_id = agent.unique_id
            interactions = self.model.agent_interactions[agent_id]
            total_interactions = sum(interactions.values())

            self.learn_from_history(agent)


            if total_interactions == 0:
                continue  # Skip agents with no interactions

            # Calculate the success rate of agents this agent has interacted with
            self.success_rates = {k: v for k, v in interactions.items() if v > 0}

            # Calculate the upper and lower percentiles
            lower_limit, upper_limit = self.calculate_percentiles()
            # print(self.success_rates)
            # Adjust trust based on community sources and success rates
            for other_agent_id, success_rate in self.success_rates.items():
                if success_rate > upper_limit:  # High success rate
                    self.past_selection[other_agent_id] = self.allnoiseselection.get(other_agent_id,0)
                    self.find_top(other_agent_id)
    


    def get_past_selections(self, agent, n=10):
        #  the past n selections of the agent from historical data
        agent_id = agent.unique_id
        past_selections = []
        for history in reversed(self.historical_data):
            if agent_id in history['noise_selection']:
                past_selections.append(history['noise_selection'][agent_id])
            if len(past_selections) >= n:
                break
        return past_selections



    def learn_from_history(self, agent):
        if len(self.historical_data) < 5:
            return  
        
        # Find the past selection of agent
        past_selections = self.get_past_selections(agent, 10)
        print(past_selections)



    def step(self):
        self.update_historical_data()
        #self.learn_from_history()
        self.influence_system()
