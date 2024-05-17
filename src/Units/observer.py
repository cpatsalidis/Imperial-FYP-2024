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
            'job_sizes': [agent.jobSize for agent in self.model.schedule.agents],
            'urgencies': [agent.jobUrgency for agent in self.model.schedule.agents],
            'trust_levels': {agent.unique_id: agent.trust.copy() for agent in self.model.schedule.agents}
        }
        self.historical_data.append(current_data)
        if len(self.historical_data) > 100:  # Limit historical data length
            self.historical_data.pop(0)

    def influence_system(self):
        # Influence the regulator
        self.influence_regulator()
        # Influence the regulated units
        self.influence_regulated_units()

    def calculate_percentiles(self):
        lower_percentile=25
        upper_percentile=75
        k = list(self.success_rates.values())
        lower_limit = np.percentile(k, lower_percentile)
        upper_limit = np.percentile(k, upper_percentile)
        return lower_limit, upper_limit

    def influence_regulated_units(self):
        # Adjust trust levels based on historical performance and community sources
        for agent in self.model.schedule.agents:
            agent_id = agent.unique_id
            interactions = self.model.agent_interactions[agent_id]
            total_interactions = sum(interactions.values())

            if total_interactions == 0:
                continue  # Skip agents with no interactions

            # Calculate the success rate of agents this agent has interacted with
            self.success_rates = {k: v for k, v in interactions.items() if v > 0}

            # Calculate the upper and lower percentiles
            lower_limit, upper_limit = self.calculate_percentiles()
            #print(self.success_rates)
            # Adjust trust based on community sources and success rates
            for other_agent_id, success_rate in self.success_rates.items():
                if success_rate > upper_limit:  # High success rate
                    agent.update_trust({other_agent_id: 1.2 })
                elif success_rate < lower_limit:  # Low success rate
                    agent.update_trust({other_agent_id: 0.8 })


    def learn_from_history(self):
        # Implement a simple learning algorithm to adjust influence_factor
        if len(self.historical_data) < 2:
            return  # Not enough data to learn from

        recent_data = self.historical_data[-1]
        previous_data = self.historical_data[-2]

        # Calculate changes in metrics
        avg_job_size_change = np.mean(recent_data['job_sizes']) - np.mean(previous_data['job_sizes'])
        avg_urgency_change = np.mean(recent_data['urgencies']) - np.mean(previous_data['urgencies'])

        # Adjust influence_factor based on observed changes
        if avg_job_size_change > 0 and avg_urgency_change > 0:
            self.influence_factor += self.learning_rate
        else:
            self.influence_factor -= self.learning_rate

        # Ensure influence_factor remains within a reasonable range
        self.influence_factor = max(0.1, min(2, self.influence_factor))

    def step(self):
        self.update_historical_data()
        #self.learn_from_history()
        self.influence_system()
