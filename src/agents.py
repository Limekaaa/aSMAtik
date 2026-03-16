# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import mesa
from mesa.datacollection import DataCollector
from mesa import Agent

class greenAgent(Agent):
    def __init__(self, model, policy_name:str, **kwargs):
        super().__init__(model)
        self.model = model
        self.knowledge = {}
        self.actions = ["move", "pick", "drop"]
        self.inventory = {}

        policy_import = __import__("policies." + policy_name, fromlist="Policy")

        policy = policy_import.Policy(self.model, self.actions, **kwargs)

        self.deliberate = policy.deliberate


    def do(self):
        pass

    def update(self):
        pass

    def step_agent (self):
        #self.update(self.knowledge, percepts)
        action = self.deliberate(self.knowledge)
        #percepts = self.model.do(self, action)
