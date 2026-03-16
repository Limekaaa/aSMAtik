# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import mesa
from mesa.datacollection import DataCollector
from mesa import Agent


class RobotAgent(Agent):
    def __init__(self, model, policy_name:str, **kwargs):
        super().__init__(model)
        self.model = model
        self.knowledge = {}
        self.actions = ["left", "right", "up", "down", "pick", "drop", "transform"]
        self.inventory = {}

        policy_import = __import__("policies." + policy_name, fromlist="Policy")

        policy = policy_import.Policy(self.model, self.actions, **kwargs)

        self.deliberate = policy.deliberate

    def update(self, percepts:dict):
        self.knowledge.update(percepts)

    def can_access_zone(self, zone):
        pass

    
    def can_pick_up_type(self, waste_type):
        pass

    def step_agent (self):
        action = self.deliberate(self.knowledge)
        percepts = self.model.do(self, action)
        self.knowledge.update(percepts)


class GreenRobot(RobotAgent):
    def __init__(self, model, policy_name:str, **kwargs):
        super().__init__(model, policy_name, **kwargs)

    def can_access_zone(self, zone):
        if zone in ["z1"]:
            return True
        return False
    
    def can_pick_up_type(self, waste_type):
        if waste_type in ["green"]:
            return True
        return False


class YellowRobot(RobotAgent):
    def __init__(self, model, policy_name:str, **kwargs):
        super().__init__(model, policy_name, **kwargs)

    def can_access_zone(self, zone):
        if zone in ["z1", "z2"]:
            return True
        return False
    
    def can_pick_up_type(self, waste_type):
        if waste_type in ["yellow", "green"]:
            return True
        return False

class RedRobot(RobotAgent):
    def __init__(self, model, policy_name:str, **kwargs):
        super().__init__(model, policy_name, **kwargs)

    def can_access_zone(self, zone):
        if zone in ["z1", "z2", "z3"]:
            return True
        return False
    
    def can_pick_up_type(self, waste_type):
        return True