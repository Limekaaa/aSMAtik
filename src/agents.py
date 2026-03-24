# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import random
from mesa import Agent
from policies.utils import waste_here


class RobotAgent(Agent):
    def __init__(self, model, policy_name: str, **kwargs):
        super().__init__(model)
        
        self.model = model
        self.knowledge = {}
        self.inventory = []
        self.robot_type = None
        self.max_inventory = 0
        self.disposed_waste_count = 0
        self.last_action = {}
        self.n_unread_messages = 0
        self.messages = []
        
        # Load policy with available actions
        available_actions = self.model.available_actions
        policy_module = __import__("policies." + policy_name, fromlist="Policy")
        policy = policy_module.Policy(self.model, available_actions, **kwargs)
        self.deliberate = policy.deliberate

    def update(self, percepts: dict):
        self.knowledge.update(percepts)

    def can_access_zone(self, zone):
        pass

    def can_pick_up_type(self, waste_type):
        pass

    def step(self):
        """Mesa step method."""
        
        action = self.deliberate(self)
        self.last_action = action
        percepts = self.model.do(self, action)
        self.knowledge.update(percepts)


class GreenRobot(RobotAgent):
    def __init__(self, model, policy_name: str, **kwargs):
        super().__init__(model, policy_name, **kwargs)
        self.robot_type = "green"

    def can_access_zone(self, zone):
        if zone in ["z1"]:
            return True
        return False
    
    def can_pick_up_type(self, waste_type):
        if waste_type in ["green"]:
            return True
        return False


class YellowRobot(RobotAgent):
    def __init__(self, model, policy_name: str, **kwargs):
        super().__init__(model, policy_name, **kwargs)
        self.robot_type = "yellow"

    def can_access_zone(self, zone):
        if zone in ["z1", "z2"]:
            return True
        return False
    
    def can_pick_up_type(self, waste_type):
        if waste_type in ["yellow", "green"]:
            return True
        return False

class RedRobot(RobotAgent):
    def __init__(self, model, policy_name: str, **kwargs):
        super().__init__(model, policy_name, **kwargs)
        self.robot_type = "red"

    def can_access_zone(self, zone):
        if zone in ["z1", "z2", "z3"]:
            return True
        return False
    
    def can_pick_up_type(self, waste_type):
        return True