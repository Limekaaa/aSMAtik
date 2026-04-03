import torch
import math
from src.model import RobotMission # Assuming your model.py is in src.model

class RLEnvironmentWrapper:
    """
    RL Wrapper for the RobotMission Mesa model.
    Calculates R_global, PBRS, Handoff bonuses, and Communication penalties.
    """
    def __init__(self, **kwargs):
        # Hyperparameters for the Reward Function
        self.gamma = 1.0
        self.lambda_deposit = 400.0
        self.lambda_time = 0.0
        self.omega_dist = 1.0
        
        # Critical Math Constraint: V(Yellow) > 2 * V(Green), V(Red) > 2 * V(Yellow)
        self.omega_class = {
            'green': 10.0,
            'yellow': 50, #100.0,  # 25 > 2 * 10
            'red': 130 #300.0      # 60 > 2 * 25
        }
        
        self.beta_handoff = 5.0
        self.lambda_comm = 0.001
        
        # Initialize underlying model
        self.env_kwargs = kwargs
        self.mesa_model = RobotMission(**self.env_kwargs)
        
        # State tracking for temporal rewards (PBRS and Global Delta)
        self.previous_potential = 0.0
        self.previous_disposed_count = 0
        self.drop_tracker = {} # Tracks { (x, y): agent_id } for handoff bonuses

    def reset(self):
        """Resets the simulation and the reward trackers."""
        self.mesa_model = RobotMission(**self.env_kwargs)
        self.drop_tracker = {}
        self.previous_disposed_count = self.mesa_model._count_disposed_waste()
        self.previous_potential = self._calculate_global_potential()
        
        return self._get_all_observations()

    def step(self):
        """
        Advances the simulation by one step and calculates the RL rewards.
        """
        # 1. Step the underlying Mesa model (agents act sequentially here)
        
        old_inventories = {a.unique_id: a.inventory.copy() for a in self.mesa_model.robots}

        self.mesa_model.step()
        
        # 2. Extract new observations for the next step
        observations = self._get_all_observations()
        
        # 3. Calculate Rewards
        raw_rewards = self._calculate_rewards(old_inventories)
        rewards = {aid: (reward / 1) for aid, reward in raw_rewards.items()}
        
        # 4. Check terminal state
        dones = {agent.unique_id: self.mesa_model.is_done() for agent in self.mesa_model.robots}
        dones['__all__'] = self.mesa_model.is_done()
        
        return observations, rewards, dones, {}

    def _get_all_observations(self):
        """Extracts the policy inputs for all agents."""
        # Note: In your training loop, you will pull the 'hx' and 'cx' from 
        # the agents directly, so this just returns a dummy dict to match gym API.
        # The actual forward pass relies on the agent.knowledge['hx'] anyway.
        return {agent.unique_id: agent.knowledge for agent in self.mesa_model.robots}

    def _calculate_rewards(self, old_inventories=None):
        """
        Computes the composite reward: R_global + F_shaping + R_handoff + R_comm
        """
        rewards = {agent.unique_id: 0.0 for agent in self.mesa_model.robots}
        
        # ---------------------------------------------------------
        # A. Global Objective (Shared)
        # ---------------------------------------------------------
        current_disposed = self.mesa_model._count_disposed_waste()
        delta_deposit = current_disposed - self.previous_disposed_count
        self.previous_disposed_count = current_disposed
        
        r_global = (self.lambda_deposit * delta_deposit) - self.lambda_time
        
        # Add shared global reward to all agents
        for agent_id in rewards:
            rewards[agent_id] += r_global

        # ---------------------------------------------------------
        # B. Potential-Based Reward Shaping (PBRS)
        # ---------------------------------------------------------
        current_potential = self._calculate_global_potential()
        f_shaping = (self.gamma * current_potential) - self.previous_potential
        self.previous_potential = current_potential
        
        # PBRS is distributed equally to all agents to encourage global cooperation
        for agent_id in rewards:
            rewards[agent_id] += f_shaping

        # ---------------------------------------------------------
        # C. Local Heuristics (Handoffs & Communication)
        # ---------------------------------------------------------
        for agent in self.mesa_model.robots:
            action = getattr(agent, 'last_action', {})
            if not action:
                continue
                
            a_type = action.get('type')
            
            # 1. Communication Penalty
            if a_type == 'send_message':
                rewards[agent.unique_id] -= self.lambda_comm
            

            # # DROP 2 AND 3 IF REWARD HACKING BECOMES AN ISSUE
            # # 2. Track Drops for Handoffs 
            # elif a_type == 'put_down':
            #     # Record that this agent dropped something at this exact coordinate
            #     self.drop_tracker[agent.pos] = agent.unique_id
                
            # # 3. Reward Handoffs on Pickups
            # elif a_type == 'pick_up':
            #     if agent.pos in self.drop_tracker:
            #         dropper_id = self.drop_tracker[agent.pos]
            #         # If picked up by a DIFFERENT agent, grant the bonus to the dropper
            #         if dropper_id != agent.unique_id:
            #             if dropper_id in rewards:
            #                 rewards[dropper_id] += self.beta_handoff
            #         # Clear the tracker for this tile since the item was picked up
            #         del self.drop_tracker[agent.pos]

            # 4. Reward Transformations (Crafting)
            elif a_type == 'transform':
                old_inv = old_inventories[agent.unique_id]
                new_inv = agent.inventory
                
                # Check Green -> Yellow Craft
                if agent.robot_type == 'green':
                    if old_inv.count('green') > new_inv.count('green') and new_inv.count('yellow') > old_inv.count('yellow'):
                        rewards[agent.unique_id] += 10.0
                        
                # Check Yellow -> Red Craft
                elif agent.robot_type == 'yellow':
                    if (old_inv.count('green') + old_inv.count('yellow')) > (new_inv.count('green') + new_inv.count('yellow')) and new_inv.count('red') > old_inv.count('red'):
                        rewards[agent.unique_id] += 10.0

        return rewards

    def _calculate_global_potential(self):
        """
        Calculates Phi(s) for all active garbage.
        Ensures potential is strictly POSITIVE to prevent free rewards from discounting.
        """
        phi = 0.0
        collector_pos = self.mesa_model.waste_disposal_zone.pos
        
        if not collector_pos:
            return 0.0

        # Maximum possible Manhattan distance on a 20x10 grid is 30
        max_dist = self.mesa_model.width + self.mesa_model.height

        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        def y_axis_dist(p1, p2):
            return abs(p1[1] - p2[1])
        
        used_dist_func = manhattan_dist # Change to y_axis_dist if you want to prioritize vertical proximity

        # 1. Evaluate waste currently on the grid
        for waste in self.mesa_model.waste_pieces:
            c_g = self.omega_class.get(waste.waste_type, 0.0)
            #d_g = manhattan_dist(waste.pos, collector_pos)
            d_g = used_dist_func(waste.pos, collector_pos)
            # ADD the inverted distance so closeness = higher positive score
            phi += (c_g + self.omega_dist * (max_dist - d_g))

        # 2. Evaluate waste currently held by agents
        for agent in self.mesa_model.robots:
            for item_type in agent.inventory:
                c_g = self.omega_class.get(item_type, 0.0)
                d_g = used_dist_func(agent.pos, collector_pos)
                phi += (c_g + self.omega_dist * (max_dist - d_g))

        return phi