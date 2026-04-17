import torch
from src.model_RL_train import RobotMission 
from policies.RL_baseline_RL_train import RLPolicies
from src.critic import GlobalCritic
import src.workspace as ws

class RLEnvironmentWrapper:
    def __init__(self, **kwargs):
        self.gamma = 1.0
        self.lambda_deposit = 400.0
        self.lambda_time = 0.0
        self.omega_dist = 1.0
        self.omega_class = {'green': 10.0, 'yellow': 50.0, 'red': 130.0}
        self.beta_handoff = 5.0
        self.lambda_comm = 0.001
        
        self.env_kwargs = kwargs
        self.mesa_model = RobotMission(**self.env_kwargs)
        
        # We initialize a "dummy" policy purely to use its _knowledge_to_input function
        self.dummy_policy_gy = RLPolicies(self.mesa_model, ws.green_yellow_rl_actions, None, hidden_dim=ws.kwargs.get('hidden_dim', 64))
        self.dummy_policy_r = RLPolicies(self.mesa_model, ws.red_rl_actions, None, hidden_dim=ws.kwargs.get('hidden_dim', 64))
        
        # We initialize a "dummy" critic purely to use its extract_global_state function
        self.dummy_critic = GlobalCritic(self.mesa_model, grid_width=self.mesa_model.width, grid_height=self.mesa_model.height, hidden_dim=ws.kwargs.get('hidden_dim', 64))
        
        self.previous_potential = 0.0
        self.previous_disposed_count = 0

    def reset(self):
        self.mesa_model = RobotMission(**self.env_kwargs)
        self.dummy_policy_gy.model = self.mesa_model
        self.dummy_policy_r.model = self.mesa_model
        
        for agent in self.mesa_model.robots:
            policy = self.dummy_policy_gy if agent.robot_type in ['green', 'yellow'] else self.dummy_policy_r
            policy._init_knowledge(agent)
            
        self.previous_disposed_count = self.mesa_model._count_disposed_waste()
        self.previous_potential = self._calculate_global_potential()
        
        return self._get_all_observations()

    def step(self, actions_dict):
        old_inventories = {a.unique_id: a.inventory.copy() for a in self.mesa_model.robots}
        
        # --- Pre-process Communications ---
        for aid, action in actions_dict.items():
            if action['type'] == 'send_message':
                recipients_str = action['recipient_ids']
                actual_recipients = []
                
                # Translate string targets into physical integer Agent IDs
                for char in recipients_str:
                    if char == 'g':
                        actual_recipients.extend([a.unique_id for a in self.mesa_model.robots if a.robot_type == 'green' and a.unique_id != aid])
                    elif char == 'y':
                        actual_recipients.extend([a.unique_id for a in self.mesa_model.robots if a.robot_type == 'yellow' and a.unique_id != aid])
                    elif char == 'r':
                        actual_recipients.extend([a.unique_id for a in self.mesa_model.robots if a.robot_type == 'red' and a.unique_id != aid])
                
                # Build the content vector
                agent = next(a for a in self.mesa_model.robots if a.unique_id == aid)
                content_vector = torch.cat([
                    torch.tensor(agent.pos, dtype=torch.float32),
                    torch.tensor([agent.inventory.count("green"), agent.inventory.count("yellow"), agent.inventory.count("red")], dtype=torch.float32)
                ]).flatten()
                
                # Inject translated data
                action['recipient_ids'] = actual_recipients
                action['content'] = (action['content'], content_vector)

        # Push external actions into the physics engine
        self.mesa_model.step(actions_dict)
        
        # --- FIX: Calculate raw rewards and apply 1/100 Normalization! ---
        raw_rewards = self._calculate_rewards(old_inventories)
        rewards = {aid: (reward / 100.0) for aid, reward in raw_rewards.items()}
        
        dones = {agent.unique_id: self.mesa_model.is_done() for agent in self.mesa_model.robots}
        dones['__all__'] = self.mesa_model.is_done()
        
        info = {
            'waste_left': self.mesa_model._count_total_waste(),
            'disposed_waste': self.mesa_model._count_disposed_waste()
        }
        
        return self._get_all_observations(), rewards, dones, info

    def _get_all_observations(self):
        obs = {}
        for agent in self.mesa_model.robots:
            # Select the correct policy for the agent's color
            policy = self.dummy_policy_gy if agent.robot_type in ['green', 'yellow'] else self.dummy_policy_r
            
            if 'hx' not in agent.knowledge:
                policy._init_knowledge(agent)
            
            x_input, dynamic_mask = policy._knowledge_to_input(agent)
            
            obs[agent.unique_id] = {
                'x_input': x_input,
                'mask': dynamic_mask,
                'color': agent.robot_type
            }
            
        # --- Extract Global Critic States ---
        grid_state, collector_coords = self.dummy_critic.extract_global_state(self.mesa_model)
        
        # Remove the dummy batch dimension (1, ...) because train_mappo uses torch.stack()
        obs['global_grid'] = grid_state.squeeze(0) 
        obs['collector_coords'] = collector_coords.squeeze(0) if collector_coords.dim() > 1 else collector_coords
        
        return obs

    def _calculate_rewards(self, old_inventories=None):
        """
        Computes the composite reward: R_global + F_shaping + R_handoff + R_comm
        """
        rewards = {agent.unique_id: 0.0 for agent in self.mesa_model.robots}
        
        current_disposed = self.mesa_model._count_disposed_waste()
        delta_deposit = current_disposed - self.previous_disposed_count
        self.previous_disposed_count = current_disposed
        
        r_global = (self.lambda_deposit * delta_deposit) - self.lambda_time
        
        for agent_id in rewards:
            rewards[agent_id] += r_global

        current_potential = self._calculate_global_potential()
        f_shaping = (self.gamma * current_potential) - self.previous_potential
        self.previous_potential = current_potential
        
        for agent_id in rewards:
            rewards[agent_id] += f_shaping

        for agent in self.mesa_model.robots:
            action = getattr(agent, 'last_action', {})
            if not action: continue
                
            a_type = action.get('type')
            
            if a_type == 'send_message':
                rewards[agent.unique_id] -= self.lambda_comm
                
            elif a_type == 'transform':
                old_inv = old_inventories[agent.unique_id]
                new_inv = agent.inventory
                
                if agent.robot_type == 'green':
                    if old_inv.count('green') > new_inv.count('green') and new_inv.count('yellow') > old_inv.count('yellow'):
                        rewards[agent.unique_id] += 100.0
                        
                elif agent.robot_type == 'yellow':
                    # Reward for Green -> Yellow
                    if (old_inv.count('green') > new_inv.count('green')) and (new_inv.count('yellow') > old_inv.count('yellow')):
                        rewards[agent.unique_id] += 100.0
                    # Reward for Yellow -> Red
                    elif (old_inv.count('yellow') > new_inv.count('yellow')) and (new_inv.count('red') > old_inv.count('red')):
                        rewards[agent.unique_id] += 100.0

        return rewards

    def _calculate_global_potential(self):
        """
        Calculates Phi(s) for all active garbage.
        """
        phi = 0.0
        collector_pos = self.mesa_model.waste_disposal_zone.pos
        
        if not collector_pos: return 0.0

        max_dist = self.mesa_model.width + self.mesa_model.height

        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        used_dist_func = manhattan_dist 

        for waste in self.mesa_model.waste_pieces:
            c_g = self.omega_class.get(waste.waste_type, 0.0)
            d_g = used_dist_func(waste.pos, collector_pos)
            phi += (c_g + self.omega_dist * (max_dist - d_g))

        for agent in self.mesa_model.robots:
            for item_type in agent.inventory:
                c_g = self.omega_class.get(item_type, 0.0)
                d_g = used_dist_func(agent.pos, collector_pos)
                phi += (c_g + self.omega_dist * (max_dist - d_g))

        for agent in self.mesa_model.robots:
            target_weights = {}
            if agent.robot_type == 'green':
                target_weights = {'green': 1.0}
            elif agent.robot_type == 'yellow':
                target_weights = {'yellow': 1.0, 'green': 0.5}
            elif agent.robot_type == 'red':
                target_weights = {'red': 1.0, 'yellow': 0.6, 'green': 0.3}
                
            radar_max_score = 0.5 * max_dist
            
            for item in agent.inventory:
                weight = target_weights.get(item, 0.0)
                phi += weight * radar_max_score
            
            if len(agent.inventory) < getattr(agent, 'max_inventory', 2):
                valid_wastes = [w for w in self.mesa_model.waste_pieces if w.waste_type in target_weights]
                
                if valid_wastes:
                    best_seek_score = 0.0
                    for w in valid_wastes:
                        dist = used_dist_func(agent.pos, w.pos)
                        weight = target_weights[w.waste_type]
                        score = weight * 0.5 * (max_dist - dist)
                        if score > best_seek_score:
                            best_seek_score = score
                    phi += best_seek_score
                else:
                    phi += radar_max_score

        return phi