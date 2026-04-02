import torch
from torch import nn
from torch.distributions import Categorical
from policies.utils import waste_here, get_accessible_neighbors
import src.workspace as ws

class RLPolicies(nn.Module):
    def __init__(self, model, available_actions, nn_path, **kwargs):
        super(RLPolicies, self).__init__()

        self.model = model
        self.available_actions = available_actions
        self.is_first_step = True
        self.n_possible_messages = sum([1 for action in self.available_actions if action['type'] == 'send_message'])
        self.phase_1 = kwargs.get("phase_1", False)

        # Grid dimensions for the CNN
        self.grid_w = model.grid.width
        self.grid_h = model.grid.height

        # Sizing boundaries
        self.spatial_flat_size = 8 * self.grid_w * self.grid_h
        self.inv_size = 3 # count of each waste type in inventory
        self.msg_size = 5 * self.n_possible_messages
        
        output_dim = len(self.available_actions)
        self.hidden_dim = kwargs.get("hidden_dim", 64)

        # input_dim = 5* self.n_possible_messages + 4 + 9*4 # spatial features + message features
        # # 1. Feature Extractor
        # self.embedding = nn.Sequential(
        #     nn.Linear(input_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU()
        # )

        # 1. Spatial Vision (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((3, 3)), # Compresses any grid size down to 3x3 for stability
            nn.Flatten()
        )

        cnn_output_dim = 32 * 3 * 3 # 288
        
        combined_dim = cnn_output_dim + self.inv_size + self.msg_size
        self.feature_mlp = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU()
        )

        # 2. LSTM Memory Cell (Processes exactly one step at a time)
        self.lstm = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)

        # 3. Actor Head (Outputs raw logits, NOT probabilities)
        self.actor_head = nn.Linear(self.hidden_dim, output_dim)

        self.coords_garbage_collector = (-1, -1) # Will be updated when discovered

        if nn_path:
            try:
                # map_location='cpu' ensures it loads safely even if trained on a CUDA GPU
                if torch.cuda.is_available():
                    self.load_state_dict(torch.load(nn_path, map_location=torch.device('cuda')))
                else:                    
                    self.load_state_dict(torch.load(nn_path, map_location=torch.device('cpu')))
                print(f"Successfully loaded model weights from: {nn_path} on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

            except Exception as e:
                print(f"Warning: Failed to load model weights from {nn_path}. Initializing from scratch. Error: {e}")
                print("Try loading on CPU...")
                self.load_state_dict(torch.load(nn_path, map_location=torch.device('cpu')))    
            
    def forward(self, x, hx, cx):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 1. Unpack the single tensor
        spatial_flat = x[:, :self.spatial_flat_size]
        other_features = x[:, self.spatial_flat_size:]
        
        # 2. Reshape spatial back into 2D Image: (Batch, Channels, W, H)
        spatial_img = spatial_flat.view(-1, 8, self.grid_w, self.grid_h)
        
        # 3. Run CNN
        cnn_features = self.cnn(spatial_img)
        
        # 4. Recombine and run MLP
        combined_features = torch.cat([cnn_features, other_features], dim=1)
        emb = self.feature_mlp(combined_features)
        
        # 5. LSTM & Actor
        hx, cx = self.lstm(emb, (hx, cx))
        logits = self.actor_head(hx)
        
        return logits, hx, cx
    """
    def logits_to_action(self, logits):
        action = torch.multinomial(logits, num_samples=1)
        chosen_action = action.item()

        return self.available_actions[chosen_action]
    """
    def logits_to_action(self, logits, dynamic_mask=None):
        # Categorical handles raw logits perfectly for PPO
        penalty = (1.0 - dynamic_mask) * -1e9
        logits = logits + penalty
        if self.phase_1:
            mask = torch.zeros_like(logits)
            for i, action in enumerate(self.available_actions):
                # If the action is a radio action, apply a massive penalty
                if action['type'] in ['send_message', 'read_message']:
                    mask[0, i] = -1e9 
            
            # Add the mask to the raw logits (banning the comm actions)
            logits = logits + mask

        dist = Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        return self.available_actions[action_idx.item()], action_idx.item(), log_prob.item()
    
    def process_messages(self, agent, messages):
        """Process received messages and update knowledge."""
        agent.knowledge['messages'] = {}
        for msg in messages:
            agent.knowledge['messages'][msg['content'][0]] = msg['content'][1] # si 2 même messages sont envoyés, le dernier écrase le précédent
        agent.n_unread_messages -= len(messages)
        
    
    def _init_knowledge(self, agent):
        agent.knowledge["green_agents_ids"] = [
            a.unique_id
            for a in self.model.robots
            if a.robot_type == "green" and a.unique_id != agent.unique_id
        ]
        agent.knowledge["yellow_agents_ids"] = [
            a.unique_id
            for a in self.model.robots
            if a.robot_type == "yellow" and a.unique_id != agent.unique_id
        ]
        agent.knowledge["red_agents_ids"] = [
            a.unique_id
            for a in self.model.robots
            if a.robot_type == "red" and a.unique_id != agent.unique_id
        ]

        device = next(self.parameters()).device

        agent.knowledge['hx'] = torch.zeros(1, self.hidden_dim)
        agent.knowledge['cx'] = torch.zeros(1, self.hidden_dim)

    # def _knowledge_to_input(self, agent):
    #     # This function can be used to convert the agent's knowledge into a tensor input for the policy network.
    #     adjacent_cells = agent.knowledge.get("adjacent_cells", {})
    #     message_vect = torch.zeros(5*self.n_possible_messages) # coord x sender, coord y sender, n_green_items sender, n_yellow_items sender, n_red_items sender
    #     spatial_vect = torch.zeros((4+9*4)) # coord x, coord y, n_green_items, n_yellow_items, 9 features per adjacent cells (4)

    #     # Spatial features _______________________________________________________________
    #     spatial_vect[:2] = torch.tensor(agent.pos)
    #     spatial_vect[2] = agent.inventory.count("green")
    #     spatial_vect[3] = agent.inventory.count("yellow")

    #     percepts = agent.knowledge.get('adjacent_cells', {})
    #     directions = self.model.direction_names

        

    #     if percepts != {}:
    #         for direction, cell in zip(percepts.keys(), percepts.values()):
    #             dir_idx = directions.index(direction)
    #             spatial_vect[4+dir_idx*9] = cell['waste'].count("green")
    #             spatial_vect[4+dir_idx*9+1] = cell['waste'].count("yellow")
    #             spatial_vect[4+dir_idx*9+2] = cell['waste'].count("red")
    #             spatial_vect[4+dir_idx*9+3] = cell['robots'].count("green")
    #             spatial_vect[4+dir_idx*9+4] = cell['robots'].count("yellow")
    #             spatial_vect[4+dir_idx*9+5] = cell['robots'].count("red")
    #             spatial_vect[4+dir_idx*9+6] = 1 if cell['zone'] == 'z1' else 0
    #             spatial_vect[4+dir_idx*9+7] = 1 if cell['zone'] == 'z2' else 0
    #             spatial_vect[4+dir_idx*9+8] = 1 if cell['zone'] == 'z3' else 0

    #     # Message features _______________________________________________________________
    #     if percepts.get('messages', {}) != {}:
    #         for i in range(self.n_possible_messages):
    #             message_vect[i*5:i*5+5] = agent.knowledge['messages'].get(i, torch.zeros(5))
        
    #     device = next(self.parameters()).device

    #     # Mask Generation ________________________________________________________________
        
    #     mask = torch.zeros(len(self.available_actions))
    #     x, y = agent.pos
    #     max_x, max_y = self.model.grid.width - 1, self.model.grid.height - 1
        
    #     for i, action in enumerate(self.available_actions):
    #         a_type = action['type']
            
    #         if a_type == 'move':
    #             dx, dy = action['direction']
    #             if 0 <= x + dx <= max_x and 0 <= y + dy <= max_y:
    #                 mask[i] = 1.0
                    
    #         elif a_type == 'pick_up':
    #             cell_contents = self.model.grid.get_cell_list_contents([agent.pos])
    #             if any(hasattr(obj, 'waste_type') for obj in cell_contents):
    #                 mask[i] = 1.0
                    
    #         elif a_type == 'put_down':
    #             if len(agent.inventory) > 0:
    #                 mask[i] = 1.0
                    
    #         elif a_type == 'transform':
    #             if agent.robot_type == 'green' and agent.inventory.count('green') >= 2:
    #                 mask[i] = 1.0
    #             elif agent.robot_type == 'yellow' and (agent.inventory.count('green') >= 2 or agent.inventory.count('yellow') >= 2):
    #                 mask[i] = 1.0
                    
    #         elif a_type == 'dispose':
    #             # Safely check if agent is on the collector zone
    #             if hasattr(self.model, 'waste_disposal_zone') and agent.pos == self.model.waste_disposal_zone.pos and len(agent.inventory) > 0:
    #                 mask[i] = 1.0          
    #         else:
    #             # wait, send_message, read_message are always physically legal
    #             mask[i] = 1.0

    #     return torch.cat([spatial_vect, message_vect]).to(device), mask.to(device)

    def _knowledge_to_input(self, agent):
        # 1. Initialize empty global map (8 channels)
        spatial_img = torch.zeros((8, self.grid_w, self.grid_h))
        
        x, y = agent.pos
        spatial_img[0, x, y] = 1.0 # Channel 0: Self Pos
        
        # Helper function to paint a cell onto the image channels
        def paint_cell(cx, cy, cell_info, is_self=False):
            if not (0 <= cx < self.grid_w and 0 <= cy < self.grid_h): return
            
            # Channels 1, 2, 3: Waste
            spatial_img[1, cx, cy] = cell_info.get('waste', []).count("green")
            spatial_img[2, cx, cy] = cell_info.get('waste', []).count("yellow")
            spatial_img[3, cx, cy] = cell_info.get('waste', []).count("red")
            
            # Channels 4, 5, 6: Other Agents (Subtract 1 if evaluating self)
            g_count = cell_info.get('robots', []).count("green")
            y_count = cell_info.get('robots', []).count("yellow")
            r_count = cell_info.get('robots', []).count("red")
            
            if is_self:
                if agent.robot_type == "green": g_count = max(0, g_count - 1)
                if agent.robot_type == "yellow": y_count = max(0, y_count - 1)
                if agent.robot_type == "red": r_count = max(0, r_count - 1)
                
            spatial_img[4, cx, cy] = g_count
            spatial_img[5, cx, cy] = y_count
            spatial_img[6, cx, cy] = r_count
            
            # Channel 7: Static Map (Zones & Discovered Collector)
            z = cell_info.get('zone')
            if z == 'z1': spatial_img[7, cx, cy] = 0.33
            elif z == 'z2': spatial_img[7, cx, cy] = 0.66
            elif z == 'z3': spatial_img[7, cx, cy] = 1.0
            
            if cell_info.get('disposal_zone', False):
                spatial_img[7, cx, cy] = 2.0 # Discovery Beacon!
                self.coords_garbage_collector = (cx, cy)

            if self.coords_garbage_collector != (-1, -1):
                ccx, ccy = self.coords_garbage_collector
                spatial_img[7, ccx, ccy] = 2.0 # Keep the beacon alive on the map once discovered


        # Paint Current Cell (Extracting directly to ensure accuracy)
        current_cell_contents = self.model.grid.get_cell_list_contents([agent.pos])
        current_cell_info = {
            'waste': [obj.waste_type for obj in current_cell_contents if hasattr(obj, 'waste_type')],
            'robots': [obj.robot_type for obj in current_cell_contents if hasattr(obj, 'robot_type')],
            'zone': agent.knowledge.get('zone', self.model._get_zone(x)),
            'disposal_zone': any(obj == getattr(self.model, 'waste_disposal_zone', None) for obj in current_cell_contents)
        }
        paint_cell(x, y, current_cell_info, is_self=True)

        # Paint Adjacent Cells (From Percepts)
        percepts = agent.knowledge.get('adjacent_cells', {})
        for direction, cell_info in percepts.items():
            adj_x, adj_y = cell_info['position']
            paint_cell(adj_x, adj_y, cell_info, is_self=False)

        # 2. Extract Inventories
        inv_vect = torch.tensor([
            agent.inventory.count("green"),
            agent.inventory.count("yellow"),
            agent.inventory.count("red")
        ], dtype=torch.float32)

        # 3. Extract Messages
        message_vect = torch.zeros(5 * self.n_possible_messages)
        if agent.knowledge.get('messages', {}) != {}:
            for i in range(self.n_possible_messages):
                message_vect[i*5:i*5+5] = agent.knowledge['messages'].get(i, torch.zeros(5))

        # Flatten and combine
        device = next(self.parameters()).device
        combined_input = torch.cat([spatial_img.flatten(), inv_vect, message_vect]).to(device)

        # 4. Action Mask
        mask = torch.zeros(len(self.available_actions))
        max_x, max_y = self.grid_w - 1, self.grid_h - 1
        
        for i, action in enumerate(self.available_actions):
            a_type = action['type']
            
            if a_type == 'move':
                dx, dy = action['direction']
                if 0 <= x + dx <= max_x and 0 <= y + dy <= max_y: mask[i] = 1.0
            elif a_type == 'pick_up':
                if any(hasattr(obj, 'waste_type') for obj in current_cell_contents): mask[i] = 1.0
            elif a_type == 'put_down':
                if len(agent.inventory) > 0: mask[i] = 1.0
            elif a_type == 'transform':
                if agent.robot_type == 'green' and agent.inventory.count('green') >= 2: mask[i] = 1.0
                elif agent.robot_type == 'yellow' and (agent.inventory.count('green') >= 2 or agent.inventory.count('yellow') >= 2): mask[i] = 1.0
            elif a_type == 'dispose':
                if current_cell_info['disposal_zone'] and len(agent.inventory) > 0: mask[i] = 1.0
            else:
                mask[i] = 1.0

        return combined_input, mask.to(device)

    
    def deliberate(self, agent):
        if 'hx' not in agent.knowledge:
            self._init_knowledge(agent)
            self.is_first_step = False


        x_input, dynamic_mask = self._knowledge_to_input(agent)
        device = next(self.parameters()).device
        x_input = x_input.to(device)
        agent.knowledge['hx'] = agent.knowledge['hx'].to(device)
        agent.knowledge['cx'] = agent.knowledge['cx'].to(device)

        # SAVE THE TENSORS FOR THE TRAINING SCRIPT BEFORE UPDATING THEM
        agent.knowledge['last_x_input'] = x_input.detach()
        agent.knowledge['last_hx'] = agent.knowledge['hx'].detach()
        agent.knowledge['last_cx'] = agent.knowledge['cx'].detach()

        logits, new_hx, new_cx = self.forward(
            x_input, 
            agent.knowledge['hx'], 
            agent.knowledge['cx']
        )
        # CRITICAL FIX: Overwrite the knowledge with DETACHED hidden states
        # This severs the memory leak between steps.
        agent.knowledge['hx'] = new_hx.detach()
        agent.knowledge['cx'] = new_cx.detach()

        dynamic_mask = torch.zeros_like(dynamic_mask).to(device) # no masking in case of error.
        chosen_action, action_idx, log_prob = self.logits_to_action(logits, dynamic_mask)

        agent.knowledge['last_action_idx'] = action_idx
        agent.knowledge['last_action_log_prob'] = log_prob

        if chosen_action['type'] == 'send_message':
            recipients = chosen_action['recipient_ids']
            chosen_action['recipient_ids'] = []
            
            for i in recipients:
                if i == 'g':
                    chosen_action['recipient_ids'].extend(agent.knowledge["green_agents_ids"])
                elif i == 'y':
                    chosen_action['recipient_ids'].extend(agent.knowledge["yellow_agents_ids"])
                elif i == 'r':
                    chosen_action['recipient_ids'].extend(agent.knowledge["red_agents_ids"])

            content_vector = torch.cat([
                torch.tensor(agent.pos),
                torch.tensor([agent.inventory.count("green"), agent.inventory.count("yellow"), agent.inventory.count("red")])
            ]).flatten()

            chosen_action['content'] = (chosen_action['content'], content_vector)

        # if chosen_action['type'] == 'move':
        #     neighbors = get_accessible_neighbors(self.model, agent, agent.pos)
        #     possible_directions = [d for _, d in neighbors]
        #     if chosen_action['direction'] not in possible_directions:
        #         chosen_action = {'type': 'wait'}

        return chosen_action

class Policy:
    def __init__(self, model, available_actions, **kwargs):
        self.model = model
        self.available_actions = available_actions
        self.green_pol = RLPolicies(model, ws.green_yellow_rl_actions, kwargs.get("green_nn_path", None),  **kwargs)
        self.yellow_pol = RLPolicies(model, ws.green_yellow_rl_actions, kwargs.get("yellow_nn_path", None), **kwargs)
        self.red_pol = RLPolicies(model, ws.red_rl_actions, kwargs.get("red_nn_path", None), **kwargs)

    def process_messages(self, agent, messages):
        if agent.robot_type == "green":
            self.green_pol.process_messages(agent, messages)
        elif agent.robot_type == "yellow":
            self.yellow_pol.process_messages(agent, messages)
        elif agent.robot_type == "red":
            self.red_pol.process_messages(agent, messages)
        else:
            raise ValueError(f"Unknown robot type: {agent.robot_type}")

    def deliberate(self, agent):
        if agent.robot_type == "green":
            return self.green_pol.deliberate(agent)
        elif agent.robot_type == "yellow":
            return self.yellow_pol.deliberate(agent)
        elif agent.robot_type == "red":
            return self.red_pol.deliberate(agent)
        else:
            raise ValueError(f"Unknown robot type: {agent.robot_type}")
                
                