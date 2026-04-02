import torch
from torch import nn

class GlobalCritic(nn.Module):
    """
    Centralized Critic for MAPPO. 
    Evaluates the Global State to compute the baseline V(s).
    """
    def __init__(self, grid_width=20, grid_height=10, hidden_dim=128):
        super(GlobalCritic, self).__init__()
        
        self.width = grid_width
        self.height = grid_height
        
        # --- 1. The Spatial Feature Extractor (CNN) ---
        # Input shape: (Batch, Channels=6, Height, Width)
        self.cnn = nn.Sequential(
            # First convolution: keeps dimensions same, extracts local patterns
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Pooling: reduces spatial dimensions by half (e.g., 10x20 -> 5x10)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second convolution: deeper feature extraction
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the flattened dimension dynamically based on grid size
        cnn_output_height = self.height // 2
        cnn_output_width = self.width // 2
        self.cnn_flat_dim = 32 * cnn_output_height * cnn_output_width
        
        # --- 2. The Coordinates Extractor (MLP) ---
        # Processes the (x, y) coordinates of the garbage collector
        self.coords_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        
        # --- 3. The Fusion & Value Head ---
        # Combines the flattened CNN map with the embedded coordinates
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.cnn_flat_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Outputs a single scalar: V(s)
        )

    def forward(self, grid_state, collector_coords):
        """
        Forward pass for the Critic.
        Args:
            grid_state: Tensor of shape (Batch, 6, Height, Width)
            collector_coords: Tensor of shape (Batch, 2)
        """
        # Ensure batch dimensions
        if grid_state.dim() == 3:
            grid_state = grid_state.unsqueeze(0)
        if collector_coords.dim() == 1:
            collector_coords = collector_coords.unsqueeze(0)
            
        cnn_features = self.cnn(grid_state)
        coord_features = self.coords_mlp(collector_coords)
        
        # Merge spatial map context with exact collector location
        fused_features = torch.cat([cnn_features, coord_features], dim=1)
        
        state_value = self.fusion_mlp(fused_features)
        
        return state_value

    def extract_global_state(self, model):
        """
        Parses the Mesa model into PyTorch tensors for the Critic.
        PyTorch expects images in format (Batch, Channels, Height, Width).
        Therefore, Width maps to x, and Height maps to y.
        """
        # Channels: 0: G-Robots, 1: Y-Robots, 2: R-Robots, 3: G-Waste, 4: Y-Waste, 5: R-Waste
        grid_state = torch.zeros((1, 6, self.height, self.width))
        
        # 1. Populate Robot Channels
        for robot in model.robots:
            x, y = robot.pos
            if robot.robot_type == "green":
                grid_state[0, 0, y, x] += 1
            elif robot.robot_type == "yellow":
                grid_state[0, 1, y, x] += 1
            elif robot.robot_type == "red":
                grid_state[0, 2, y, x] += 1
                
        # 2. Populate Waste Channels
        for waste in model.waste_pieces:
            x, y = waste.pos
            if waste.waste_type == "green":
                grid_state[0, 3, y, x] += 1
            elif waste.waste_type == "yellow":
                grid_state[0, 4, y, x] += 1
            elif waste.waste_type == "red":
                grid_state[0, 5, y, x] += 1
                
        # 3. Extract and Normalize Collector Coordinates
        # Normalizing coordinates between [0, 1] prevents massive gradients in the MLP
        if model.waste_disposal_zone is not None:
            c_x, c_y = model.waste_disposal_zone.pos
            normalized_x = c_x / max(1, self.width - 1)
            normalized_y = c_y / max(1, self.height - 1)
            collector_coords = torch.tensor([normalized_x, normalized_y], dtype=torch.float32)
        else:
            collector_coords = torch.zeros(2, dtype=torch.float32)
            
        return grid_state, collector_coords

    def evaluate(self, model):
        """
        Helper method to instantly evaluate the environment's current value.
        """
        grid_state, collector_coords = self.extract_global_state(model)
        value = self.forward(grid_state, collector_coords)
        return value.item()