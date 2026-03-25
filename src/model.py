# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import random
import numpy as np
import mesa
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from src.mailbox import Mailbox
import src.workspace as ws


class RobotMission(Model):
    """
    The RobotMission model simulates robots collecting and transforming radioactive waste
    across three zones with varying radioactivity levels.

    Environment:
    - z1 (west): low radioactivity (0-0.33), contains initial green waste
    - z2 (middle): medium radioactivity (0.33-0.66)
    - z3 (east): high radioactivity (0.66-1.0), contains waste disposal zone
    
    Parameters:
    - width: grid width
    - height: grid height
    - num_green_robots: number of green robots
    - num_yellow_robots: number of yellow robots
    - num_red_robots: number of red robots
    - num_initial_waste: initial number of green waste pieces
    """
    
    def __init__(
        self,
        width=20,
        height=10,
        num_green_robots=1,
        num_yellow_robots=1,
        num_red_robots=1,
        num_initial_waste=4,
        seed=None
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.num_green_robots = num_green_robots
        self.num_yellow_robots = num_yellow_robots
        self.num_red_robots = num_red_robots
        self.num_initial_waste = num_initial_waste

        self.available_actions = [
            'move', 'pick_up', 'transform', 'put_down',
            'dispose', 'wait', 'read_message', 'send_message'
        ]

        self.mailbox = Mailbox()

        # Zone boundaries (x-coordinates)
        self.zone_boundaries = {
            'z1': (0, width // 3),
            'z2': (width // 3, 2 * width // 3),
            'z3': (2 * width // 3, width)
        }

        # Meeting checkpoints (borders between zones)
        self.z1_z2_border = (self.zone_boundaries['z1'][1] - 1, self.height // 2)
        self.z2_z3_border = (self.zone_boundaries['z2'][1] - 1, self.height // 2)

        # Grid
        self.grid = MultiGrid(width, height, torus=False)
        
        # Custom tracking lists
        self.robots = []
        self.waste_pieces = []
        self.radioactivity_agents = []
        self.waste_disposal_zone = None

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Green Waste": self._count_green_waste,
                "Yellow Waste": self._count_yellow_waste,
                "Red Waste": self._count_red_waste,
                "Total Waste": self._count_total_waste,
                "Disposed Waste": self._count_disposed_waste
            }
        )

        self._setup_environment()
        self._initialize_agents()

        self.running = True
        self.datacollector.collect(self)

    def _setup_environment(self):
        """Setup the grid with radioactivity agents and waste disposal zone."""

        from .objects import RadioactivityAgent, WasteDisposalZoneAgent, WasteAgent

        # Create radioactivity agents for each zone (one per cell)
        for x in range(self.width):
            for y in range(self.height):
                zone = self._get_zone(x)
                radioactivity_level = self._calculate_radioactivity(zone)

                radioactivity = RadioactivityAgent(model=self, zone=zone)
                radioactivity.radioactivity_level = radioactivity_level
                self.grid.place_agent(radioactivity, (x, y))
                self.radioactivity_agents.append(radioactivity)

        # Place waste disposal zone (random cell in z3)
        z3_start, z3_end = self.zone_boundaries['z3']
        disposal_x = random.randint(z3_start, z3_end - 1)
        disposal_y = random.randint(0, self.height - 1)

        self.waste_disposal_zone = WasteDisposalZoneAgent(model=self)
        self.grid.place_agent(self.waste_disposal_zone, (disposal_x, disposal_y))

        # Create initial green waste in z1
        z1_start, z1_end = self.zone_boundaries['z1']
        for _ in range(self.num_initial_waste):
            x = random.randint(z1_start, z1_end - 1)
            y = random.randint(0, self.height - 1)

            waste = WasteAgent(model=self, waste_type='green')
            self.grid.place_agent(waste, (x, y))
            self.waste_pieces.append(waste)

    def _initialize_agents(self):
        """Initialize robot agents."""
        from .agents import GreenRobot, YellowRobot, RedRobot

        policy_name = ws.POLICY
        kwargs = {"holding_threshold": 10}

        # Green robots
        for _ in range(self.num_green_robots):
            robot = GreenRobot(model=self, policy_name=policy_name, **kwargs)
            robot.max_inventory = ws.GREEN_MAX_INVENTORY
            robot.inventory = []
            robot.disposed_waste_count = 0

            z1_start, z1_end = self.zone_boundaries['z1']
            x = random.randint(z1_start, z1_end - 1)
            y = random.randint(0, self.height - 1)

            self.grid.place_agent(robot, (x, y))
            self.robots.append(robot)

        # Yellow robots
        for _ in range(self.num_yellow_robots):
            robot = YellowRobot(model=self, policy_name=policy_name, **kwargs)
            robot.max_inventory = ws.YELLOW_MAX_INVENTORY
            robot.inventory = []
            robot.disposed_waste_count = 0

            z2_start, z2_end = self.zone_boundaries['z2']
            x = random.randint(z2_start, z2_end - 1)
            y = random.randint(0, self.height - 1)

            self.grid.place_agent(robot, (x, y))
            self.robots.append(robot)

        # Red robots
        for _ in range(self.num_red_robots):
            robot = RedRobot(model=self, policy_name=policy_name, **kwargs)
            robot.max_inventory = ws.RED_MAX_INVENTORY
            robot.inventory = []
            robot.disposed_waste_count = 0

            z3_start, z3_end = self.zone_boundaries['z3']
            x = random.randint(z3_start, z3_end - 1)
            y = random.randint(0, self.height - 1)

            self.grid.place_agent(robot, (x, y))
            self.robots.append(robot)

    def _get_zone(self, x):
        """Determine which zone a position belongs to based on x-coordinate."""
        if x < self.zone_boundaries['z1'][1]:
            return 'z1'
        elif x < self.zone_boundaries['z2'][1]:
            return 'z2'
        else:
            return 'z3'

    def _calculate_radioactivity(self, zone):
        """Calculate random radioactivity level based on zone."""
        if zone == 'z1':
            return random.uniform(0, 0.33)
        elif zone == 'z2':
            return random.uniform(0.33, 0.66)
        else:
            return random.uniform(0.66, 1.0)

    def do(self, agent, action):
        """
        Execute an agent's action and return updated percepts.
         
        Args:
            agent: The robot agent performing the action
            action: Dictionary describing the action:
                    {'type': 'move', 'direction': (dx, dy)} |
                    {'type': 'pick_up'} |
                    {'type': 'transform'} |
                    {'type': 'put_down'} |
                    {'type': 'wait'}
        
        Returns:
            percepts: Dictionary with information about adjacent cells
        """
        if action['type'] == 'move':
            self._execute_move(agent, action)
        elif action['type'] == 'pick_up':
            self._execute_pick_up(agent)
        elif action['type'] == 'transform':
            self._execute_transform(agent)
        elif action['type'] == 'put_down':
            self._execute_put_down(agent)
        elif action['type'] == 'dispose':
            self._execute_dispose(agent)
        elif action['type'] == 'wait':
            pass
        elif action['type'] == 'read_message':
            self._read_message(agent)
        elif action['type'] == 'send_message':
            self._send_message(
                agent.unique_id,
                action.get('recipient_ids', []),
                action.get('content', None)
            )
        else:
            raise ValueError(f"Unknown action type: {action['type']}")

        return self._get_percepts(agent)

    def _execute_move(self, agent, action):
        dx, dy = action.get('direction', (0, 0))
        current_pos = agent.pos
        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy

        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            new_zone = self._get_zone(new_x)
            if agent.can_access_zone(new_zone):
                self.grid.move_agent(agent, (new_x, new_y))

    def _execute_pick_up(self, agent):
        current_pos = agent.pos

        if len(agent.inventory) >= agent.max_inventory:
            return

        objects_here = self.grid.get_cell_list_contents([current_pos])
        for obj in objects_here:
            if hasattr(obj, 'waste_type') and obj != agent:
                waste_type = obj.waste_type

                if agent.can_pick_up_type(waste_type):
                    agent.inventory.append(waste_type)
                    self.grid.remove_agent(obj)
                    if obj in self.waste_pieces:
                        self.waste_pieces.remove(obj)
                    return

    def _execute_transform(self, agent):
        if agent.robot_type == 'green':
            if agent.inventory.count('green') >= 2:
                agent.inventory.remove('green')
                agent.inventory.remove('green')
                agent.inventory.append('yellow')

        elif agent.robot_type == 'yellow':
            if agent.inventory.count('green') >= 2:
                agent.inventory.remove('green')
                agent.inventory.remove('green')
                agent.inventory.append('yellow')

            if agent.inventory.count('yellow') >= 2:
                agent.inventory.remove('yellow')
                agent.inventory.remove('yellow')
                agent.inventory.append('red')

    def _execute_put_down(self, agent):
        if agent.inventory:
            waste_type = agent.inventory.pop()

            from .objects import WasteAgent
            waste = WasteAgent(model=self, waste_type=waste_type)
            self.grid.place_agent(waste, agent.pos)
            self.waste_pieces.append(waste)

    def _execute_dispose(self, agent):
        if agent.pos == self.waste_disposal_zone.pos and agent.inventory:
            if 'red' in agent.inventory:
                agent.inventory.remove('red')
                agent.disposed_waste_count += 4
            elif 'yellow' in agent.inventory:
                agent.inventory.remove('yellow')
                agent.disposed_waste_count += 2
            elif 'green' in agent.inventory:
                agent.inventory.remove('green')
                agent.disposed_waste_count += 1

    def _read_message(self, agent):
        messages = self.mailbox.read_messages(agent.unique_id)
        agent.process_messages(messages)

    def _send_message(self, sender_id, recipient_ids, content):
        for agent in self.robots:
            if agent.unique_id in recipient_ids:
                agent.n_unread_messages += 1
        self.mailbox.send_message(sender_id, recipient_ids, content)

    def _get_percepts(self, agent):
        x, y = agent.pos
        zone = self._get_zone(x)

        percepts = {
            'position': agent.pos,
            'zone': zone,
            'inventory': agent.inventory.copy(),
            'adjacent_cells': {}
        }

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction_names = ['west', 'east', 'north', 'south']

        for direction, name in zip(directions, direction_names):
            dx, dy = direction
            adj_x, adj_y = x + dx, y + dy

            if 0 <= adj_x < self.width and 0 <= adj_y < self.height:
                adj_pos = (adj_x, adj_y)
                cell_contents = self.grid.get_cell_list_contents([adj_pos])

                cell_info = {
                    'position': adj_pos,
                    'zone': self._get_zone(adj_x),
                    'waste': [],
                    'robots': [],
                    'disposal_zone': False,
                    'radioactivity': None
                }

                for obj in cell_contents:
                    if hasattr(obj, 'waste_type'):
                        cell_info['waste'].append(obj.waste_type)
                    elif hasattr(obj, 'robot_type'):
                        cell_info['robots'].append(obj.robot_type)
                    elif obj == self.waste_disposal_zone:
                        cell_info['disposal_zone'] = True
                    elif hasattr(obj, 'radioactivity_level'):
                        cell_info['radioactivity'] = obj.radioactivity_level

                percepts['adjacent_cells'][name] = cell_info

        return percepts

    def step(self):
        """Execute one step of the simulation."""
        random.shuffle(self.robots)
        for robot in self.robots:
            robot.step()

        self.datacollector.collect(self)

        if self.is_done():
            self.running = False

    def is_done(self):
        """Determine if the mission is complete (all waste disposed)."""
        return self._count_total_waste() == 0 and all(robot.inventory == [] for robot in self.robots)

    # Data collection methods
    def _count_green_waste(self):
        return sum(1 for waste in self.waste_pieces if waste.waste_type == 'green')

    def _count_yellow_waste(self):
        return sum(1 for waste in self.waste_pieces if waste.waste_type == 'yellow')

    def _count_red_waste(self):
        return sum(1 for waste in self.waste_pieces if waste.waste_type == 'red')

    def _count_total_waste(self):
        return len(self.waste_pieces)

    def _count_disposed_waste(self):
        total_disposed = 0
        for robot in self.robots:
            if hasattr(robot, 'disposed_waste_count'):
                total_disposed += robot.disposed_waste_count
        return total_disposed