# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

from mesa import Agent
import random


class RadioactivityAgent(Agent):
    """
    Agent representing the radioactivity level of a cell.
    """

    def __init__(self, zone, unique_id=None, model=None):
        # Generate a unique id if not provided
        if unique_id is None:
            unique_id = f"radioactivity_{random.randint(0, 999999)}"
        
        super().__init__(unique_id, model)
        
        self.zone = zone  # z1, z2 or z3

        if zone == "z1":
            self.level = random.uniform(0, 0.33)
        elif zone == "z2":
            self.level = random.uniform(0.33, 0.66)
        elif zone == "z3":
            self.level = random.uniform(0.66, 1)


class WasteDisposalZoneAgent(Agent):
    """
    Agent representing the waste disposal zone.
    This is a specific cell located in the eastern part of the grid.
    """

    def __init__(self, unique_id=None, model=None):
        # Generate a unique id if not provided
        if unique_id is None:
            unique_id = "disposal_zone"
        
        super().__init__(unique_id, model)
        
        self.type = "disposal_zone"


class WasteAgent(Agent):
    """
    Agent representing a waste object.
    """

    def __init__(self, waste_type, unique_id=None, model=None):
        # Generate a unique id if not provided
        if unique_id is None:
            unique_id = f"waste_{waste_type}_{random.randint(0, 999999)}"
        
        super().__init__(unique_id, model)
        
        self.waste_type = waste_type