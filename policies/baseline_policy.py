# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import random
from policies.utils import waste_here, get_accessible_neighbors

class Policy:
    def __init__(self, model, available_actions, **kwargs):
        self.model = model
        self.available_actions = available_actions

    def deliberate(self, agent, knowledge):
        """
        Decide the next action based on the robot type and percepts.
        knowledge: dict with 'position', 'zone', 'inventory'
        """

        # Meeting checkpoints
        z1_z2 = self.model.z1_z2_border
        z2_z3 = self.model.z2_z3_border
        disposal = self.model.waste_disposal_zone.pos

        # Stock info
        inv = agent.inventory
        pos = agent.pos
        zone = self.model._get_zone(pos[0])

        # --- GREEN ROBOTS ---
        if agent.robot_type == "green":
            # if 2 green -> transform
            if inv.count("green") >= 2:
                return {"type": "transform"}

            # if yellow -> go to z1/z2 checkpoint
            if "yellow" in inv:
                if pos == z1_z2:
                    return {"type": "put_down"}
                else:
                    dx = 0 if pos[0] == z1_z2[0] else (1 if z1_z2[0] > pos[0] else -1)
                    dy = 0 if pos[1] == z1_z2[1] else (1 if z1_z2[1] > pos[1] else -1)
                    return {"type": "move", "direction": (dx, dy)}

            # Pick green
            if waste_here(self.model, pos, "green"):
                return {"type": "pick_up"}

            # Otherwise explore by prioritizing unvisited neighbors
            neighbors = get_accessible_neighbors(self.model, agent, pos)
            visited = knowledge.get("visited", set())

            unvisited = [n for n in neighbors if n[0] not in visited]

            if unvisited:
                _, direction = random.choice(unvisited)
                return {"type": "move", "direction": direction}

            return {"type": "move", "direction": random.choice(neighbors)[1]}

        # --- YELLOW ROBOTS ---
        if agent.robot_type == "yellow":
            # if 2 yellow -> transform
            if inv.count("yellow") >= 2 or inv.count("green") >= 2:
                return {"type": "transform"}

            # if red -> go to z2/z3 checkpoint
            if "red" in inv:
                if pos == z2_z3:
                    return {"type": "put_down"}
                else:
                    dx = 0 if pos[0] == z2_z3[0] else (1 if z2_z3[0] > pos[0] else -1)
                    dy = 0 if pos[1] == z2_z3[1] else (1 if z2_z3[1] > pos[1] else -1)
                    return {"type": "move", "direction": (dx, dy)}

            # Pick yellow or green
            if waste_here(self.model, pos, "yellow") or waste_here(self.model, pos, "green"):
                return {"type": "pick_up"}

            # Otherwise wait at the checkpoint z1/z2
            if pos != z1_z2:
                dx = 0 if pos[0] == z1_z2[0] else (1 if z1_z2[0] > pos[0] else -1)
                dy = 0 if pos[1] == z1_z2[1] else (1 if z1_z2[1] > pos[1] else -1)
                return {"type": "move", "direction": (dx, dy)}
            return {"type": "wait"}

        # --- RED ROBOTS ---
        if agent.robot_type == "red":
            # Go to Waste Disposal Zone if has waste
            if inv:
                if pos == disposal:
                    return {"type": "dispose"}
                else:
                    dx = 0 if pos[0] == disposal[0] else (1 if disposal[0] > pos[0] else -1)
                    dy = 0 if pos[1] == disposal[1] else (1 if disposal[1] > pos[1] else -1)
                    return {"type": "move", "direction": (dx, dy)}

            # Pick red or yellow or green
            if waste_here(self.model, pos, "red") or waste_here(self.model, pos, "yellow") or waste_here(self.model, pos, "green"):
                return {"type": "pick_up"}

            # Otherwise wait at the checkpoint z2/z3
            if pos != z2_z3:
                dx = 0 if pos[0] == z2_z3[0] else (1 if z2_z3[0] > pos[0] else -1)
                dy = 0 if pos[1] == z2_z3[1] else (1 if z2_z3[1] > pos[1] else -1)
                return {"type": "move", "direction": (dx, dy)}

            return {"type": "wait"}

        # Default
        return {"type": "wait"}