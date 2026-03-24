# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import random
from policies.utils import waste_here, get_accessible_neighbors

HOLDING_THRESHOLD = 10

class Policy:
    def __init__(self, model, available_actions, **kwargs):
        self.model = model
        self.available_actions = available_actions
        self.holding_threshold = kwargs.get("holding_threshold", HOLDING_THRESHOLD)
        self.is_first_step = True


    def process_messages(self, agent, messages):
        """Process received messages and update knowledge."""
        for msg in messages:
            if msg["content"]["type"] == "visit_status":
                agent.knowledge["everything_visited"] = True
        agent.n_unread_messages -= len(messages)


    def deliberate(self, agent):
        """
        Decide the next action based on the robot type and percepts.
        knowledge: dict with 'position', 'zone', 'inventory'
        """

        if self.is_first_step:
            agent.knowledge["visited"] = set()
            agent.knowledge["untaken_waste"] = set()
            agent.knowledge["holding_steps"] = 0
            agent.knowledge["adjacent_cells"] = {}
            agent.knowledge["everything_visited"] = False
            agent.knowledge["green_agents_ids"] = [a.unique_id for a in self.model.schedule.agents if a.robot_type == "green"] 
            agent.knowledge["yellow_agents_ids"] = [a.unique_id for a in self.model.schedule.agents if a.robot_type == "yellow"]
            agent.knowledge["red_agents_ids"] = [a.unique_id for a in self.model.schedule.agents if a.robot_type == "red"]
            self.is_first_step = False


        agent.knowledge["visited"].add(agent.pos)

        if agent.n_unread_messages > 0:
            return {"type": "read_message"}

        if agent.pos in agent.knowledge["untaken_waste"] and not waste_here(self.model, agent.pos, "green"):
            agent.knowledge["untaken_waste"].remove(agent.pos)

        for cell in agent.knowledge.get("adjacent_cells", {}).values():
            if "green" in cell["waste"]:
                agent.knowledge["untaken_waste"].add(cell["position"])

        if agent.inventory:
            agent.knowledge["holding_steps"] += 1
        else:
            agent.knowledge["holding_steps"] = 0

        knowledge = agent.knowledge

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
            
            # If knows about unpicked green waste, go there
            untaken_waste = knowledge.get("untaken_waste", set())
            visited = knowledge.get("visited", set())
            if len(inv) < agent.max_inventory and untaken_waste:
                target = random.choice(list(untaken_waste))

                dx = 0 if pos[0] == target[0] else (1 if target[0] > pos[0] else -1)
                dy = 0 if pos[1] == target[1] else (1 if target[1] > pos[1] else -1)
                return {"type": "move", "direction": (dx, dy)}

            # If has visited whole grid and knows no unpicked green waste, inform green robots
            z1_start, z1_end = self.model.zone_boundaries['z1']
            z1_size = (z1_end - z1_start) * self.model.height

            if not untaken_waste and len(visited) >= z1_size and not knowledge.get("everything_visited", False):
                agent.knowledge["everything_visited"] = True
                return {"type": "send_message", "recipient_ids": agent.knowledge["green_agents_ids"], "content": {"type": "visit_status"}}

            # If has green waste but no known unpicked waste and has visited whole grid, go to checkpoint


            if (
                "green" in inv
                and not untaken_waste
                and knowledge.get("everything_visited", False)
            ):
                if pos == z1_z2:
                    return {"type": "put_down"}
                else:
                    dx = 0 if pos[0] == z1_z2[0] else (1 if z1_z2[0] > pos[0] else -1)
                    dy = 0 if pos[1] == z1_z2[1] else (1 if z1_z2[1] > pos[1] else -1)
                    return {"type": "move", "direction": (dx, dy)}

            # Otherwise explore by prioritizing unvisited neighbors
            neighbors = get_accessible_neighbors(self.model, agent, pos)

            unvisited = [n for n in neighbors if n[0] not in visited]

            if unvisited:
                _, direction = random.choice(unvisited)
                return {"type": "move", "direction": direction}

            return {"type": "move", "direction": random.choice(neighbors)[1]}

        # --- YELLOW ROBOTS ---
        if agent.robot_type == "yellow":
            # if 2 yellow or 2 green -> transform
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
            
            # If holding waste for too long, go to checkpoint to put it down
            if inv and knowledge.get("holding_steps", 0) > self.holding_threshold:
                if pos == z2_z3:
                    return {"type": "put_down"}
                else:
                    dx = 0 if pos[0] == z2_z3[0] else (1 if z2_z3[0] > pos[0] else -1)
                    dy = 0 if pos[1] == z2_z3[1] else (1 if z2_z3[1] > pos[1] else -1)
                    return {"type": "move", "direction": (dx, dy)}

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