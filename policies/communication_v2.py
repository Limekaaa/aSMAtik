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

    # -----------------------------
    # Message processing
    # -----------------------------
    def process_messages(self, agent, messages):
        """Process received messages and update knowledge."""
        for msg in messages:
            content = msg["content"]
            msg_type = content.get("type")

            if msg_type == "visit_status":
                agent.knowledge["everything_visited"] = True

            elif msg_type == "waste_spotted":
                position = tuple(content["position"])
                waste_type = content.get("waste_type")

                agent.knowledge["untaken_waste"].add(position)

                if "reported_wastes" not in agent.knowledge:
                    agent.knowledge["reported_wastes"] = {}
                if waste_type is not None:
                    agent.knowledge["reported_wastes"][position] = waste_type

            elif msg_type == "red_ready_on_border":
                position = tuple(content["position"])
                agent.knowledge["untaken_waste"].add(position)

                if "reported_wastes" not in agent.knowledge:
                    agent.knowledge["reported_wastes"] = {}
                agent.knowledge["reported_wastes"][position] = "red"

        agent.n_unread_messages -= len(messages)

    # -----------------------------
    # Knowledge init
    # -----------------------------
    def _init_knowledge(self, agent):
        agent.knowledge["visited"] = set()
        agent.knowledge["untaken_waste"] = set()
        agent.knowledge["holding_steps"] = 0
        agent.knowledge["adjacent_cells"] = {}
        agent.knowledge["everything_visited"] = False
        agent.knowledge["already_reported"] = set()
        agent.knowledge["reported_wastes"] = {}
        agent.knowledge["exploration_target"] = None
        agent.knowledge["already_signaled_red_drop"] = set()
        agent.knowledge["border_patrol_direction"] = 1

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

    # -----------------------------
    # Utility methods
    # -----------------------------
    def _position_has_any_waste(self, pos):
        return (
            waste_here(self.model, pos, "green")
            or waste_here(self.model, pos, "yellow")
            or waste_here(self.model, pos, "red")
        )

    def _move_towards(self, pos, target):
        dx = 0 if pos[0] == target[0] else (1 if target[0] > pos[0] else -1)
        dy = 0 if pos[1] == target[1] else (1 if target[1] > pos[1] else -1)
        return {"type": "move", "direction": (dx, dy)}

    def _get_closest_target(self, pos, targets):
        def manhattan(target):
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        return min(targets, key=manhattan) if targets else None

    def _get_targets_of_type(self, knowledge, waste_type):
        reported = knowledge.get("reported_wastes", {})
        return {pos for pos, wtype in reported.items() if wtype == waste_type}

    def _get_unvisited_cells_in_z1(self, visited):
        z1_start, z1_end = self.model.zone_boundaries["z1"]
        unvisited = []

        for x in range(z1_start, z1_end):
            for y in range(self.model.height):
                if (x, y) not in visited:
                    unvisited.append((x, y))

        return unvisited

    def _get_closest_unvisited_in_z1(self, pos, visited):
        unvisited = self._get_unvisited_cells_in_z1(visited)
        return self._get_closest_target(pos, unvisited)

    def _is_on_z1_z2_border(self, pos):
        return pos[0] == self.model.zone_boundaries["z1"][1] - 1

    def _is_on_z2_z3_border(self, pos):
        return pos[0] == self.model.zone_boundaries["z2"][1] - 1

    def _move_towards_z1_z2_border(self, pos):
        target_x = self.model.zone_boundaries["z1"][1] - 1
        return self._move_towards(pos, (target_x, pos[1]))

    def _move_towards_z2_z3_border(self, pos):
        target_x = self.model.zone_boundaries["z2"][1] - 1
        return self._move_towards(pos, (target_x, pos[1]))

    def _patrol_border(self, agent, pos, border_type):
        if border_type == "z1_z2":
            border_x = self.model.zone_boundaries["z1"][1] - 1
        else:
            border_x = self.model.zone_boundaries["z2"][1] - 1

        if pos[0] != border_x:
            return self._move_towards(pos, (border_x, pos[1]))

        direction = agent.knowledge.get("border_patrol_direction", 1)
        next_y = pos[1] + direction

        if 0 <= next_y < self.model.height:
            return {"type": "move", "direction": (0, direction)}

        agent.knowledge["border_patrol_direction"] = -direction
        return {"type": "move", "direction": (0, -direction)}

    def _maybe_read_message(self, agent):
        """Read messages early unless an urgent local action should take priority."""
        if agent.n_unread_messages <= 0:
            return None

        pos = agent.pos
        inv = agent.inventory

        # Urgent local actions keep priority
        if agent.robot_type == "green":
            if inv.count("green") >= 2:
                return None
            if "yellow" in inv and self._is_on_z1_z2_border(pos):
                return None
            if waste_here(self.model, pos, "green") and not self._is_on_z1_z2_border(pos):
                return None

        elif agent.robot_type == "yellow":
            if inv.count("yellow") >= 2 or inv.count("green") >= 2:
                return None
            if "red" in inv and self._is_on_z2_z3_border(pos):
                return None
            if not self._is_on_z2_z3_border(pos):
                if not inv and (waste_here(self.model, pos, "green") or waste_here(self.model, pos, "yellow")):
                    return None

        elif agent.robot_type == "red":
            disposal = self.model.waste_disposal_zone.pos
            if inv and pos == disposal:
                return None
            if pos != disposal and (
                waste_here(self.model, pos, "red")
                or waste_here(self.model, pos, "yellow")
                or waste_here(self.model, pos, "green")
            ):
                return None

        return {"type": "read_message"}

    def _update_local_knowledge_from_adjacent(self, agent):
        """Update untaken_waste using visible cells."""
        if agent.pos in agent.knowledge["untaken_waste"] and not self._position_has_any_waste(agent.pos):
            agent.knowledge["untaken_waste"].discard(agent.pos)
            if "reported_wastes" in agent.knowledge:
                agent.knowledge["reported_wastes"].pop(agent.pos, None)

        for cell in agent.knowledge.get("adjacent_cells", {}).values():
            pos_cell = cell["position"]

            # Ignore only the transfer point that corresponds to the robot's own drop behavior
            ignore = False
            if agent.robot_type == "green" and self._is_on_z1_z2_border(pos_cell):
                ignore = True
            elif agent.robot_type == "yellow" and self._is_on_z2_z3_border(pos_cell):
                ignore = True
            elif agent.robot_type == "red" and pos_cell == self.model.waste_disposal_zone.pos:
                ignore = True

            if ignore:
                agent.knowledge["untaken_waste"].discard(pos_cell)
                if "reported_wastes" in agent.knowledge:
                    agent.knowledge["reported_wastes"].pop(pos_cell, None)
                continue

            visible_pickable = any(agent.can_pick_up_type(w) for w in cell["waste"])

            if visible_pickable:
                agent.knowledge["untaken_waste"].add(pos_cell)
            else:
                agent.knowledge["untaken_waste"].discard(pos_cell)
                if "reported_wastes" in agent.knowledge:
                    agent.knowledge["reported_wastes"].pop(pos_cell, None)

    def _maybe_report_unreachable_waste(self, agent):
        """
        If the agent sees waste it cannot pick up, it may report it.
        """
        for cell in agent.knowledge.get("adjacent_cells", {}).values():
            pos_cell = cell["position"]

            # Do not report wastes on transfer/disposal places
            if (
                self._is_on_z1_z2_border(pos_cell)
                or self._is_on_z2_z3_border(pos_cell)
                or pos_cell == self.model.waste_disposal_zone.pos
            ):
                continue

            for waste_type in cell["waste"]:
                if agent.can_pick_up_type(waste_type):
                    continue

                report_key = (waste_type, pos_cell)
                if report_key in agent.knowledge["already_reported"]:
                    continue

                recipients = []
                if waste_type == "yellow":
                    recipients = agent.knowledge["yellow_agents_ids"]
                elif waste_type == "red":
                    recipients = agent.knowledge["red_agents_ids"]

                if recipients:
                    agent.knowledge["already_reported"].add(report_key)
                    return {
                        "type": "send_message",
                        "recipient_ids": recipients,
                        "content": {
                            "type": "waste_spotted",
                            "position": pos_cell,
                            "waste_type": waste_type,
                        },
                    }

        return None

    # -----------------------------
    # Main decision
    # -----------------------------
    def deliberate(self, agent):
        if self.is_first_step:
            self._init_knowledge(agent)
            self.is_first_step = False

        knowledge = agent.knowledge
        pos = agent.pos
        inv = agent.inventory

        # Clean obsolete targets when reaching them (especially those coming from messages)
        if pos in knowledge.get("untaken_waste", set()) and not self._position_has_any_waste(pos):
            knowledge["untaken_waste"].discard(pos)
            if "reported_wastes" in knowledge:
                knowledge["reported_wastes"].pop(pos, None)

        knowledge["visited"].add(pos)

        read_action = self._maybe_read_message(agent)
        if read_action is not None:
            return read_action

        self._update_local_knowledge_from_adjacent(agent)

        report_action = self._maybe_report_unreachable_waste(agent)
        if report_action is not None:
            return report_action

        if inv:
            knowledge["holding_steps"] += 1
        else:
            knowledge["holding_steps"] = 0

        disposal = self.model.waste_disposal_zone.pos
        untaken_waste = knowledge.get("untaken_waste", set())
        visited = knowledge.get("visited", set())

        # ==========================================================
        # GREEN ROBOTS
        # ==========================================================
        if agent.robot_type == "green":
            z1_start, z1_end = self.model.zone_boundaries["z1"]
            z1_size = (z1_end - z1_start) * self.model.height

            # Transform first
            if inv.count("green") >= 2:
                return {"type": "transform"}

            # If holding yellow, forward it
            if "yellow" in inv:
                if self._is_on_z1_z2_border(pos):
                    return {"type": "put_down"}
                return self._move_towards_z1_z2_border(pos)

            # Pick green on current cell, except on transfer border
            if waste_here(self.model, pos, "green") and not self._is_on_z1_z2_border(pos):
                return {"type": "pick_up"}

            # Signal exploration finished
            if (
                not untaken_waste
                and len(visited) >= z1_size
                and not knowledge.get("everything_visited", False)
            ):
                knowledge["everything_visited"] = True
                if knowledge["green_agents_ids"]:
                    return {
                        "type": "send_message",
                        "recipient_ids": knowledge["green_agents_ids"],
                        "content": {"type": "visit_status"},
                    }

            # Endgame: move any remaining green forward, even alone
            if knowledge.get("everything_visited", False):
                if inv:
                    if self._is_on_z1_z2_border(pos):
                        return {"type": "put_down"}
                    return self._move_towards_z1_z2_border(pos)

                if untaken_waste:
                    target = self._get_closest_target(pos, untaken_waste)
                    return self._move_towards(pos, target)

                if not self._is_on_z1_z2_border(pos):
                    return self._move_towards_z1_z2_border(pos)

                # Patrol border instead of blocking
                return self._patrol_border(agent, pos, "z1_z2")

            # Normal exploration
            # 1) Priorité aux déchets connus
            if len(inv) < agent.max_inventory and untaken_waste:
                target = self._get_closest_target(pos, untaken_waste)
                return self._move_towards(pos, target)

            # 2) Sinon, aller vers la case non visitée la plus proche de z1
            target = self._get_closest_unvisited_in_z1(pos, visited)
            knowledge["exploration_target"] = target
            if target is not None and target != pos:
                return self._move_towards(pos, target)

            # 3) Si tout z1 a été visité, comportement de fin (déplacement neutre)
            neighbors = get_accessible_neighbors(self.model, agent, pos)
            if neighbors:
                return {"type": "move", "direction": random.choice(neighbors)[1]}

            return {"type": "wait"}

        # ==========================================================
        # YELLOW ROBOTS
        # ==========================================================
        if agent.robot_type == "yellow":
            # Always transform as soon as possible
            if inv.count("yellow") >= 2 or inv.count("green") >= 2:
                return {"type": "transform"}

            # Forward red immediately + notify red robots when ready on border
            if "red" in inv:
                if self._is_on_z2_z3_border(pos):
                    if (
                        pos not in knowledge.get("already_signaled_red_drop", set())
                        and knowledge.get("red_agents_ids")
                    ):
                        knowledge["already_signaled_red_drop"].add(pos)
                        return {
                            "type": "send_message",
                            "recipient_ids": knowledge["red_agents_ids"],
                            "content": {
                                "type": "red_ready_on_border",
                                "position": pos,
                            },
                        }
                    return {"type": "put_down"}

                return self._move_towards_z2_z3_border(pos)

            # Pick only compatible waste on current cell
            if not self._is_on_z2_z3_border(pos):
                if not inv:
                    if waste_here(self.model, pos, "green") or waste_here(self.model, pos, "yellow"):
                        return {"type": "pick_up"}

                elif all(w == "green" for w in inv):
                    if waste_here(self.model, pos, "green"):
                        return {"type": "pick_up"}

                elif all(w == "yellow" for w in inv):
                    if waste_here(self.model, pos, "yellow"):
                        return {"type": "pick_up"}

            # If carrying green/yellow, FIRST try to find a second compatible waste
            if inv:
                compatible_targets = self._get_compatible_targets_for_yellow(agent, untaken_waste)

                if compatible_targets:
                    target = self._get_closest_target(pos, compatible_targets)
                    return self._move_towards(pos, target)

                # Stay around z1/z2 for a while to improve chances of transform
                if knowledge.get("holding_steps", 0) <= 2 * self.holding_threshold:
                    if not self._is_on_z1_z2_border(pos):
                        return self._move_towards_z1_z2_border(pos)
                    return self._patrol_border(agent, pos, "z1_z2")

                # Only after that, move it forward
                if self._is_on_z2_z3_border(pos):
                    return {"type": "put_down"}
                return self._move_towards_z2_z3_border(pos)

            # If empty, go where known waste is
            compatible_targets = self._get_compatible_targets_for_yellow(agent, untaken_waste)
            if compatible_targets:
                target = self._get_closest_target(pos, compatible_targets)
                return self._move_towards(pos, target)

            # Otherwise stay near z1/z2 and patrol
            if not self._is_on_z1_z2_border(pos):
                return self._move_towards_z1_z2_border(pos)

            return self._patrol_border(agent, pos, "z1_z2")

        # ==========================================================
        # RED ROBOTS
        # ==========================================================
        if agent.robot_type == "red":
            # If carrying anything, go dispose
            if inv:
                if pos == disposal:
                    return {"type": "dispose"}
                return self._move_towards(pos, disposal)

            # Pick on current cell
            if (
                (
                    waste_here(self.model, pos, "red")
                    or waste_here(self.model, pos, "yellow")
                    or waste_here(self.model, pos, "green")
                )
                and pos != disposal
            ):
                return {"type": "pick_up"}

            # Go to known waste if any (prioritize reported red targets)
            red_targets = self._get_targets_of_type(knowledge, "red")
            if red_targets:
                target = self._get_closest_target(pos, red_targets)
                return self._move_towards(pos, target)

            if untaken_waste:
                target = self._get_closest_target(pos, untaken_waste)
                return self._move_towards(pos, target)

            # Otherwise patrol z2/z3 instead of freezing
            if not self._is_on_z2_z3_border(pos):
                return self._move_towards_z2_z3_border(pos)

            return self._patrol_border(agent, pos, "z2_z3")

        return {"type": "wait"}
    
    def _get_compatible_targets_for_yellow(self, agent, targets):
        inv = agent.inventory
        compatible = set()

        for target in targets:
            if not inv:
                if waste_here(self.model, target, "green") or waste_here(self.model, target, "yellow"):
                    compatible.add(target)

            elif all(w == "green" for w in inv):
                if waste_here(self.model, target, "green"):
                    compatible.add(target)

            elif all(w == "yellow" for w in inv):
                if waste_here(self.model, target, "yellow"):
                    compatible.add(target)

        return compatible