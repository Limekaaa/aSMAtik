# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import random


class Policy:
    def __init__(self, model, actions: list):
        self.actions = actions
        self.model = model

    def deliberate(self, agent):
        """Return an action as a dictionary."""
        action_type = self.model.random.choice(self.actions)
        
        # Build action dictionary based on action type
        if action_type == 'move':
            # Random direction: up, down, left, right
            direction = self.model.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            return {'type': 'move', 'direction': direction}
        elif action_type == 'pick_up':
            return {'type': 'pick_up'}
        elif action_type == 'transform':
            return {'type': 'transform'}
        elif action_type == 'put_down':
            return {'type': 'put_down'}
        elif action_type == 'dispose':
            return {'type': 'dispose'}
        else:
            return {'type': 'wait'}