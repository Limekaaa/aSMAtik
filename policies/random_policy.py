class Policy:
    def __init__(self, model, actions:list):
        self.actions = actions
        self.model = model

    def deliberate(self, knowledge:dict):
        return self.model.random.choice(self.actions)