import random

class DynaModel:
    def __init__(self):
        self.model = {}

    def store(self, state, action, reward, next_state, is_done):
        key = (state, action)
        value = (reward, next_state, is_done)
        self.model[key] = value

    def sample(self):
        key, value = random.choice(list(self.model.items()))
        state, action = key
        reward, next_state, is_done = value
        return state, action, reward, next_state, is_done