import random

class DynaPlusModel:
    def __init__(self):
        self.model = {}
        self.last_visited = {}
        self.time_step = 0
        self.num_actions = 4
        self.visited_state = set()

    def store(self, state, action, reward, next_state, is_done):
        self.time_step += 1
        if state not in self.visited_state:
            for a in range(self.num_actions):
                key = (state, a)
                value = (0, state, False)
                self.model[key] = value
                self.last_visited[key] = 0
            self.visited_state.add(state)
        key = (state, action)
        value = (reward, next_state, is_done)
        self.model[key] = value
        self.last_visited[key] = self.time_step

    def sample(self):
        key, value = random.choice(list(self.model.items()))
        state, action = key
        reward, next_state, is_done = value
        tau = self.time_step - self.last_visited[(state, action)]
        return state, action, reward, next_state, is_done, tau