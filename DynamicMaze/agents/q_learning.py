import numpy as np
import random

class QLearningAgent:
    def __init__(self, rows, cols, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = {0: "North", 1: "South", 2: "West", 3: "East"}
        self.num_actions = 4
        self.q_table = np.zeros((rows, cols, self.num_actions))

    def get_q_values(self, state):
        return self.q_table[state[0], state[1], :]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        q_values = self.get_q_values(state)
        max_value = np.max(q_values)
        best_actions = np.where(q_values == max_value)[0]
        return int(random.choice(best_actions))

    def update(self, state, action, reward, next_state, is_done):
        old_q_value = self.q_table[state[0]][state[1]][action]
        if is_done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            max_value = np.max(next_q_values)
            target = reward + self.gamma * max_value
        self.q_table[state[0]][state[1]][action] += self.alpha * (target - old_q_value)
        