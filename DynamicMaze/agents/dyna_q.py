from agents.q_learning import QLearningAgent
from agents.dyna_model import DynaModel
from agents.dyna_planning import planning

class DynaQAgent(QLearningAgent):
    def __init__(self, rows, cols, alpha, gamma, epsilon, planning_steps):
        super().__init__(rows, cols, alpha, gamma, epsilon)
        self.dyna_model = DynaModel()
        self.planning_steps = planning_steps

    def learn(self, state, action, reward, next_state, is_done):
        self.update(state, action, reward, next_state, is_done)
        self.dyna_model.store(state, action, reward, next_state, is_done)
        planning(self, self.dyna_model, self.planning_steps)