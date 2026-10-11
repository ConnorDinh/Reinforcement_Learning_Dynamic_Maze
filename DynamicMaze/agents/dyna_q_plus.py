from agents.q_learning import QLearningAgent
from agents.dyna_plus_model import DynaPlusModel
from agents.dyna_plus_planning import planning_plus

class DynaQPlusAgent(QLearningAgent):
    def __init__(self, rows, cols, alpha, gamma, epsilon, planning_steps, kappa):
        super().__init__(rows, cols, alpha, gamma, epsilon)
        self.dyna_q_plus_model = DynaPlusModel()
        self.planning_steps = planning_steps
        self.kappa = kappa

    def learn(self, state, action, reward, next_state, is_done):
        self.update(state, action, reward, next_state, is_done)
        self.dyna_q_plus_model.store(state, action, reward, next_state, is_done)
        planning_plus(self, self.dyna_q_plus_model, self.planning_steps, self.kappa)