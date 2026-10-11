
import unittest

from agents.q_learning import QLearningAgent


class TestQLearning(unittest.TestCase):

    def test_q_update(self):
        agent = QLearningAgent(
            rows=6,
            cols=9,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )

        state = (2, 3)
        next_state = (2, 4)
        action = 0

        # Set next state's best Q-value to 2
        agent.q_table[2, 4, 1] = 2.0

        agent.update(
            state,
            action,
            reward=1,
            next_state=next_state,
            is_done=False
        )

        self.assertAlmostEqual(
            agent.q_table[2, 3, 0],
            0.28
        )

    def test_terminal_update(self):
        agent = QLearningAgent(
            6, 9, 0.1, 0.9, 0.1
        )

        state = (2, 3)
        next_state = (0, 4)

        # Terminal transitions should not bootstrap
        agent.q_table[0, 4, 0] = 10.0

        agent.update(
            state,
            action=0,
            reward=1,
            next_state=next_state,
            is_done=True
        )

        self.assertAlmostEqual(
            agent.q_table[2, 3, 0],
            0.1
        )


if __name__ == "__main__":
    unittest.main()
