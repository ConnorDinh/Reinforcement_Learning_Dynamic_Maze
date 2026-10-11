
import unittest

from agents.dyna_plus_planning import planning_plus


class MockAgent:
    def __init__(self):
        self.updates = []

    def update(self, state, action, reward, next_state, is_done):
        self.updates.append(
            (state, action, reward, next_state, is_done)
        )


class MockModel:
    def sample(self):
        return (
            (2, 3),   # state
            1,        # action
            0,        # reward
            (2, 3),   # next state
            False,    # is_done
            400       # tau
        )


class TestDynaPlusPlanning(unittest.TestCase):

    def test_exploration_bonus(self):
        agent = MockAgent()
        model = MockModel()

        planning_plus(
            agent,
            model,
            planning_steps=1,
            kappa=0.001
        )

        self.assertEqual(len(agent.updates), 1)

        state, action, reward, next_state, is_done = (
            agent.updates[0]
        )

        # 0.001 * sqrt(400) = 0.02
        self.assertAlmostEqual(reward, 0.02)

    def test_zero_bonus(self):
        agent = MockAgent()
        model = MockModel()

        planning_plus(
            agent,
            model,
            planning_steps=1,
            kappa=0
        )

        self.assertEqual(agent.updates[0][2], 0)


if __name__ == "__main__":
    unittest.main()
