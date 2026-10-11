
import unittest

from agents.dyna_plus_model import DynaPlusModel


class TestDynaPlusModel(unittest.TestCase):

    def test_initializes_all_actions(self):
        model = DynaPlusModel()

        state = (2, 3)
        next_state = (2, 4)

        model.store(
            state,
            action=1,
            reward=0,
            next_state=next_state,
            is_done=False
        )

        # All four actions should exist
        for action in range(4):
            self.assertIn(
                (state, action),
                model.model
            )

        # Untried actions should be self-loops
        self.assertEqual(
            model.model[(state, 0)],
            (0, state, False)
        )

        # Real action should store actual transition
        self.assertEqual(
            model.model[(state, 1)],
            (0, next_state, False)
        )

    def test_timestamps(self):
        model = DynaPlusModel()

        state = (2, 3)

        model.store(state, 0, 0, (2, 4), False)
        model.store(state, 1, 0, (3, 3), False)

        self.assertEqual(model.time_step, 2)
        self.assertEqual(model.last_visited[(state, 0)], 1)
        self.assertEqual(model.last_visited[(state, 1)], 2)

        # Untried action retains timestamp zero
        self.assertEqual(model.last_visited[(state, 2)], 0)

    def test_does_not_reset_existing_actions(self):
        model = DynaPlusModel()

        state = (2, 3)

        model.store(state, 0, 1, (2, 4), False)
        model.store(state, 1, 0, (3, 3), False)

        self.assertEqual(
            model.model[(state, 0)],
            (1, (2, 4), False)
        )


if __name__ == "__main__":
    unittest.main()
