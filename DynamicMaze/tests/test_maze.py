from environment.maze import Maze
import pytest

def test_maze_movement():
    test_maze = Maze(3, 4, (2, 0), (0, 3), [(1, 1)])

    assert test_maze.reset() == (2, 0)

    test_maze.step("West")
    assert test_maze.agent_position == (2, 0)

    test_maze.step("East")
    assert test_maze.agent_position == (2, 1)

    test_maze.step("North")
    assert test_maze.agent_position == (2, 1)

    test_maze.step("East")
    test_maze.step("East")
    test_maze.step("North")
    test_maze.step("North")
    assert test_maze.agent_position == (0, 3)
    assert test_maze.reward == 1
    assert test_maze.is_done == True

    postion = test_maze.reset()
    assert postion == (2, 0)
    assert test_maze.agent_position == (2, 0)
    assert test_maze.is_done == False

def test_change_walls():
    maze = Maze(3, 4, (2, 0), (0, 3), [(1, 1)])
    # Initially, (1, 1) is blocked.
    maze.reset()
    maze.step("East")
    maze.step("North")
    assert maze.agent_position == (2, 1)

    # Remove the old wall and add a new wall at (1, 2).
    maze.change_walls([(1, 2)])

    assert maze.walls == {(1, 2)}

    # The previously blocked cell is now accessible.
    maze.step("North")
    assert maze.agent_position == (1, 1)

    # The newly blocked cell cannot be entered.
    maze.step("East")
    assert maze.agent_position == (1, 1)

    # Invalid changes must be rejected.
    with pytest.raises(Exception):
        maze.change_walls([(0, 3)])  # Goal

    with pytest.raises(Exception):
        maze.change_walls([(3, 0)])  # Outside grid

    with pytest.raises(Exception):
        maze.change_walls([(1, 1)])  # Current agent position

    # Invalid changes must leave the existing walls unchanged.
    assert maze.walls == {(1, 2)}