import numpy as np
import random

def evaluate_agent(maze, agent, max_steps):
    state = maze.reset()
    path = [state]
    is_done = False
    for _ in range(max_steps):
        q_values = agent.get_q_values(state)
        max_value = np.max(q_values)
        best_actions = np.where(q_values == max_value)[0]
        action = agent.actions[int(random.choice(best_actions))]
        state, _, is_done = maze.step(action)
        path.append(state)
        if is_done:
            break
    return path, is_done