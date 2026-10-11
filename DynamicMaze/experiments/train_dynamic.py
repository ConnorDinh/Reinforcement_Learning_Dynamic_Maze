from agents.dyna_q import DynaQAgent
from agents.dyna_q_plus import DynaQPlusAgent

def train_dynamic(maze, agent, initial_walls, new_walls, num_episodes, change_episode, max_steps):
    maze.reset()
    maze.change_walls(initial_walls)
    episode_rewards = []
    episode_steps = []
    for episode in range(num_episodes):
        state = maze.reset()
        if episode == change_episode:
            maze.change_walls(new_walls)
        total_reward = 0
        steps_taken = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            direction = agent.actions[action]
            next_state, reward, is_done = maze.step(direction)
            if isinstance(agent, (DynaQAgent, DynaQPlusAgent)):
                agent.learn(state, action, reward, next_state, is_done)
            else:
                agent.update(state, action, reward, next_state, is_done)
            total_reward += reward
            steps_taken += 1
            state = next_state
            if is_done:
                break
        episode_rewards.append(total_reward)
        episode_steps.append(steps_taken)
    return episode_rewards, episode_steps