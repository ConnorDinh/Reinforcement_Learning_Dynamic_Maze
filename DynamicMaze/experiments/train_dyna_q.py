def train_dyna_q(maze, agent, num_episodes, max_steps):
    episode_rewards = []
    episode_steps = []
    for episode in range(num_episodes):
        state = maze.reset()
        total_reward = 0
        steps_taken = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            direction = agent.actions[action]
            next_state, reward, is_done = maze.step(direction)
            agent.learn(state, action, reward, next_state, is_done)
            total_reward += reward
            steps_taken += 1
            state = next_state
            if is_done:
                break
        episode_rewards.append(total_reward)
        episode_steps.append(steps_taken)
    return episode_rewards, episode_steps