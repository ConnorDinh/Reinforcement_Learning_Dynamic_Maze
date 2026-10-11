def planning(agent, model, planning_steps):
    for _ in range(planning_steps):
        state, action, reward, next_state, is_done = model.sample()
        agent.update(state, action, reward, next_state, is_done)
        