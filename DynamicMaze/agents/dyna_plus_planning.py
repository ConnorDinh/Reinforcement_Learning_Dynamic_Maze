import math

def planning_plus(agent, model, planning_steps, kappa):
    for _ in range(planning_steps):
        state, action, reward, next_state, is_done, tau = model.sample()
        bonus = kappa * math.sqrt(tau)
        planning_reward = reward + bonus
        agent.update(state, action, planning_reward, next_state, is_done)
        