from agents.q_learning import QLearningAgent
from agents.dyna_q import DynaQAgent
from agents.dyna_q_plus import DynaQPlusAgent
from environment.maze import Maze
from experiments.train_dynamic import train_dynamic

def run_shortcut_experiment():
    initial_walls = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]

    new_walls = [(2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (2, 7)]

    start = (5, 4)
    goal = (0, 4)

    num_episodes = 400
    change_episode = 200
    max_steps = 500

    q_learning_agent = QLearningAgent(6, 9, 0.1, 0.95, 0.1)
    dyna_q_agent = DynaQAgent(6, 9, 0.1, 0.95, 0.1, 10)
    dyna_q_plus_agent = DynaQPlusAgent(rows=6, cols=9, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=10, kappa=0.001)
    maze = Maze(6, 9, start, goal, initial_walls)
    
    
    q_learning_rewards, q_learning_steps = train_dynamic(maze, q_learning_agent, initial_walls, new_walls, num_episodes, change_episode, max_steps)
    maze.reset()
    dyna_q_rewards, dyna_q_steps = train_dynamic(maze, dyna_q_agent, initial_walls, new_walls, num_episodes, change_episode, max_steps)
    maze.reset()
    dyna_q_p_rewards, dyna_q_p_steps = train_dynamic(maze, dyna_q_plus_agent, initial_walls, new_walls, num_episodes, change_episode, max_steps)
    print("____________________")
    print("Short Cut Experiment")
    for i in range(8):
        print("Q Learning: ")
        print("Average steps in episodes " + str(i*50) + " to " + str((i+1)*50 - 1) + " is: " + str(sum(q_learning_steps[i*50:i*50 + 50])/50))
        
        print("Dyna Q: ")
        print("Average steps in episodes " + str(i*50) + " to " + str((i+1)*50 - 1) + " is: " + str(sum(dyna_q_steps[i*50:i*50 + 50])/50))

        print("Dyna Q Plus: ")
        print("Average steps in episodes " + str(i*50) + " to " + str((i+1)*50 - 1) + " is: " + str(sum(dyna_q_p_steps[i*50:i*50 + 50])/50))

        
    print("Q Learning Success Rate: " + str(sum(q_learning_rewards)/len(q_learning_rewards) * 100) + "%")
    print("Dyna Q Success Rate: " + str(sum(dyna_q_rewards)/len(dyna_q_rewards) * 100) + "%")
    print("Dyna Q Plus Success Rate: " + str(sum(dyna_q_p_rewards)/len(dyna_q_p_rewards) * 100) + "%")
    print()