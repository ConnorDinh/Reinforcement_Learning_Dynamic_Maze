from agents.dyna_q import DynaQAgent
from agents.q_learning import QLearningAgent
from environment.maze import Maze
from experiments.train_dynamic import train_dynamic

def run_blocking_experiment():
    initial_walls = [(1, 4), (2, 4), (3, 4), (4, 4)]
    new_walls = initial_walls + [(0, 4)]
    q_learning_agent = QLearningAgent(6, 9, 0.1, 0.95, 0.1)
    dyna_q_agent = DynaQAgent(6, 9, 0.1, 0.95, 0.1, 10)
    maze = Maze(6, 9, (5, 0), (0, 8), initial_walls)
    
    
    q_learning_rewards, q_learning_steps = train_dynamic(maze, q_learning_agent, initial_walls, new_walls, 400, 200, 500)
    maze.reset()
    dyna_q_rewards, dyna_q_steps = train_dynamic(maze, dyna_q_agent, initial_walls, new_walls, 400, 200, 500)
    print("____________________")
    print("Blocking Experiment")
    for i in range(8):
        print("Q Learning: ")
        print("Average steps in episodes " + str(i*50) + " to " + str((i+1)*50 - 1) + " is: " + str(sum(q_learning_steps[i*50:i*50 + 50])/50))
        
        print("Dyna Q: ")
        print("Average steps in episodes " + str(i*50) + " to " + str((i+1)*50 - 1) + " is: " + str(sum(dyna_q_steps[i*50:i*50 + 50])/50))

        
    print("Q Learning Success Rate: " + str(sum(q_learning_rewards)/len(q_learning_rewards) * 100) + "%")
    print("Dyna Q Success Rate: " + str(sum(dyna_q_rewards)/len(dyna_q_rewards) * 100) + "%")
    print()

    