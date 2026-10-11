from environment.maze import Maze
from agents.q_learning import QLearningAgent
from agents.dyna_q import DynaQAgent
from experiments.train_q_learning import train_q_learning
from experiments.train_dyna_q import train_dyna_q

def compare_agents():
    maze = Maze(6, 9, (5, 0), (0, 8), [(1, 2), (2, 2), (3, 2), (4, 2), (3, 5), (4, 5)])
    q_learning_agent = QLearningAgent(6, 9, 0.1, 0.95, 0.1)
    dyna_q_agent = DynaQAgent(6, 9, 0.1, 0.95, 0.1, 10)
    q_learning_rewards, q_learning_steps = train_q_learning(maze, q_learning_agent, 300, 500)
    maze.reset()
    dyna_q_rewards, dyna_q_steps = train_dyna_q(maze, dyna_q_agent, 300, 500)
    print("Q Learning: ")
    print("Average steps in first 50 episodes: " + str(sum(q_learning_steps[:50])/50))
    print("Average steps in last 50 episodes: " + str(sum(q_learning_steps[-50:])/50))
    print("Success Rate: " + str(sum(q_learning_rewards)/len(q_learning_rewards) * 100) + "%")
    print("Dyna Q: ")
    print("Average steps in first 50 episodes: " + str(sum(dyna_q_steps[:50])/50))
    print("Average steps in last 50 episodes: " + str(sum(dyna_q_steps[-50:])/50))
    print("Success Rate: " + str(sum(dyna_q_rewards)/len(dyna_q_rewards) * 100) + "%")
