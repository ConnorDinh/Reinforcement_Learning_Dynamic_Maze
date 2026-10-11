import random
import numpy as np

from agents.q_learning import QLearningAgent
from agents.dyna_q import DynaQAgent
from agents.dyna_q_plus import DynaQPlusAgent
from environment.maze import Maze
from experiments.train_dynamic import train_dynamic
from experiments.evaluate import evaluate_agent


def run_multi_seed_experiment(num_seeds=10):
    initial_walls = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]

    new_walls = [(2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (2, 7)]

    num_episodes = 400
    change_episode = 200
    max_steps = 500

    results = {"Q-Learning": [],"Dyna-Q": [],"Dyna-Q+": []}

    evaluations = {"Q-Learning": [],"Dyna-Q": [],"Dyna-Q+": []}

    for seed in range(num_seeds):
        print(f"Running seed {seed + 1}/{num_seeds}")

        agents = {
            "Q-Learning": QLearningAgent(6, 9, 0.1, 0.95, 0.1),
            "Dyna-Q": DynaQAgent(6, 9, 0.1, 0.95, 0.1, 10),
            "Dyna-Q+": DynaQPlusAgent(6, 9, 0.1, 0.95, 0.1, 10, 0.001)
        }

        for name, agent in agents.items():
            # Reproducible training for each algorithm
            random.seed(seed)
            np.random.seed(seed)

            maze = Maze(6, 9,(5, 4),(0, 4),initial_walls)

            rewards, steps = train_dynamic(maze,agent,initial_walls,new_walls,num_episodes,change_episode,max_steps)

            results[name].append(steps)

            # Greedy evaluation in the final maze
            path, success = evaluate_agent(maze, agent, max_steps)

            evaluations[name].append({"success": success,"steps": len(path) - 1,"shortcut": success and (2, 4) in path})

    print("\n===== MULTI-SEED RESULTS =====")

    for name in results:
        all_steps = np.array(results[name])

        mean_steps = np.mean(all_steps, axis=0)
        std_steps = np.std(all_steps, axis=0)

        print(f"\n{name}")

        print("Before change (150–199):", round(np.mean(mean_steps[150:200]), 2))

        print("After change (200–249):", round(np.mean(mean_steps[200:250]), 2))

        print("Final 50 episodes:", round(np.mean(mean_steps[350:400]), 2))

        successful = sum(item["success"] for item in evaluations[name])

        shortcuts = sum(item["shortcut"] for item in evaluations[name])

        print(f"Greedy evaluation success: " f"{successful}/{num_seeds}")

        print(f"Shortcut discovery: " f"{shortcuts}/{num_seeds}")

    return results, evaluations