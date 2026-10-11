import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COLOR_MAP = ListedColormap(["#FFD16A", "#F5B52E", "#4F7D32", "#D9361E"])

def plot_maze(maze, path):
    grid = np.zeros((maze.rows, maze.cols))
    for wall in maze.walls:
        grid[wall[0]][wall[1]] = 1

    grid[maze.goal[0]][maze.goal[1]] = 2
    grid[maze.agent_position[0]][maze.agent_position[1]] = 3

    fig, ax = plt.subplots(figsize= (maze.cols,maze.rows))
    ax.imshow(grid, COLOR_MAP, vmin=0, vmax=3)
    ax.set_xticks(np.arange(maze.cols))
    ax.set_yticks(np.arange(maze.rows))
    ax.set_xticks(np.arange(-0.5, maze.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.rows, 1), minor=True)    
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Dynamic Maze Environment")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    start_r, start_c = maze.start
    goal_r, goal_c = maze.goal
    if path is not None:
        rows = [position[0] for position in path]
        cols = [position[1] for position in path]
        ax.plot(cols, rows, color="blue", linewidth=2, marker="o", markersize=4)
        
    if maze.agent_position != maze.start:
        ax.text(start_c, start_r, "S", ha="center", va="center",  color="black", fontweight="bold")

    if maze.agent_position != maze.goal:
        ax.text(goal_c, goal_r, "G", ha="center", va="center", color="white", fontweight="bold")

    agent_r, agent_c = maze.agent_position
    ax.text(agent_c, agent_r, "A", ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plt.show()