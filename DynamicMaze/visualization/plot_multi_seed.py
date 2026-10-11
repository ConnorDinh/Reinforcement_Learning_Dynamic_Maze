import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_multi_seed(results):
    plt.figure(figsize=(12, 6))

    for name, runs in results.items():
        data = np.array(runs)

        mean_steps = np.mean(data, axis=0)
        std_steps = np.std(data, axis=0)

        episodes = np.arange(len(mean_steps))

        line, = plt.plot(episodes,mean_steps,label=name)

        plt.fill_between(episodes,mean_steps - std_steps,mean_steps + std_steps,color=line.get_color(),alpha=0.15)

    plt.axvline(x=200,color="black",linestyle="--",label="Shortcut opens")

    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title("Shortcut Maze: Q-Learning vs Dyna-Q vs Dyna-Q+")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / "shortcut_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
