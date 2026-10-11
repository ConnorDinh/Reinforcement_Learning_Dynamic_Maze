
import csv
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path("outputs")


def save_results(results, evaluations):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save episode-level results
    steps_path = OUTPUT_DIR / "episode_steps.csv"

    with open(steps_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "algorithm",
            "seed",
            "episode",
            "steps"
        ])

        for name, runs in results.items():
            for seed, episode_steps in enumerate(runs):
                for episode, steps in enumerate(episode_steps):
                    writer.writerow([
                        name,
                        seed,
                        episode,
                        steps
                    ])

    # 2. Save summary statistics
    summary_path = OUTPUT_DIR / "summary.csv"

    with open(summary_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "algorithm",
            "before_change",
            "after_change",
            "final_50",
            "success_rate",
            "shortcut_rate"
        ])

        for name, runs in results.items():
            data = np.asarray(runs)

            before_change = np.mean(data[:, 150:200])
            after_change = np.mean(data[:, 200:250])
            final_50 = np.mean(data[:, 350:400])

            evals = evaluations[name]

            success_rate = np.mean([
                item["success"] for item in evals
            ])

            shortcut_rate = np.mean([
                item["shortcut"] for item in evals
            ])

            writer.writerow([
                name,
                round(before_change, 2),
                round(after_change, 2),
                round(final_50, 2),
                round(success_rate, 3),
                round(shortcut_rate, 3)
            ])

    # 3. Save greedy evaluation details
    evaluation_path = OUTPUT_DIR / "evaluations.csv"

    with open(evaluation_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "algorithm",
            "seed",
            "success",
            "steps",
            "shortcut",
            "optimal"
        ])

        for name, evals in evaluations.items():
            for seed, item in enumerate(evals):
                writer.writerow([
                    name,
                    seed,
                    item["success"],
                    item["steps"],
                    item["shortcut"],
                    item["success"] and item["steps"] == 5
                ])

    print(f"Results saved to {OUTPUT_DIR.resolve()}")
