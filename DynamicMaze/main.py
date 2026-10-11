from experiments.multi_seed_experiment import run_multi_seed_experiment
from experiments.save_results import save_results
from visualization.plot_multi_seed import plot_multi_seed


def main():
    results, evaluations = run_multi_seed_experiment(
        num_seeds=10
    )

    save_results(results, evaluations)

    plot_multi_seed(results)


if __name__ == "__main__":
    main()