import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

SCENARIOS = {
    "Normal Traffic": {
        "A*": "astar_rewards.json",
        "Q-learning": "q_learning_eval_rewards.json",
        "DQN": "dqn_eval_rewards.json",
    },
    "Peak Canteen Traffic": {
        "A*": "astar_peak_canteen_rewards.json",
        "Q-learning": "q_learning_peak_canteen_eval_rewards.json",
        "DQN": "dqn_peak_canteen_eval_rewards.json",
    },
}

SCENARIO_OUTPUTS = {
    "Normal Traffic": "normal_comparison.png",
    "Peak Canteen Traffic": "peak_comparison.png",
}

BAR_OUTPUT = "normal_vs_peak_bar.png"


def load_rewards(filename):
    path = BASE_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(data, window=5):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i + 1]) / len(data[start:i + 1]))
    return result


def plot_scenario(title, results, output_name):
    common_len = min(len(values) for values in results.values())
    trimmed = {name: values[:common_len] for name, values in results.items()}
    episodes = list(range(1, common_len + 1))

    plt.figure(figsize=(11, 6))

    for name, rewards in trimmed.items():
        plt.plot(episodes, rewards, marker="o", linewidth=1.8, label=name)
        plt.plot(
            episodes,
            moving_average(rewards, window=5),
            linewidth=2.2,
            linestyle="--",
            label=f"{name} MA(5)"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{title} Algorithm Comparison Using Evaluation Results")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / output_name, dpi=200)
    plt.close()

    print(f"\n{title}:")
    for name, rewards in trimmed.items():
        avg_reward = sum(rewards) / len(rewards)
        print(
            f"  {name}: avg={avg_reward:.2f}, "
            f"min={min(rewards):.2f}, max={max(rewards):.2f}"
        )


def plot_normal_vs_peak_bars(scenario_results, output_name):
    algorithms = ["A*", "Q-learning", "DQN"]
    normal_avgs = []
    peak_avgs = []

    for algorithm in algorithms:
        normal_rewards = scenario_results["Normal Traffic"][algorithm]
        peak_rewards = scenario_results["Peak Canteen Traffic"][algorithm]
        normal_avgs.append(sum(normal_rewards) / len(normal_rewards))
        peak_avgs.append(sum(peak_rewards) / len(peak_rewards))

    x = range(len(algorithms))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], normal_avgs, width=width, label="Normal Traffic")
    plt.bar([i + width / 2 for i in x], peak_avgs, width=width, label="Peak Canteen Traffic")
    plt.xticks(list(x), algorithms)
    plt.ylabel("Average Total Reward")
    plt.title("Average Evaluation Reward Across Traffic Scenarios")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(BASE_DIR / output_name, dpi=200)
    plt.close()

    print("\nAverage evaluation reward by scenario:")
    for algorithm, normal_avg, peak_avg in zip(algorithms, normal_avgs, peak_avgs):
        print(f"  {algorithm}: normal={normal_avg:.2f}, peak={peak_avg:.2f}")


def main():
    scenario_results = {}

    for scenario_name, files in SCENARIOS.items():
        scenario_results[scenario_name] = {
            algorithm: load_rewards(filename)
            for algorithm, filename in files.items()
        }

    for scenario_name, results in scenario_results.items():
        plot_scenario(scenario_name, results, SCENARIO_OUTPUTS[scenario_name])

    plot_normal_vs_peak_bars(scenario_results, BAR_OUTPUT)

    print("\nSaved figures:")
    for output_name in SCENARIO_OUTPUTS.values():
        print(f"  {output_name}")
    print(f"  {BAR_OUTPUT}")


if __name__ == "__main__":
    main()
