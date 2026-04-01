import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

TRAINING_FILES = {
    "Q-learning": {
        "Normal Traffic": "q_learning_rewards.json",
        "Peak Canteen Traffic": "q_learning_peak_canteen_rewards.json",
    },
    "DQN": {
        "Normal Traffic": "dqn_rewards.json",
        "Peak Canteen Traffic": "dqn_peak_canteen_rewards.json",
    },
}

OUTPUT_FILES = {
    "Q-learning": "q_learning_training_curve.png",
    "DQN": "dqn_training_curve.png",
}


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


def plot_algorithm_training(algorithm, scenario_files):
    plt.figure(figsize=(11, 6))

    for scenario_name, filename in scenario_files.items():
        rewards = load_rewards(filename)
        episodes = list(range(1, len(rewards) + 1))

        plt.plot(episodes, rewards, linewidth=1.5, alpha=0.45, label=f"{scenario_name} Reward")
        plt.plot(
            episodes,
            moving_average(rewards, window=5),
            linewidth=2.4,
            label=f"{scenario_name} MA(5)"
        )

        avg_reward = sum(rewards) / len(rewards)
        print(
            f"{algorithm} - {scenario_name}: "
            f"episodes={len(rewards)}, avg={avg_reward:.2f}, "
            f"min={min(rewards):.2f}, max={max(rewards):.2f}"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{algorithm} Training Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / OUTPUT_FILES[algorithm], dpi=200)
    plt.close()

    print(f"Saved {OUTPUT_FILES[algorithm]}")


def main():
    for algorithm, scenario_files in TRAINING_FILES.items():
        plot_algorithm_training(algorithm, scenario_files)


if __name__ == "__main__":
    main()
