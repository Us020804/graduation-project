import json
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. 读取数据
# ===============================

# 普通场景
with open("astar_rewards.json") as f:
    astar_normal = json.load(f)

with open("q_learning_rewards.json") as f:
    ql_normal = json.load(f)

with open("dqn_rewards.json") as f:
    dqn_normal = json.load(f)

with open("dqn_eval_rewards.json") as f:
    dqn_eval_normal = json.load(f)

# 高峰场景
with open("astar_peak_canteen_rewards.json") as f:
    astar_peak = json.load(f)

with open("q_learning_peak_canteen_rewards.json") as f:
    ql_peak = json.load(f)

with open("dqn_peak_canteen_rewards.json") as f:
    dqn_peak = json.load(f)

with open("dqn_peak_canteen_eval_rewards.json") as f:
    dqn_eval_peak = json.load(f)

# ===============================
# 2. 图1：普通场景三算法对比
# ===============================
plt.figure()
plt.plot(astar_normal, label="A*")
plt.plot(ql_normal, label="Q-learning")
plt.plot(dqn_normal, label="DQN")
plt.title("Normal Traffic Scenario")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.savefig("normal_comparison.png")

# ===============================
# 3. 图2：高峰场景三算法对比
# ===============================
plt.figure()
plt.plot(astar_peak, label="A*")
plt.plot(ql_peak, label="Q-learning")
plt.plot(dqn_peak, label="DQN")
plt.title("Peak Traffic Scenario")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.savefig("peak_comparison.png")

# ===============================
# 4. 图3：平均奖励柱状图（重点）
# ===============================
labels = ["A*", "Q-learning", "DQN"]

normal_means = [
    np.mean(astar_normal),
    np.mean(ql_normal),
    np.mean(dqn_normal)
]

peak_means = [
    np.mean(astar_peak),
    np.mean(ql_peak),
    np.mean(dqn_peak)
]

x = np.arange(len(labels))
width = 0.35

plt.figure()
plt.bar(x - width/2, normal_means, width, label="Normal")
plt.bar(x + width/2, peak_means, width, label="Peak")

plt.xticks(x, labels)
plt.ylabel("Average Reward")
plt.title("Normal vs Peak Comparison")
plt.legend()
plt.savefig("normal_vs_peak_bar.png")

# ===============================
# 5. 图4：DQN训练 + eval
# ===============================
plt.figure()
plt.plot(dqn_peak, label="Train Reward")
plt.plot(
    np.linspace(0, len(dqn_peak), len(dqn_eval_peak)),
    dqn_eval_peak,
    marker='o',
    linestyle='--',
    label="Eval Reward"
)
plt.title("DQN Training Curve (Peak)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig("dqn_training_curve.png")

print("✅ 所有图已生成！")