import matplotlib.pyplot as plt

# 你的 reward_history
reward_history = [47, 47, 48, 48, 48, 47, 48, 48, 48, 48, 48, 48, 49, 48, 48, 48, 48, 47, 48, 67, 48, 48, 65, 47, 48, 48, 48, 50, 47, 48]

episodes = list(range(1, len(reward_history) + 1))

# 计算简单滑动平均
window = 5
moving_avg = []
for i in range(len(reward_history)):
    start = max(0, i - window + 1)
    avg = sum(reward_history[start:i + 1]) / len(reward_history[start:i + 1])
    moving_avg.append(avg)

plt.figure(figsize=(10, 5))
plt.plot(episodes, reward_history, marker='o', label='Episode Reward')
plt.plot(episodes, moving_avg, label='Moving Average (5)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Training Reward Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()