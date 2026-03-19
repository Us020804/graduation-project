import matplotlib.pyplot as plt

# Q-learning：无移动代价
q_no_cost = [
    47, 47, 48, 48, 48, 47, 48, 48, 48, 48,
    48, 48, 49, 48, 48, 48, 48, 47, 48, 67,
    48, 48, 65, 47, 48, 48, 48, 50, 47, 48
]

# Q-learning：有移动代价
q_with_cost = [
    39.8, 40.0, 43.5, 43.9, 48.7, 39.1, 43.8, 44.6, 52.5, 43.5,
    43.7, 43.8, 43.6, 43.7, 43.4, 43.6, 43.8, 43.8, 43.6, 43.7,
    40.0, 47.4, 43.7, 43.6, 43.6, 45.5, 42.9, 43.2, 43.5, 43.8
]

# A* 基线
astar_rewards = [50.1] * 30

episodes = list(range(1, 31))

def moving_average(data, window=5):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        avg = sum(data[start:i+1]) / len(data[start:i+1])
        result.append(avg)
    return result

q_no_cost_ma = moving_average(q_no_cost, 5)
q_with_cost_ma = moving_average(q_with_cost, 5)

plt.figure(figsize=(10, 6))

plt.plot(episodes, q_no_cost, marker='o', label='Q-learning (No Move Cost)')
plt.plot(episodes, q_with_cost, marker='s', label='Q-learning (With Move Cost)')
plt.plot(episodes, astar_rewards, linestyle='--', linewidth=2, label='A* Baseline')

plt.plot(episodes, q_no_cost_ma, label='Q-learning No Cost MA(5)')
plt.plot(episodes, q_with_cost_ma, label='Q-learning With Cost MA(5)')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Algorithm Performance Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Q-learning 无移动代价平均奖励：", round(sum(q_no_cost) / len(q_no_cost), 2))
print("Q-learning 有移动代价平均奖励：", round(sum(q_with_cost) / len(q_with_cost), 2))
print("A* 平均奖励：", round(sum(astar_rewards) / len(astar_rewards), 2))