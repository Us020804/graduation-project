import matplotlib.pyplot as plt

# 不加移动代价
rewards_no_cost = [
    47, 47, 48, 48, 48, 47, 48, 48, 48, 48,
    48, 48, 49, 48, 48, 48, 48, 47, 48, 67,
    48, 48, 65, 47, 48, 48, 48, 50, 47, 48
]

# 加移动代价
rewards_with_cost = [
    39.8, 40.0, 43.5, 43.9, 48.7, 39.1, 43.8, 44.6, 52.5, 43.5,
    43.7, 43.8, 43.6, 43.7, 43.4, 43.6, 43.8, 43.8, 43.6, 43.7,
    40.0, 47.4, 43.7, 43.6, 43.6, 45.5, 42.9, 43.2, 43.5, 43.8
]

episodes = list(range(1, 31))

def moving_average(data, window=5):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        avg = sum(data[start:i+1]) / len(data[start:i+1])
        result.append(avg)
    return result

ma_no_cost = moving_average(rewards_no_cost, window=5)
ma_with_cost = moving_average(rewards_with_cost, window=5)

plt.figure(figsize=(10, 5))

plt.plot(episodes, rewards_no_cost, marker='o', label='No Move Cost')
plt.plot(episodes, rewards_with_cost, marker='s', label='With Move Cost')

plt.plot(episodes, ma_no_cost, label='No Cost MA(5)')
plt.plot(episodes, ma_with_cost, label='With Cost MA(5)')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Reward Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()