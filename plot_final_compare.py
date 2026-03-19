import matplotlib.pyplot as plt

# Q-learning 无移动代价
q_no_cost = [
    47, 47, 48, 48, 48, 47, 48, 48, 48, 48,
    48, 48, 49, 48, 48, 48, 48, 47, 48, 67,
    48, 48, 65, 47, 48, 48, 48, 50, 47, 48
]

# Q-learning 有移动代价
q_with_cost = [
    39.8, 40.0, 43.5, 43.9, 48.7, 39.1, 43.8, 44.6, 52.5, 43.5,
    43.7, 43.8, 43.6, 43.7, 43.4, 43.6, 43.8, 43.8, 43.6, 43.7,
    40.0, 47.4, 43.7, 43.6, 43.6, 45.5, 42.9, 43.2, 43.5, 43.8
]

# A* 无移动代价
astar_no_cost = [52.0] * 30

# A* 有移动代价
astar_with_cost = [50.1] * 30

episodes = list(range(1, 31))

plt.figure(figsize=(10, 6))

plt.plot(episodes, q_no_cost, marker='o', label='Q-learning (No Move Cost)')
plt.plot(episodes, q_with_cost, marker='s', label='Q-learning (With Move Cost)')
plt.plot(episodes, astar_no_cost, linestyle='--', linewidth=2, label='A* (No Move Cost)')
plt.plot(episodes, astar_with_cost, linestyle='-.', linewidth=2, label='A* (With Move Cost)')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Final Algorithm Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("平均奖励：")
print(f"Q-learning 无移动代价: {sum(q_no_cost)/len(q_no_cost):.2f}")
print(f"Q-learning 有移动代价: {sum(q_with_cost)/len(q_with_cost):.2f}")
print(f"A* 无移动代价: {sum(astar_no_cost)/len(astar_no_cost):.2f}")
print(f"A* 有移动代价: {sum(astar_with_cost)/len(astar_with_cost):.2f}")