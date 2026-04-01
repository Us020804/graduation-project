from env import UAVEnv
from q_learning import QLearningAgent
import json

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']

env = UAVEnv(
    sumocfg_file="jnu_clean.sumocfg",
    uav_start=(1600, 1600),
    uav_radius=200,
    step_size=20,
    x_min=0,
    x_max=2938,
    y_min=0,
    y_max=2318,
    max_steps=20,
    gui=False,
    move_cost=0.1
)

agent = QLearningAgent(
    actions=ACTIONS,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1
)

episodes = 30
reward_history = []

print("当前训练版本：包含 STAY 动作，并记录每轮总奖励")

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    start_covered = env.count_covered_vehicles()
    print(f"Episode {episode + 1} 初始状态: {state}, 初始覆盖车辆数: {start_covered}")

    for step in range(env.max_steps):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.learn(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if episode == 0 and step < 10:
            print(f"  step={step+1}, action={action}, next_state={next_state}, reward={reward}, done={done}")

        if done:
            break

    reward_history.append(total_reward)
    print(f"Episode {episode + 1}: total reward = {total_reward:.2f}")

env.close_sumo()

print("训练完成")
print("Q表状态数量：", len(agent.q_table))
print("每轮总奖励列表：")
print(reward_history)

with open("q_learning_rewards.json", "w", encoding="utf-8") as f:
    json.dump(reward_history, f, ensure_ascii=False)

print("奖励结果已保存到 q_learning_rewards.json")