import torch
from env import UAVEnv
from dqn import DQNAgent

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']

def build_dqn_state(env):
    uav_x, uav_y = env.get_state()
    vehicle_positions = env.get_vehicle_positions()
    covered_count = env.count_covered_vehicles()
    vehicle_count = len(vehicle_positions)

    if vehicle_positions:
        xs = [x for _, x, _ in vehicle_positions]
        ys = [y for _, _, y in vehicle_positions]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
    else:
        center_x, center_y = uav_x, uav_y

    dx = center_x - uav_x
    dy = center_y - uav_y

    return (uav_x, uav_y, dx, dy, covered_count, vehicle_count)

env = UAVEnv(
    sumocfg_file="jnu_clean.sumocfg",
    uav_start=(1600, 1600),
    uav_radius=200,
    step_size=20,
    x_min=0,
    x_max=5000,
    y_min=0,
    y_max=5000,
    max_steps=20,
    gui=False,
    move_cost=0.1
)

agent = DQNAgent(
    state_dim=6,
    action_dim=len(ACTIONS),
    actions=ACTIONS,
    lr=0.0005,
    gamma=0.9,
    epsilon_start=0.0,   # 评估时不探索
    epsilon_min=0.0,
    epsilon_decay=1.0,
    batch_size=32,
    target_update=20,
    buffer_capacity=5000
)

# 加载最优模型
agent.q_net.load_state_dict(torch.load("dqn_best.pth", map_location="cpu"))
agent.target_net.load_state_dict(agent.q_net.state_dict())
agent.q_net.eval()
agent.target_net.eval()

episodes = 30
reward_history = []

print("当前评估版本：DQN Greedy Evaluation (epsilon=0)")

for episode in range(episodes):
    env.reset()
    state = build_dqn_state(env)
    total_reward = 0

    for step in range(env.max_steps):
        action, action_idx = agent.choose_action(state)
        _, reward, done = env.step(action)
        next_state = build_dqn_state(env)

        total_reward += reward
        state = next_state

        if episode == 0 and step < 10:
            print(f"  step={step+1}, action={action}, next_state={next_state}, reward={reward:.2f}")

        if done:
            break

    reward_history.append(total_reward)
    print(f"Episode {episode + 1}: total reward = {total_reward:.2f}")

env.close_sumo()

avg_reward = sum(reward_history) / len(reward_history)
print("DQN 评估完成")
print("每轮总奖励列表：")
print(reward_history)
print(f"平均奖励: {avg_reward:.2f}")