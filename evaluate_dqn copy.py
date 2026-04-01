import json

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
    sumocfg_file="jnu_peak_canteen.sumocfg",
    uav_start=(1600, 1600),
    uav_radius=200,
    step_size=20,
    x_min=0,
    x_max=2938,
    y_min=0,
    y_max=2318,
    max_steps=50,
    gui=False,
    move_cost=0.1
)

agent = DQNAgent(
    state_dim=6,
    action_dim=len(ACTIONS),
    actions=ACTIONS,
    lr=0.0005,
    gamma=0.9,
    epsilon_start=0.0,
    epsilon_min=0.0,
    epsilon_decay=1.0,
    batch_size=32,
    target_update=20,
    buffer_capacity=5000
)

agent.q_net.load_state_dict(torch.load("dqn_peak_canteen_best.pth", map_location="cpu"))
agent.target_net.load_state_dict(agent.q_net.state_dict())
agent.q_net.eval()
agent.target_net.eval()

episodes = 30
reward_history = []

print("Current evaluation: DQN greedy policy on peak canteen traffic")

for episode in range(episodes):
    env.reset()
    state = build_dqn_state(env)
    total_reward = 0

    for step in range(env.max_steps):
        action, _ = agent.choose_action(state)
        _, reward, done = env.step(action)
        next_state = build_dqn_state(env)

        total_reward += reward
        state = next_state

        if episode == 0 and step < 10:
            print(
                f"  step={step + 1}, action={action}, "
                f"next_state={next_state}, reward={reward:.2f}"
            )

        if done:
            break

    reward_history.append(total_reward)
    print(f"Episode {episode + 1}: total reward = {total_reward:.2f}")

env.close_sumo()

avg_reward = sum(reward_history) / len(reward_history)
print("Peak canteen DQN evaluation completed")
print(reward_history)
print(f"Average reward: {avg_reward:.2f}")

with open("dqn_peak_canteen_eval_rewards.json", "w", encoding="utf-8") as f:
    json.dump(reward_history, f, ensure_ascii=False)

print("Saved dqn_peak_canteen_eval_rewards.json")
