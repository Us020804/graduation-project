from env import UAVEnv
from astar import AStarPlanner
import json

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

planner = AStarPlanner(
    x_min=0,
    x_max=2938,
    y_min=0,
    y_max=2318,
    step_size=20
)

episodes = 30
reward_history = []

print("当前测试版本：A* 多轮基线")

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    start_covered = env.count_covered_vehicles()
    print(f"Episode {episode + 1} 初始状态: {state}, 初始覆盖车辆数: {start_covered}")

    for step in range(env.max_steps):
        vehicle_positions = env.get_vehicle_positions()
        uav_pos = env.get_state()

        action, target_pos, path = planner.choose_action(uav_pos, vehicle_positions)
        next_state, reward, done = env.step(action)

        total_reward += reward

        if episode == 0 and step < 10:
            print(
                f"  step={step+1}, uav_pos={uav_pos}, target={target_pos}, "
                f"action={action}, reward={reward}"
            )

        if done:
            break

    reward_history.append(total_reward)
    print(f"Episode {episode + 1}: total reward = {total_reward:.2f}")

env.close_sumo()

print("A* 测试完成")
print("每轮总奖励列表：")
print(reward_history)

with open("astar_peak_canteen_rewards.json", "w", encoding="utf-8") as f:
    json.dump(reward_history, f, ensure_ascii=False)

print("奖励结果已保存到 astar_peak_canteen_rewards.json")
