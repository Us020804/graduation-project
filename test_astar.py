from env import UAVEnv
from astar import AStarPlanner

# 创建环境
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
    gui=True
)

# 创建 A* 规划器
planner = AStarPlanner(
    x_min=0,
    x_max=5000,
    y_min=0,
    y_max=5000,
    step_size=20
)

# 启动环境
state = env.reset()
total_reward = 0

print("开始测试 A* 基线")
print("=" * 40)

for step in range(env.max_steps):
    vehicle_positions = env.get_vehicle_positions()
    uav_pos = env.get_state()

    action, target_pos, path = planner.choose_action(uav_pos, vehicle_positions)
    next_state, reward, done = env.step(action)

    total_reward += reward

    print(f"第 {step+1} 步")
    print("当前无人机位置:", uav_pos)
    print("当前车辆数:", len(vehicle_positions))
    print("目标中心点:", target_pos)
    print("A* 选择动作:", action)
    print("奖励:", reward)
    print("累计奖励:", total_reward)
    print("-" * 40)

    if done:
        break

env.close_sumo()
print("A* 测试结束，总奖励：", total_reward)