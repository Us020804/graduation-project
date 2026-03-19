from env import UAVEnv

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

env.start_sumo()
env.reset_uav()

for step in range(20):
    env.simulation_step()

    uav_pos = env.get_state()
    vehicle_positions = env.get_vehicle_positions()
    covered_count = env.count_covered_vehicles()

    print(f"第 {step + 1} 步")
    print("当前无人机位置：", uav_pos)
    print("当前车辆总数：", len(vehicle_positions))
    print("当前覆盖车辆数：", covered_count)

    if vehicle_positions:
        xs = [x for _, x, _ in vehicle_positions]
        ys = [y for _, _, y in vehicle_positions]

        print("车辆x范围：", (min(xs), max(xs)))
        print("车辆y范围：", (min(ys), max(ys)))
        print("前3辆车的位置：", vehicle_positions[:3])

    print("-" * 30)

env.close_sumo()