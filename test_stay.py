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

total_reward = 0

for step in range(20):
    next_state, reward, done = env.step('STAY')
    total_reward += reward
    print(f"第 {step+1} 步, 状态={next_state}, 奖励={reward}, 累计奖励={total_reward}")
    if done:
        break

env.close_sumo()
print("总奖励：", total_reward)