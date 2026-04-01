from env import UAVEnv

env = UAVEnv("jnu_clean.sumocfg", gui=False, move_cost=0.1)

state = env.reset()
print("初始状态:", state)

actions = ["STAY", "RIGHT", "UP", "LEFT", "DOWN"]

for action in actions:
    next_state, reward, done = env.step(action)
    print(f"动作: {action}, next_state={next_state}, reward={reward}, done={done}")

env.close_sumo()