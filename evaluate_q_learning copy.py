import json

from env import UAVEnv
from q_learning import QLearningAgent


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


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

agent = QLearningAgent(
    actions=ACTIONS,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.0
)

agent.load_q_table("q_learning_peak_canteen_q_table.pkl")

episodes = 30
reward_history = []

print("Current evaluation: Q-learning greedy policy on peak canteen traffic")

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(env.max_steps):
        action = agent.choose_best_action(state)
        next_state, reward, done = env.step(action)

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
print("Peak canteen Q-learning evaluation completed")
print(reward_history)
print(f"Average reward: {avg_reward:.2f}")

with open("q_learning_peak_canteen_eval_rewards.json", "w", encoding="utf-8") as f:
    json.dump(reward_history, f, ensure_ascii=False)

print("Saved q_learning_peak_canteen_eval_rewards.json")
