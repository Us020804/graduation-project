from env import UAVEnv
from dqn import DQNAgent
import json
import torch

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


def evaluate_agent(agent, env, episodes=5):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    eval_rewards = []

    for _ in range(episodes):
        env.reset()
        state = build_dqn_state(env)
        total_reward = 0

        for _ in range(env.max_steps):
            action, action_idx = agent.choose_action(state)
            _, reward, done = env.step(action)
            next_state = build_dqn_state(env)

            total_reward += reward
            state = next_state

            if done:
                break

        eval_rewards.append(total_reward)

    agent.epsilon = old_epsilon
    return sum(eval_rewards) / len(eval_rewards)


env = UAVEnv(
    sumocfg_file="jnu_clean.sumocfg",
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
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.98,
    batch_size=32,
    target_update=20,
    buffer_capacity=5000
)

episodes = 200
best_eval_reward = float('-inf')
reward_history = []
eval_history = []

print("当前训练版本：DQN（有移动代价 + epsilon衰减）")

for episode in range(episodes):
    env.reset()
    state = build_dqn_state(env)
    total_reward = 0
    loss_history = []

    for step in range(env.max_steps):
        action, action_idx = agent.choose_action(state)
        _, reward, done = env.step(action)
        next_state = build_dqn_state(env)

        agent.store_transition(state, action_idx, reward, next_state, done)
        loss = agent.learn()

        if loss is not None:
            loss_history.append(loss)

        total_reward += reward
        state = next_state

        if episode == 0 and step < 10:
            print(
                f"  step={step+1}, action={action}, next_state={next_state}, "
                f"reward={reward:.2f}, loss={loss}"
            )

        if done:
            break

    agent.decay_epsilon()
    reward_history.append(total_reward)

    if (episode + 1) % 10 == 0:
        eval_reward = evaluate_agent(agent, env, episodes=5)
        eval_history.append(eval_reward)
        print(f"  >>> Eval reward after Episode {episode + 1}: {eval_reward:.2f}")

        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            torch.save(agent.q_net.state_dict(), "dqn_best.pth")
            print(f"  >>> 保存新最优模型，best eval reward = {best_eval_reward:.2f}")

    avg_loss = sum(loss_history) / len(loss_history) if loss_history else None
    if avg_loss is not None:
        print(
            f"Episode {episode + 1}: total reward = {total_reward:.2f}, "
            f"avg loss = {avg_loss:.4f}, epsilon = {agent.epsilon:.4f}"
        )
    else:
        print(
            f"Episode {episode + 1}: total reward = {total_reward:.2f}, "
            f"avg loss = None, epsilon = {agent.epsilon:.4f}"
        )

env.close_sumo()

print("DQN 训练完成")
print("经验池大小：", len(agent.replay_buffer))
print("每轮总奖励列表：")
print(reward_history)

with open("dqn_rewards.json", "w", encoding="utf-8") as f:
    json.dump(reward_history, f, ensure_ascii=False)

with open("dqn_eval_rewards.json", "w", encoding="utf-8") as f:
    json.dump(eval_history, f, ensure_ascii=False)

print("奖励结果已保存到 dqn_rewards.json")
print("评估结果已保存到 dqn_eval_rewards.json")
