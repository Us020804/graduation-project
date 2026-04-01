from dqn import DQNAgent

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


agent = DQNAgent(
    state_dim=2,
    action_dim=len(ACTIONS),
    actions=ACTIONS
)

state = (1600, 1600)

action, action_idx = agent.choose_action(state)
print("选择的动作：", action)
print("动作索引：", action_idx)

next_state = (1620, 1600)
reward = 2.0
done = False

agent.store_transition(state, action_idx, reward, next_state, done)
print("当前经验池大小：", len(agent.replay_buffer))

loss = agent.learn()
print("当前 loss：", loss)