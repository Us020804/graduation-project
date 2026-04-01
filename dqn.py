import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        actions,
        lr=0.0005,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.98,
        batch_size=32,
        target_update=20,
        buffer_capacity=5000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = actions
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.learn_step_count = 0

    def normalize_state(self, state):
        state = np.array(state, dtype=np.float32)

        # 兼容旧版二维状态
        if len(state) == 2:
            x, y = state
            return np.array([
                x / 3000.0,
                y / 2400.0
            ], dtype=np.float32)

        # 兼容旧版5维状态
        elif len(state) == 5:
            x, y, dx, dy, covered = state
            return np.array([
                x / 3000.0,
                y / 2400.0,
                np.clip(dx / 1000.0, -1.0, 1.0),
                np.clip(dy / 1000.0, -1.0, 1.0),
                np.clip(covered / 10.0, 0.0, 1.0)
            ], dtype=np.float32)

        # 当前项目推荐使用的6维状态
        elif len(state) == 6:
            x, y, dx, dy, covered, vehicle_count = state
            return np.array([
                x / 3000.0,
                y / 2400.0,
                np.clip(dx / 1000.0, -1.0, 1.0),
                np.clip(dy / 1000.0, -1.0, 1.0),
                np.clip(covered / 10.0, 0.0, 1.0),
                np.clip(vehicle_count / 300.0, 0.0, 1.0)
            ], dtype=np.float32)

        else:
            raise ValueError(f"不支持的状态维度: {len(state)}")

    def choose_action(self, state):
        state = self.normalize_state(state)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.q_net(state_tensor)

            action_idx = int(torch.argmax(q_values, dim=1).item())

        return self.actions[action_idx], action_idx

    def store_transition(self, state, action_idx, reward, next_state, done):
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)
        self.replay_buffer.push(state, action_idx, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.learn_step_count += 1
        if self.learn_step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)