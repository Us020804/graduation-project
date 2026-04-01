import pickle
import random


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

    def choose_action(self, state):
        self.check_state_exist(state)

        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            state_actions = self.q_table[state]
            max_q = max(state_actions.values())
            best_actions = [a for a, q in state_actions.items() if q == max_q]
            action = random.choice(best_actions)

        return action

    def choose_best_action(self, state):
        self.check_state_exist(state)
        state_actions = self.q_table[state]
        max_q = max(state_actions.values())
        best_actions = [a for a, q in state_actions.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        self.check_state_exist(state)
        self.check_state_exist(next_state)

        q_predict = self.q_table[state][action]

        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * max(self.q_table[next_state].values())

        self.q_table[state][action] += self.alpha * (q_target - q_predict)

    def save_q_table(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filepath):
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
