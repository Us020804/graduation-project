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

    def learn(self, state, action, reward, next_state):
        self.check_state_exist(state)
        self.check_state_exist(next_state)

        q_predict = self.q_table[state][action]
        q_target = reward + self.gamma * max(self.q_table[next_state].values())

        self.q_table[state][action] += self.alpha * (q_target - q_predict)