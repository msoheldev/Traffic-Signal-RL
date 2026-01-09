import random
import torch
import torch.optim as optim
from collections import deque
from dqn import DQN


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.FloatTensor(state)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])

            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.model(next_state))

            prediction = self.model(state)[action]
            loss = (prediction - target) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay