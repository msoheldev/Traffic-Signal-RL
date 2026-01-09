from env import TrafficEnv
from agent import DQNAgent
import torch

env = TrafficEnv()
agent = DQNAgent(state_size=5, action_size=4)

episodes = 300
scores = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for _ in range(50):
        action = agent.act(state)
        next_state, reward, _ = env.step(action)
        agent.remember(state, action, reward, next_state)
        agent.train()
        state = next_state
        total_reward += reward

    scores.append(total_reward)
    print(f"Episode {ep+1}, Reward: {total_reward}")

torch.save(agent.model.state_dict(), "traffic_dqn.pth")