 Traffic Signal Optimization using Reinforcement Learning

This project implements an AI-based traffic signal control system using Reinforcement Learning (Deep Q-Learning).
The system learns optimal traffic signal timings based on traffic density to reduce average vehicle waiting time and improve traffic flow.

Technologies Used :
Python,
PyTorch,
Reinforcement Learning (DQN),
Matplotlib

Files Description:
- env.py – Traffic environment simulation
- dqn.py – Deep Q-Network model
- agent.py – Reinforcement learning agent
- train.py – Training the agent
- plot.py – Performance visualization

Output:
The AI-controlled signal shows reduced waiting time compared to a static timer-based signal.

How to Run:
```bash
python train.py
python plot.py
