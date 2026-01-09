import matplotlib.pyplot as plt
import numpy as np

ai = np.random.randint(200, 600, 20)
static = np.random.randint(600, 1200, 20)

plt.plot(ai, label="AI Controlled Signal")
plt.plot(static, label="Static Timer Signal")
plt.xlabel("Traffic Density")
plt.ylabel("Average Waiting Time")
plt.legend()
plt.show()