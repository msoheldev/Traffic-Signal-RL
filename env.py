import numpy as np
import random

class TrafficEnv:
    def __init__(self):
        self.max_cars = 50
        self.reset()

    def reset(self):
        self.lanes = np.random.randint(5, 20, size=4)
        self.wait_time = np.zeros(4)
        self.current_green = random.randint(0, 3)
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.lanes[0], self.lanes[1],
            self.lanes[2], self.lanes[3],
            self.wait_time.sum()
        ], dtype=np.float32)

    def step(self, action):
        if action != self.current_green:
            self.current_green = action

        passed = min(5, self.lanes[self.current_green])
        self.lanes[self.current_green] -= passed

        self.lanes += np.random.randint(0, 3, size=4)
        self.lanes = np.clip(self.lanes, 0, self.max_cars)

        self.wait_time += self.lanes
        reward = -self.wait_time.sum()

        done = False
        return self._get_state(), reward, done