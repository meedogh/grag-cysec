import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, joint_action, reward, next_state, done):
        self.buffer.append((state, joint_action, reward, next_state, done))


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
