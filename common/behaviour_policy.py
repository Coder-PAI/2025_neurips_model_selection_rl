import random
from typing import Any

import numpy as np


class EpsilonGreedyPolicy:
    def __init__(
        self,
        policy: Any,
        epsilon: float,
        noise_scaler: float = 8.0,
    ):
        self.policy = policy
        self.epsilon = epsilon
        self.noise_scaler = noise_scaler

    def predict(self, states):
        # Epsilon-greedy exploration in the continuous case is to consider additive Gaussian noise
        batch_action = self.policy.predict(states)
        batch_noise = np.array(
            [
                (
                    self.noise_scaler
                    * np.random.normal(0, 1, size=(batch_action.shape[1],))
                    if random.random() <= self.epsilon
                    else np.zeros((batch_action.shape[1],))
                )
                for _ in range(len(states))
            ],
            dtype=np.float32,
        )
        return batch_action + batch_noise
