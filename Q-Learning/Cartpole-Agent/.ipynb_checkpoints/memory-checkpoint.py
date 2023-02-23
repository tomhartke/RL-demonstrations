import torch
import numpy as np
from typing import Tuple


class ExperienceReplayMemory:
    def __init__(self, max_length: int):
        self.states = []
        self.actions = []
        self.rewards = []
        self.successor_states = []
        self.is_terminal = []
        self.max_length = max_length
        self._all_lists = [
            self.states,
            self.actions,
            self.rewards,
            self.successor_states,
            self.is_terminal,
        ]

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        successor_state: np.ndarray,
        is_terminal: bool,
    ):
        for l, item in zip(
            self._all_lists, [state, action, reward, successor_state, is_terminal]
        ):
            dtype = (
                bool
                if isinstance(item, bool)
                else int
                if isinstance(item, int)
                else torch.float32
            )
            l.append(torch.tensor(item, dtype=dtype))
            # If max length, remove the oldest states
            if len(l) > self.max_length:
                l.pop(0)

#     def sample(self, size: int) -> Tuple[torch.Tensor, ...]:
#         size = min(size, len(self.rewards))
#         idxs = np.random.choice(
#             list(range(len(self.rewards))), replace=False, size=size
#         )
#         return tuple(torch.stack([l[idx] for idx in idxs]) for l in self._all_lists)

    def sample(self, size: int) -> Tuple[torch.Tensor, ...]:
        size = min(size, len(self.rewards))
        idxs = np.random.choice(
            list(range(len(self.rewards))), replace=False, size=size
        )
        return tuple(torch.stack([l[idx] for idx in idxs]) for l in self._all_lists)
