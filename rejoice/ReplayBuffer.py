# Named tuple for storing experience steps gathered in training
from collections import OrderedDict, deque, namedtuple
from torch.utils.data.dataset import IterableDataset
from typing import List, Tuple
import numpy as np
import torch_geometric as pyg
import torch

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
        return (
            list(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(next_states),
        )


def exp_collate(batch):
    columns = list(zip(*batch))
    states = pyg.data.Batch.from_data_list(columns[0])
    next_states = pyg.data.Batch.from_data_list(columns[4])
    actions = torch.tensor(columns[1], dtype=torch.int64)
    rewards = np.array(columns[2], dtype=float)
    dones = np.array(columns[3], dtype=bool)

    return states, actions, rewards, dones, next_states


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
