import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from mcts_class import MCTS

HERE = Path(__file__).parent.resolve()


class ChooseMoveCheckpoint:
    def __init__(self, checkpoint_name: str, choose_move: Callable, my_mcts: MCTS, training_num_rollouts: float):
        self.neural_network = copy.deepcopy(load_checkpoint(checkpoint_name))
        self._choose_move = choose_move
        self.my_mcts = my_mcts
        self.training_num_rollouts = training_num_rollouts

    def __call__(self, state: np.ndarray) -> int:
        return self._choose_move(state, network=self.neural_network, mcts=self.my_mcts, num_rollouts=self.training_num_rollouts)


def checkpoint_model(model: nn.Module, checkpoint_name: str) -> None:
    torch.save(model, HERE / checkpoint_name)


def load_checkpoint(checkpoint_name: str) -> nn.Module:
    return torch.load(HERE / checkpoint_name)
