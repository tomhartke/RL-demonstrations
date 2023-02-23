from typing import List, Tuple

import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from collections import deque

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
    # at end run "tensorboard --logdir runs" in terminal to visualize (then go to suggested website)

from game_mechanics import (
    PokerEnv,
    choose_move_randomly,
    save_network,
    play_poker,
    State,
    load_network,
    load_checkpoint,
    ChooseMoveCheckpoint,
    checkpoint_model,
)
from opponent_policies import (feature_input, simple_policy, simple_policy_conservative, simple_policy_aggressive,
                                complex_policy, complex_policy_conservative, complex_policy_agressive)
from opponent_pool import OpponentPool, Opponent
from action_tracker import ActionCountTracker
from game_mechanics.render import human_player

TEAM_NAME = "Henry"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

# gpu = torch.device("cuda:0")
gpu = torch.device("cpu")
cpu = torch.device("cpu")


def forward_pass(
    processed_state: torch.Tensor, policy_net: nn.Module, legal_mask: torch.Tensor
) -> torch.Tensor:
    # TODO: Test legal moves masking on a batch
    action_preferences = policy_net(processed_state)
    assert (
        action_preferences.shape == legal_mask.shape
    ), f"{action_preferences.shape} != {legal_mask.shape}"
    # Compute the masked softmax of the action preferences
    intermediate = action_preferences * legal_mask  # [0.5, 0.999, 0]
    result = torch.nn.functional.softmax(
        intermediate, -1
    )  # [0.3, 0.5, 0.2] <- nonzero prob of invalid
    result = result * legal_mask  # [0.3, 0.5, 0] <- set to zero, not normalised
    # Add a tiny float on to prevent 0s and hence NaNs going forward
    return result / (torch.sum(result, dim=-1) + 1e-13).unsqueeze(
        -1
    )  # [0.35, 0.65, 0] < renormalised


def legal_moves_to_mask(legal_moves: List, is_learning: bool = False) -> torch.Tensor:
    legal_moves_mask = torch.zeros(5, device=gpu if is_learning else cpu)
    legal_moves_mask[legal_moves] = 1
    return legal_moves_mask


def preprocess_state(state: State, is_learning: bool = False) -> torch.Tensor:
    return feature_input(state, gpu if is_learning else cpu)

def choose_move(state: State, neural_network: nn.Module) -> int:
    action_probs = forward_pass(
        preprocess_state(state), neural_network, legal_moves_to_mask(state.legal_actions)
    )
    return int(Categorical(action_probs).sample())


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("train()", "prof.prof")

    # Load checkpointed network.pt
    checkpoint_name = 'Jarvis_network_11_24.pt' # 'checkpoint_70000.pt'
    my_network = load_checkpoint(checkpoint_name)
    my_network.eval()

    def choose_move_no_network(state) -> int:
        """The arguments in play_pong_game() require functions that only take the state as
        input.

        This converts choose_move() to that format.
        """
        return choose_move(state, neural_network=my_network)

    play_poker(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=0.5,
        render=True,
    )