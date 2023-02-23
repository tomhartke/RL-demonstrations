import torch
from torch import nn
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# at end run "tensorboard --logdir runs" in terminal to visualize

from opponent_policies import policy_turn_and_shoot
from opponent_policies import preprocess_state_old, choose_move_old

# from check_submission import check_submission
from game_mechanics import (
    ChooseMoveCheckpoint,
    ShooterEnv,
    checkpoint_model,
    choose_move_randomly,
    load_checkpoint,
    human_player,
    load_network,
    play_shooter,
    save_network
)

from opponent_policies import get_extra_features_from_state

# gpu = torch.device("cuda:0")
gpu = torch.device("cpu")
cpu = torch.device("cpu")

def preprocess_state(state: torch.Tensor,
                     is_learning: bool = False) -> torch.Tensor:
    """
    Process input state to feed to nn based on environment
    """
    if is_learning:
        state.to(gpu)
    else:
        state.to(cpu)

    if len(state.shape) > 1:  # we have a batch of inputs, have to process that way.
        # Could probably do this more efficiently
        new_nn_input = torch.stack([get_extra_features_from_state(substate)
                                    for substate in state])
    else: # single state
        new_nn_input = get_extra_features_from_state(state)

    return new_nn_input

def choose_move(
        state: torch.Tensor,
        neural_network: nn.Module,
) -> int:  # <--------------- Please do not change these arguments!

    action_probs = neural_network(preprocess_state(state))

    deterministic_agent = False
    semi_deterministic_agent = True
    if deterministic_agent:
        return int(torch.argmax(action_probs).item())
    elif semi_deterministic_agent:
        # this basically just adjusts the temperature of the softmax to be not equal to 1.
        action_probs_renormed = (action_probs ** 4)/torch.sum(action_probs ** 4)
        # this moves things further apart, effectively decreasing temperature of softmax
        # probs 0.6/0.4 goes to 0.8/0.2, and 0.7/0.3 goes to 0.95 chance of action, anything above is deterministic
        return int(Categorical(action_probs_renormed).sample())
    else: # nondeterministic agent
        return int(Categorical(action_probs).sample())

if __name__ == "__main__":
    # import cProfile
    # cProfile.run("train()", "prof.prof")

    # Load checkpointed network.pt
    checkpoint_name = 'Network_final.pt'  #'Network_barrier_trained_full_size_very_very_good.pt' # 'checkpoint_70000.pt'
    my_network = load_checkpoint(checkpoint_name)
    my_network.eval()

    def choose_move_no_network(state) -> int:
        """The arguments in play_pong_game() require functions that only take the state as
        input.

        This converts choose_move() to that format.
        """
        return choose_move(state, neural_network=my_network)

    # Removing the barriers and making the game half-sized will make it easier to train!
    include_barriers = True
    half_game_size = False

    # The code below plays a single game against your bot.
    # You play as the pink ship
    for _ in range(10):
        play_shooter(
            your_choose_move= choose_move_no_network,  # human_player,
            opponent_choose_move= choose_move_no_network,
            game_speed_multiplier=1.0,
            render=True,
            include_barriers=include_barriers,
            half_game_size=half_game_size,
        )
