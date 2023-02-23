from typing import List

import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from collections import deque

from tqdm import tqdm

from game_mechanics import (
    PokerEnv,
    choose_move_randomly,
    save_network,
    play_poker,
    State,
    load_network,
    ChooseMoveCheckpoint,
    checkpoint_model,
)
from opponent_policies import feature_input, simple_policy, simple_policy_conservative, simple_policy_aggressive
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


def generate_big_gamma_matrix(gamma: float, n: int) -> torch.Tensor:
    """
    Generates an upper triangular matrix of shape (n, n) where the diagonals
     are all gamma ^ n.

    This is used to efficiently compute the discounted returns for each step
     in an episode.
    """
    matrix = torch.zeros((n, n), device=gpu)
    for i in range(n):
        # This does the downward diagonals:
        # [g g^2 g^3 g^4]
        # [0  g  g^2 g^3]
        # [0  0   g  g^2]
        # [0  0   0   g ]
        matrix[range(n - i), range(i, n)] = gamma**i
    return matrix


def run_episode(policy: nn.Module, opponent: Opponent):
    env = PokerEnv(
        opponent_choose_move=opponent,
        render=False,
        verbose=False,
    )
    action_tracker = ActionCountTracker()
    memory = []
    total_return, n_steps = 0, 0

    state, reward, done, info = env.reset()
    total_return += reward
    nn_input = feature_input(state)

    while not done:
        prev_nn_input = nn_input
        legal_moves_mask = legal_moves_to_mask(state.legal_actions, True)
        action_probs = Categorical(forward_pass(nn_input, policy, legal_moves_mask))
        action = action_probs.sample()
        action_tracker.track(action.item())
        # print(f"State: {state}\nYour move:", MOVE_MAP[action.item()])
        state, reward, done, info = env.step(action.item())
        nn_input = preprocess_state(state, True)
        memory.append((prev_nn_input, action, legal_moves_mask, reward, done))
        total_return += reward
        n_steps += 1
    # print("Episode completed, total return:", total_return, "n_steps:", n_steps)
    return memory, total_return, n_steps, action_tracker


def train() -> nn.Module:
    # See docstring in opponent_pool.py for more info on the opponent pool
    opponent_pool = OpponentPool(
        [
            choose_move_randomly,
            simple_policy,
            simple_policy_conservative,
            simple_policy_aggressive,
        ]
    )
    n_inputs = 32
    n_actions = 5

    # Hyperparameters
    gamma = 0.98
    beta = 0.01

    # What is this? Since we want our agent to play to win the matchup (not just the hand)
    #  we want the reward structure to reflect this. I didn't remove the reward from winning
    #  a hand entirely, instead weighted it as 90% on the final matchup outcome and 10% on
    #  the hand outcome. The decay is multiplied by the reward_weighting each game.
    reward_weighting = 0.1
    reward_weighting_decay = 0.99997

    pol_lr = 0.002
    n_pol_neurons = 128

    val_lr = 0.01
    n_val_neurons = 128

    num_episodes = 100_000
    eps_per_debug = 500
    min_batch_size = 512

    # I found that the machine ran out of RAM beyond 10 old copies of the network in RAM!
    MAX_NUM_NETWORK_OPPONENTS = 10
    eps_per_new_opponent = num_episodes // MAX_NUM_NETWORK_OPPONENTS

    policy = nn.Sequential(
        nn.Linear(n_inputs, n_pol_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons, n_pol_neurons // 2),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons // 2, n_actions),
    )

    pol_optim = torch.optim.Adam(policy.parameters(), lr=pol_lr)

    # Value net
    V = nn.Sequential(
        nn.Linear(n_inputs, n_val_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons, n_val_neurons // 2),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons // 2, n_val_neurons // 2),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons // 2, 1),
        nn.Tanh(),
    )

    # This is simply to speed up the gamma calculation - use a big matrix multiplication
    #  instead of a loop
    gamma_matrix_size = 540
    big_gamma_matrix = generate_big_gamma_matrix(gamma, gamma_matrix_size)

    val_optim = torch.optim.Adam(V.parameters(), lr=val_lr)
    val_loss_fn = nn.MSELoss()

    past_returns = deque(maxlen=eps_per_debug)
    past_num_steps = deque(maxlen=eps_per_debug)
    action_tracker = ActionCountTracker()
    memory = []

    for ep_num in tqdm(range(num_episodes)):
        if ep_num % 100 == 99:
            opponent_pool.refresh_opponent_records()
            if ep_num % 500 == 499:
                print(
                    "Refreshing opponent records\n",
                    f"Visit counts: {opponent_pool.visit_counts}\n",
                    f"Loss counts: {opponent_pool.loss_counts}\n",
                    f"Prob losses: {np.round(opponent_pool.prob_losses,2)}\n",
                )
        if ep_num % eps_per_new_opponent == eps_per_new_opponent - 1:
            checkpoint_model(policy, f"checkpoint_{ep_num + 1}.pt")
            opponent_pool.add_opponent(
                ChooseMoveCheckpoint(f"checkpoint_{ep_num + 1}.pt", choose_move)
            )

        opponent_id = opponent_pool.select_opponent()
        ep_memory, total_return, n_steps, new_act = run_episode(policy, opponent_pool.get_opponent(opponent_id))
        opponent_pool.record_score(opponent_id, total_return // 100)
        action_tracker += new_act
        memory += ep_memory

        if len(memory) >= min_batch_size:
            states = torch.stack([item[0] for item in memory])
            actions = torch.tensor([item[1] for item in memory], requires_grad=False)
            legal_moves_masks = torch.stack([item[2] for item in memory])
            rewards = torch.tensor([item[3] for item in memory], requires_grad=False, dtype=torch.float32) / 100
            is_dones = torch.tensor([item[4] for item in memory], requires_grad=False, dtype=torch.float32)

            rewards *= reward_weighting
            rewards[-1] += (1 - reward_weighting) * total_return / 100
            reward_weighting *= reward_weighting_decay

            if len(memory) > gamma_matrix_size:
                # Expand the big gamma matrix if you need to!
                big_gamma_matrix = generate_big_gamma_matrix(gamma, len(memory))

            # Value update
            ep_gam_matrix = big_gamma_matrix[: len(memory), : len(memory)]
            for terminal_index in torch.squeeze(is_dones.nonzero(), dim=1):
                ep_gam_matrix[: terminal_index + 1, terminal_index + 1:] = 0

            returns = torch.matmul(ep_gam_matrix, rewards)
            # print("rewards", rewards)
            # print("returns", returns[:10])
            # print("V(states)", V(states).squeeze()[:10])

            val_loss = val_loss_fn(V(states).squeeze(1), returns)

            val_optim.zero_grad()
            val_loss.backward()
            val_optim.step()

            # Policy update
            action_dists = Categorical(forward_pass(states, policy, legal_moves_masks))
            pol_entropy = action_dists.entropy()
            action_logprobs = action_dists.log_prob(actions)

            with torch.no_grad():
                vals = V(states)

            pol_loss = -torch.mean(((returns - vals) * action_logprobs) + beta * pol_entropy)

            pol_optim.zero_grad()
            pol_loss.backward()
            pol_optim.step()
            memory = []

        past_returns.append(int(total_return))
        past_num_steps.append(n_steps)

        if ep_num % eps_per_debug == eps_per_debug - 1:
            print(
                "Steps:",
                sum(past_num_steps) / len(past_num_steps),
                f"Moving avg: {round(sum(past_returns) / eps_per_debug, 1)}\t",
                action_tracker,
                )
            action_tracker = ActionCountTracker()

    return policy


def choose_move(state: State, neural_network: nn.Module) -> int:
    action_probs = forward_pass(
        preprocess_state(state), neural_network, legal_moves_to_mask(state.legal_actions)
    )
    return int(Categorical(action_probs).sample())


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("train()", "prof.prof")
    my_network = train()
    save_network(my_network, TEAM_NAME)
    my_network = load_network(TEAM_NAME)

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