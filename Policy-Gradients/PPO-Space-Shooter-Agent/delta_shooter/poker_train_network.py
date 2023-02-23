from typing import List, Tuple
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
    # at end run "tensorboard --logdir runs" in terminal to visualize

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
from opponent_policies import (feature_input, simple_policy, simple_policy_conservative, simple_policy_aggressive,
                                complex_policy, complex_policy_conservative, complex_policy_agressive,
                               all_in_policy, ultra_conservative_policy)
from opponent_pool import OpponentPool, Opponent
from action_tracker import ActionCountTracker
from game_mechanics.render import human_player

"""Note: most of the structure of these files came from Henry Pulver at Delta Academy"""

TEAM_NAME = "Your team Name"  # <---- Enter your team name here!
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

def rescale_reward_in_step_of_memory(step_of_memory: Tuple, reward_weighting: float,
                                     total_return: float) -> Tuple:
    """Alter rewards in an episode to normalize to 1,
    and weigh final reward stronger"""
    def rescale_reward(orig_reward, rescale_fact, total_return, total_norm, done):
        new_reward = orig_reward / total_norm
        new_total_return = total_return / total_norm
        new_reward = (new_reward * rescale_fact if not (done) else
                      new_reward * rescale_fact + (1.0 - rescale_fact) * new_total_return)
        return new_reward
    step_memory_list = list(step_of_memory)
    step_memory_list[3] = rescale_reward(step_memory_list[3], reward_weighting,
                                         total_return, 100, step_memory_list[4])
    return tuple(step_memory_list)
def rescale_reward_penalize_all_in(step_of_memory: Tuple, all_in_penalty: float) -> Tuple:
    """Alter rewards in an episode to normalize to 1,
    and weigh final reward stronger"""
    step_memory_list = list(step_of_memory)
    if step_memory_list[1] == 4: # we went all in
        step_memory_list[3] -= all_in_penalty 
    return tuple(step_memory_list)

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
            complex_policy,
            complex_policy_conservative,
            complex_policy_agressive,
            simple_policy,
            simple_policy_conservative,
            simple_policy_aggressive,
            choose_move_randomly,
            all_in_policy,
            ultra_conservative_policy
        ],
        exploration_const = 2.0 # explore very heavily to avoid training on one type
    )

    # Hyperparameters
    gamma = 0.98
    beta = 0.04
    beta_decay = 0.9995
    beta_min = 0.005 # won't let it fall below this

    reward_weighting = 0.1 # reweight all rewards by this factor, then add removed stuff to terminal
    reward_weighting_decay = 0.9995 # rescale by this factor each network update step
    penalty_all_in = 0.1 # makes the model more conservative

    min_batch_size = 512

    max_num_network_opponents = 10

    pol_lr = 0.00025
    val_lr = 0.0005

    num_episodes = 150_000

    eps_per_opponent_refresh = 100
    eps_per_debug_printing = 500
    eps_per_logging_slow = 5000
    eps_per_new_opponent = num_episodes // max_num_network_opponents

    n_inputs = 55
    n_actions = 5
    n_pol_neurons = 256
    n_val_neurons = 128

    # Policy net
    policy = nn.Sequential(
        nn.Linear(n_inputs, n_pol_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons, n_pol_neurons // 2),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons // 2, n_pol_neurons // 4),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons // 4, n_actions),
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
    val_optim = torch.optim.Adam(V.parameters(), lr=val_lr)
    val_loss_fn = nn.MSELoss()

    # Predefine gamma matrix to speed up returns calculation
    gamma_matrix_size = 540
    big_gamma_matrix = generate_big_gamma_matrix(gamma, gamma_matrix_size)

    past_returns = deque(maxlen=eps_per_debug_printing)
    past_num_steps = deque(maxlen=eps_per_debug_printing)
    action_tracker = ActionCountTracker()
    memory = []
    tbfast = SummaryWriter()
    tbslow = SummaryWriter()

    for ep_num in tqdm(range(num_episodes)):
        if ep_num % eps_per_opponent_refresh == eps_per_opponent_refresh - 1:
            opponent_pool.refresh_opponent_records()
        if ep_num % eps_per_new_opponent == eps_per_new_opponent - 1:
            checkpoint_model(policy, f"checkpoint_{ep_num + 1}.pt")
            opponent_pool.add_opponent(
                ChooseMoveCheckpoint(f"checkpoint_{ep_num + 1}.pt", choose_move))
        if ep_num % eps_per_debug_printing == eps_per_debug_printing - 1:
            print(f"\n     Visit counts: {opponent_pool.visit_counts}\n",
                  f"     Prob losses: {np.round(opponent_pool.prob_losses,3)}\n",
                  f"     Beta: {beta:0.4f}, Reward weighting: {reward_weighting:0.3f}")

        opponent_id = opponent_pool.select_opponent()
        ep_memory, total_return, n_steps, new_act = run_episode(policy, opponent_pool.get_opponent(opponent_id))
        opponent_pool.record_score(opponent_id, total_return // 100)
        action_tracker += new_act

        # Rescale rewards within episode to value final outcome more
        ep_memory = [rescale_reward_in_step_of_memory(step_of_memory, reward_weighting, total_return)
                     for step_of_memory in ep_memory]
        ep_memory = [rescale_reward_penalize_all_in(step_of_memory, penalty_all_in)
                     for step_of_memory in ep_memory]
        memory += ep_memory

        if len(memory) >= min_batch_size:
            states = torch.stack([item[0] for item in memory])
            actions = torch.tensor([item[1] for item in memory], requires_grad=False)
            legal_moves_masks = torch.stack([item[2] for item in memory])
            rewards = torch.tensor([item[3] for item in memory], requires_grad=False, dtype=torch.float32)
            is_dones = torch.tensor([item[4] for item in memory], requires_grad=False, dtype=torch.float32)

            # Decay hyperparameters
            reward_weighting *= reward_weighting_decay
            beta *= beta_decay
            if beta < beta_min:
                beta = beta_min

            # Returns calculation
            if len(memory) > gamma_matrix_size: # Expand the big gamma matrix if necessary
                big_gamma_matrix = generate_big_gamma_matrix(gamma, len(memory))
            ep_gam_matrix = big_gamma_matrix[: len(memory), : len(memory)] # Ttruncate for this batch size
            for term_ind in torch.squeeze(is_dones.nonzero(), dim=1): # Separate different episodes
                ep_gam_matrix[: term_ind + 1, term_ind + 1:] = 0 # kills upper right of (term_ind,term_ind)
            returns = torch.matmul(ep_gam_matrix, rewards)

            # Value update
            val_params_before_update = [weight.clone().detach() for name, weight in V.named_parameters()]
            val_loss = val_loss_fn(V(states).squeeze(1), returns)

            val_optim.zero_grad()
            val_loss.backward()
            val_optim.step()

            val_params_after_update = [weight.clone().detach() for name, weight in V.named_parameters()]

            # Policy update
            pol_params_before_update = [weight.clone().detach() for name, weight in policy.named_parameters()]
            action_dists = Categorical(forward_pass(states, policy, legal_moves_masks))
            pol_entropy = action_dists.entropy()
            action_logprobs = action_dists.log_prob(actions)

            with torch.no_grad():
                vals = V(states)

            pol_loss = -torch.mean(((returns - vals) * action_logprobs) + beta * pol_entropy)

            pol_optim.zero_grad()
            pol_loss.backward()
            pol_optim.step()

            pol_params_after_update = [weight.clone().detach() for name, weight in policy.named_parameters()]

            memory = []

            # Log tensorboard fast variables
            val_update_strengths = [((val_params_before_update[_i] - val_params_after_update[_i]).std() /
                                     (val_params_before_update[_i]).std()).log10().item()
                                    for _i in range(len(val_params_before_update))]
            pol_update_strengths = [((pol_params_before_update[_i] - pol_params_after_update[_i]).std() /
                                     (pol_params_before_update[_i]).std()).log10().item()
                                    for _i in range(len(pol_params_before_update))]
            for _ind, (name, weight) in enumerate(policy.named_parameters()):
                tbfast.add_scalar(f"Policy log_10(Delta of {name})", pol_update_strengths[_ind], ep_num)
            for _ind, (name, weight) in enumerate(V.named_parameters()):
                tbfast.add_scalar(f"Value log_10(Delta of {name})", val_update_strengths[_ind], ep_num)
            tbfast.add_scalar("Policy loss", pol_loss.item(), ep_num)
            tbfast.add_scalar("Value loss", val_loss.item(), ep_num)
            tbfast.add_scalar('avg_episode_length', sum(past_num_steps) / len(past_num_steps), ep_num)
            tbfast.add_scalar('avg_raw_reward', sum(past_returns) / eps_per_debug_printing, ep_num)
            tbfast.add_scalar("Beta", beta, ep_num)
            total_actions = sum(action_tracker.action_counts.values())
            action_probs = [action_tracker.action_counts[action]/total_actions for action in range(5)]
            for action in range(5):
                tbfast.add_scalar(f"Action {action} prob.", action_probs[action], ep_num)

        # Log tensorboard slow variables, histogram of layer outputs and weights/biases
        if ep_num % eps_per_logging_slow == eps_per_logging_slow - 1:
            with torch.no_grad():
                # Choose a network input tensor, and feed it through to observe activations
                data_to_feed = ep_memory[-1][0] # use a random recent terminal state to rec
                for layer_index, module in enumerate(policy.children()):
                    data_to_feed = module(data_to_feed)
                    tbslow.add_histogram(f"layer {layer_index} output", data_to_feed, ep_num)
                # Also record weights and activations
                for name, weight in policy.named_parameters():
                    tbslow.add_histogram(name, weight, ep_num)
                    tbslow.add_histogram(f'{name}.grad', weight.grad, ep_num)

        past_returns.append(int(total_return))
        past_num_steps.append(n_steps)

        if ep_num % eps_per_debug_printing == eps_per_debug_printing - 1:
            print("     ",action_tracker)
            action_tracker = ActionCountTracker() # reset action tracker

    tbfast.close()
    tbslow.close()

    return policy


def choose_move(state: State, neural_network: nn.Module) -> int:
    action_probs = forward_pass(
        preprocess_state(state), neural_network, legal_moves_to_mask(state.legal_actions)
    )
    return int(Categorical(action_probs).sample())


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("train()", "prof.prof")
    if True:
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