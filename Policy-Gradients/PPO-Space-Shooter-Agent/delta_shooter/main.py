import torch
from torch import nn
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# at end run "tensorboard --logdir runs" in terminal to visualize

# from check_submission import check_submission
from game_mechanics import (
    ChooseMoveCheckpoint,
    ShooterEnv,
    checkpoint_model,
    choose_move_randomly,
    human_player,
    load_network,
    play_shooter,
    save_network,
)

from opponent_policies import (policy_turn_and_shoot, policy_turn_and_shoot_more_random,
                               get_extra_features_from_state, choose_move_old)
from opponent_pool import OpponentPool, Opponent
from action_tracker import ActionCountTracker
from utils_gae import calculate_gae

TEAM_NAME = "Quantum Chaos"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

# gpu = torch.device("cuda:0")
gpu = torch.device("cpu")
cpu = torch.device("cpu")


def run_episode(policy: nn.Module, opponent: Opponent,
                include_barriers: bool = True,
                half_game_size: bool = False,
                max_episode_len_allowed = 1000,
                ):
    env = ShooterEnv(
        opponent_choose_move=opponent,
        render=False,
        include_barriers=include_barriers,
        half_sized_game=half_game_size,
    )
    action_tracker = ActionCountTracker()
    memory = []
    total_return, n_steps = 0, 0

    state, reward, done, info = env.reset()
    total_return += reward

    while not done:
        prev_state = state
        prev_state_in = preprocess_state(state, is_learning=True)
        action_probs = Categorical(policy(prev_state_in))
        action = action_probs.sample()
        action_tracker.track(action.item())
        state, reward, done, info = env.step(action.item())
        if n_steps > max_episode_len_allowed - 1:
            # treat it as a draw and set done, reward 0, then break out of loop
            reward = 0
            done = True
            memory.append((prev_state, action, reward, state, done))
            break
        memory.append((prev_state, action, reward, state, done))
        total_return += reward
        n_steps += 1
    # print("Episode completed, total return:", total_return, "n_steps:", n_steps)
    return memory, total_return, n_steps, action_tracker


def train() -> nn.Module:

    # Hyperparameters
    gamma = 0.97  # only the things you did 20 steps or so ago should matter in a game
    entropy_coeff = 0.01
    lamda = 0.6  # use small initial lamda without barriers
    epsilon = 0.05
    batch_size = 1024  # 512
    epochs_per_batch = 20
    pol_lr = 0.0006
    val_lr = 0.0004
    opponent_pool_exploration_const = 1.0

    # Hyperparameters after walls introduced
    gamma_after_walls = 0.995
    lamda_after_walls = 0.90  # only the last 20 or so steps in the game should matter for look ahead
    epsilon_after_walls = 0.05
    entropy_coeff_after_walls = 0.002
    batch_size_after_walls = 4096  # needs to be a bit larger after walls since episodes are longer
    epochs_per_batch_after_walls = 10

    # Hyperparameters after full size game
    gamma_after_full_size = 0.995  # have to increase because it takes longer to get around
    lamda_after_full_size = 0.9  # only care about full returns, and want unbiased result, so close to MC
    max_episode_len_allowed_after_full_size = 400
    entropy_coeff_after_full_size = 0.0001  # almost completely turn it off

    # Hyper-hyperparameters
    num_episodes = 30_000
    eps_when_barriers_turned_on = 3_000  # when we turn on barriers. Do early to prevent overtraining. Makes training much harder
    delay_for_new_opp_after_barriers_on = 900
    eps_when_full_size = 5_000  # when we turn on full size game. Should be larger than when barriers turned on
    max_episode_len_allowed = 200  # will not train beyond this many steps. Useful to avoid getting stuck in games
    max_num_network_opponents = 10
    eps_per_opponent_refresh = 100
    eps_per_new_opponent = 3_000 # num_episodes // max_num_network_opponents
    n_pol_neurons = 512
    n_val_neurons = 512

    eps_per_debug_printing = 200
    eps_per_logging_slow = 2000
    n_inputs = 48 #24
    n_actions = 6

    include_barriers_now = False  # start off without barriers
    half_game_size_now = True  # start off with half game size

    # Policy net
    policy = nn.Sequential(
        nn.Linear(n_inputs, n_pol_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons, n_pol_neurons ),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons , n_pol_neurons // 2),
        nn.LeakyReLU(),
        nn.Linear(n_pol_neurons // 2, n_actions),
        nn.Softmax(dim=-1),
    )
    pol_optim = torch.optim.Adam(policy.parameters(), lr=pol_lr)

    # Value net
    V = nn.Sequential(
        nn.Linear(n_inputs, n_val_neurons),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons, n_val_neurons // 2),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons // 2, n_val_neurons // 4),
        nn.LeakyReLU(),
        nn.Linear(n_val_neurons // 4, 1)
    )

    val_optim = torch.optim.Adam(V.parameters(), lr=val_lr)
    val_loss_fn = nn.MSELoss()

    # See docstring in opponent_pool.py for more info on the opponent pool
    opponent_pool = OpponentPool(
        [
            choose_move_randomly,
            policy_turn_and_shoot,
            policy_turn_and_shoot_more_random,
            ChooseMoveCheckpoint("Network_barrier_trained_full_size.pt", choose_move_old),  # Old agents that work ok
            ChooseMoveCheckpoint("Network_barrier_trained_full_size_good.pt", choose_move_old),
            ChooseMoveCheckpoint("Network_barrier_trained_full_size_very_good.pt", choose_move_old),
            ChooseMoveCheckpoint("Network_barrier_trained_full_size_very_very_good.pt", choose_move_old),
            ChooseMoveCheckpoint("Network_barrier_trained_full_size_very_very_very_good.pt", choose_move),
            ChooseMoveCheckpoint("Network_final.pt", choose_move),
        ],
        exploration_const=opponent_pool_exploration_const  # sets how much we play others
    )

    past_returns = deque(maxlen=eps_per_debug_printing)
    past_num_steps = deque(maxlen=eps_per_debug_printing)
    action_tracker = ActionCountTracker()
    tbfast = SummaryWriter()
    tbslow = SummaryWriter()
    batch = []

    for ep_num in tqdm(range(num_episodes)):

        if ep_num % eps_per_opponent_refresh == eps_per_opponent_refresh - 1:
            opponent_pool.refresh_opponent_records()
        if (ep_num % eps_per_new_opponent == eps_per_new_opponent - 1
                and ep_num > eps_when_barriers_turned_on + delay_for_new_opp_after_barriers_on):
            # Only start training against self after we have learned to deal with barriers
            checkpoint_model(policy, f"checkpoint_{ep_num + 1}.pt")
            opponent_pool.add_opponent(
                ChooseMoveCheckpoint(f"checkpoint_{ep_num + 1}.pt", choose_move))
        if ep_num % eps_per_debug_printing == eps_per_debug_printing - 1:
            print(f"\n     Visit counts: {opponent_pool.visit_counts}\n",
                  f"     Prob losses: {np.round(opponent_pool.prob_losses,3)}\n",
                  f"     Entropy coef: {entropy_coeff:0.4f}, Avg. ep. length: {sum(past_num_steps) / len(past_num_steps):0.1f}\n",
                  f"     Lambda: {lamda:0.4f}, Epsilon: {epsilon:0.4f}\n",
                  f"     Walls on: {include_barriers_now}, Full size: {not(half_game_size_now)}")

        if ep_num > eps_when_barriers_turned_on:
            include_barriers_now = True
            gamma = gamma_after_walls
            lamda = lamda_after_walls
            epsilon = epsilon_after_walls
            entropy_coeff = entropy_coeff_after_walls
            batch_size = batch_size_after_walls
            epochs_per_batch = epochs_per_batch_after_walls
        if ep_num > eps_when_full_size:
            half_game_size_now = False
            gamma = gamma_after_full_size
            max_episode_len_allowed = max_episode_len_allowed_after_full_size
            entropy_coeff = entropy_coeff_after_full_size
            lamda = lamda_after_full_size

        opponent_id = opponent_pool.select_opponent()  # test
        ep_memory, total_return, n_steps, new_act = run_episode(policy,
                                                                opponent_pool.get_opponent(opponent_id),
                                                                include_barriers=include_barriers_now,
                                                                half_game_size=half_game_size_now,
                                                                max_episode_len_allowed=max_episode_len_allowed)
        opponent_pool.record_score(opponent_id, total_return)
        action_tracker += new_act
        batch += ep_memory
        past_returns.append(int(total_return))
        past_num_steps.append(n_steps)

        if len(batch) >= batch_size:
            this_batch_size = len(batch)
            # Extract data from batch
            states_unprocessed = torch.stack([item[0] for item in batch])
            actions = torch.tensor([item[1] for item in batch], requires_grad=False)
            rewards = torch.tensor([item[2] for item in batch], requires_grad=False)
            successor_states_unprocessed = torch.stack([item[3] for item in batch])
            is_terminals = torch.tensor([item[4] for item in batch], requires_grad=False)

            # preprocess states once instead of during each batch
            states = preprocess_state(states_unprocessed)
            successor_states = preprocess_state(successor_states_unprocessed)

            # Value update
            val_params_before_update = [weight.clone().detach() for name, weight in V.named_parameters()]
            for _ in range(epochs_per_batch):
                vals = torch.squeeze(V(states))
                det_vals = vals.clone().detach()
                with torch.no_grad():
                    successor_vals = (
                            torch.squeeze(V(successor_states)) * ~is_terminals
                    )

                lambda_returns = (
                        calculate_gae(rewards, det_vals, successor_vals, is_terminals, gamma, lamda) + det_vals
                )
                val_loss = val_loss_fn(vals, lambda_returns)

                val_optim.zero_grad()
                val_loss.backward()
                val_optim.step()
            val_params_after_update = [weight.clone().detach() for name, weight in V.named_parameters()]

            # Policy update
            pol_params_before_update = [weight.clone().detach() for name, weight in policy.named_parameters()]
            with torch.no_grad():
                old_pol_probs = policy(states)[range(this_batch_size), actions]
                vals = torch.squeeze(V(states))
                successor_vals = (
                        torch.squeeze(V(successor_states)) * ~is_terminals
                )
                gae = calculate_gae(rewards, vals, successor_vals, is_terminals, gamma, lamda)

            for _ in range(epochs_per_batch):
                pol_dist = Categorical(policy(states))
                pol_probs = pol_dist.probs[range(this_batch_size), actions]
                clipped_obj = torch.clip(pol_probs / old_pol_probs, 1 - epsilon, 1 + epsilon)

                ppo_obj = (
                        torch.min(clipped_obj * gae, (pol_probs / old_pol_probs) * gae)
                        + entropy_coeff * pol_dist.entropy()
                )
                pol_loss = -torch.sum(ppo_obj)/len(ppo_obj)

                pol_optim.zero_grad()
                pol_loss.backward()
                pol_optim.step()
            pol_params_after_update = [weight.clone().detach() for name, weight in policy.named_parameters()]

            batch = []

            # Log tensorboard fast variables
            val_update_strengths = [((val_params_before_update[_i] - val_params_after_update[_i]).std() /
                                     (val_params_before_update[_i]).std()).log10().item()
                                    for _i in range(len(val_params_before_update))]
            pol_update_strengths = [((pol_params_before_update[_i] - pol_params_after_update[_i]).std() /
                                     (pol_params_before_update[_i]).std()).log10().item()
                                    for _i in range(len(pol_params_before_update))]

            for _ind, (name, weight) in enumerate(policy.named_parameters()):
                tbfast.add_scalar(f"Policy log_10(Delta batch of {name})", pol_update_strengths[_ind], ep_num)
            for _ind, (name, weight) in enumerate(V.named_parameters()):
                tbfast.add_scalar(f"Value log_10(Delta batch of {name})", val_update_strengths[_ind], ep_num)
            for _ind, (name, weight) in enumerate(policy.named_parameters()):
                tbfast.add_scalar(f"Policy log_10(Delta step of {name})",
                                  pol_update_strengths[_ind] - np.log10(epochs_per_batch), ep_num)
            for _ind, (name, weight) in enumerate(V.named_parameters()):
                tbfast.add_scalar(f"Value log_10(Delta step of {name})",
                                  val_update_strengths[_ind] - np.log10(epochs_per_batch), ep_num)
            tbfast.add_scalar("Policy loss", pol_loss.item(), ep_num)
            tbfast.add_scalar("Value loss", val_loss.item(), ep_num)
            tbfast.add_scalar('avg_episode_length', sum(past_num_steps) / len(past_num_steps), ep_num)
            tbfast.add_scalar('avg_raw_reward', sum(past_returns) / eps_per_debug_printing, ep_num)
            tbfast.add_scalar("Entropy coef", entropy_coeff, ep_num)
            total_actions = sum(action_tracker.action_counts.values())
            action_probs = [action_tracker.action_counts[action]/total_actions for action in range(6)]
            for action in range(6):
                tbfast.add_scalar(f"Action {action} prob.", action_probs[action], ep_num)

        # Log tensorboard slow variables, histogram of layer outputs and weights/biases
        if ep_num % eps_per_logging_slow == eps_per_logging_slow - 1:
            with torch.no_grad():
                # Choose a network input tensor, and feed it through to observe activations
                data_to_feed = preprocess_state(ep_memory[-1][0])  # use a random recent terminal state to rec
                for layer_index, module in enumerate(policy.children()):
                    data_to_feed = module(data_to_feed)
                    tbslow.add_histogram(f"layer {layer_index} output", data_to_feed, ep_num)
                # Also record weights and activations
                for name, weight in policy.named_parameters():
                    tbslow.add_histogram(name, weight, ep_num)
                for name, weight in policy.named_parameters():
                    tbslow.add_histogram(f'{name}.grad', weight.grad, ep_num)


        if ep_num % eps_per_debug_printing == eps_per_debug_printing - 1:
            print("     ", "0/1: rot L/R, 2: forward, 3: shoot, 4/5: strafe L/R")
            print("     ", action_tracker)
            action_tracker = ActionCountTracker()  # reset action tracker

    tbfast.close()
    tbslow.close()

    return policy

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

    training = True

    # Example workflow, feel free to edit this! ###
    if training:
        my_network = train()
        save_network(my_network, TEAM_NAME)

    # Make sure this does not error! Or your
    # submission will not work in the tournament!
    # check_submission(TEAM_NAME)

    my_network = load_network(TEAM_NAME)

    def choose_move_no_network(state) -> int:
        """The arguments in play_pong_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, neural_network=my_network)

    # Removing the barriers and making the game half-sized will make it easier to train!
    include_barriers = True
    half_game_size = False

    # The code below plays a single game against your bot.
    # You play as the pink ship
    play_shooter(
        your_choose_move=choose_move_no_network, # human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1,
        render=True,
        include_barriers=include_barriers,
        half_game_size=half_game_size,
    )
