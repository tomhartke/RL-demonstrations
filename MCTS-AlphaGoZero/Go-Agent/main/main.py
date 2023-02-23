import torch
from torch import nn
from collections import deque
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# at end run "tensorboard --logdir runs" in terminal to visualize
from typing import Callable, Dict, Optional, Tuple, Literal, List, Any
import time
import scipy.ndimage
import cProfile  # then "pip install snakeviz" and "snakeviz profile.prof"

#  For using delta academy go version pulled from Git
import sys, os
needed_path = os.path.join(os.getcwd(), 'delta_go')
if needed_path not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'delta_go'))
print('new paths: ', sys.path)

# from check_submission import check_submission
from game_mechanics import (
    State,
    all_legal_moves,
    human_player,
    load_pkl,
    play_go,
    save_pkl,
    GoEnv,
    choose_move_randomly,
    reward_function,
    transition_function
)

from opponent_pool import OpponentPool, Opponent
from agzNetwork import alphaGoZeroNet, preprocess_state, get_all_legal_moves_tensor, get_all_legal_moves_smart
from mcts_class import MCTS
from file_saving_loading import ChooseMoveCheckpoint, checkpoint_model

TEAM_NAME = "Quantum Chaos"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

DEVICE_NAME = "cpu"  # or "cuda". Should work on GPU or CPU in those 2 cases

def choose_move_randomly_smart(state: State) -> int:
    """
    Need a slightly better method of rule based play to initialize learning.

    Check if allowed move fills in an eye. If so, prevent it.
    """

    legal_moves_not_filling_eyes = get_all_legal_moves_smart(state)

    chosen_action = legal_moves_not_filling_eyes[int(np.random.random() * len(legal_moves_not_filling_eyes))]

    return chosen_action


def generate_mirror_boards(prev_state_processed: torch.Tensor,
                           legal_moves_tensor: torch.Tensor,
                           policy_probs_observed: torch.Tensor,
                           reward: float) -> List[Tuple[torch.Tensor]]:
    """
    Take in prev state 17x9x9
    legal_moves_tensor 82
    policy_probs_observed 82

    Convert them to 2D tensors, and then flip and rotate them and return all the boards.

    In the end, return a list of the same 3 inputs, after transformations
    """

    legal_moves_tensor_2D = legal_moves_tensor[:-1].view(1, 9, 9)
    policy_probs_observed_2D = policy_probs_observed[:-1].view(1, 9, 9)

    data_holder = []
    chained_2D = torch.cat((prev_state_processed, legal_moves_tensor_2D, policy_probs_observed_2D), dim=0)
    data_holder.append(chained_2D)
    chained_2D_rot = chained_2D.clone().detach()
    for _ in range(3):  # 3 rotations from original
        chained_2D_rot = torch.rot90(chained_2D_rot, k=1, dims=[1, 2]).clone().detach()
        data_holder.append(chained_2D_rot)

    chained_2D_flip = torch.flip(chained_2D.clone().detach(), dims=[-1])
    data_holder.append(chained_2D_flip)
    chained_2D_rot = chained_2D_flip.clone().detach()
    for _ in range(3):  # 3 rotations
        chained_2D_rot = torch.rot90(chained_2D_rot, k=1, dims=[1, 2]).clone().detach()
        data_holder.append(chained_2D_rot)

    # now we want to convert each term in data_holder back to the original datastructure
    data_holder_reformat = []
    for _chained_2D in data_holder:
        _prev_state_processed = _chained_2D[0:17]
        _legal_moves_tensor_2D = _chained_2D[17]
        _policy_probs_observed_2D = _chained_2D[18]
        _legal_moves_tensor = torch.cat((_legal_moves_tensor_2D.reshape(-1),
                                         torch.tensor([1.0], dtype=torch.float32)), dim=0)  # pass is always legal
        _policy_probs_observed = torch.cat((_policy_probs_observed_2D.reshape(-1),
                                            torch.tensor([policy_probs_observed[-1].item()], dtype=torch.float32)),
                                           dim=0)
        data_holder_reformat.append((_prev_state_processed, _legal_moves_tensor, _policy_probs_observed, reward))

    return data_holder_reformat


def run_episode(network: nn.Module, opponent: Opponent,
                max_episode_len_allowed=1000,
                num_rollouts=10,
                num_first_moves_at_high_temp=8,
                use_random=False,
                render=False,
                dirichlet_noise_alpha=0.03,
                dirichlet_noise_epsilon=0.25
                ):
    """
    Play a game against opponent, and gather training examples in memory
    """
    env = GoEnv(
        opponent_choose_move=opponent,
        render=render,
        verbose=False,
        game_speed_multiplier=10
    )
    memory = []
    total_return, n_steps = 0, 0

    # set up things
    my_mcts = MCTS()  # need to reinitialize MCTS
    # Turn on eval mode for network since we don't have batches
    network.eval()

    state, reward, done, info = env.reset()
    total_return += reward

    print("   Playing as: ", state.to_play)

    while not done:
        prev_state = state
        prev_state_processed = preprocess_state(prev_state)  # converts to tensors
        legal_moves_tensor = get_all_legal_moves_tensor(prev_state)

        if use_random:
            action = choose_move_randomly_smart(prev_state)
        else:
            if n_steps < num_first_moves_at_high_temp:
                action = choose_move_training(prev_state, network, my_mcts,
                                              max_time_s=100.0,  # not limited by time
                                              num_rollouts=num_rollouts, temperature=1.0,
                                              add_Dirichlet_noise_to_root=True,
                                              dirichlet_noise_alpha=dirichlet_noise_alpha,
                                              dirichlet_noise_epsilon=dirichlet_noise_epsilon)
            else:
                action = choose_move_training(prev_state, network, my_mcts,
                                              max_time_s=100.0,  # not limited by time
                                              num_rollouts=num_rollouts, temperature=0.0,
                                              add_Dirichlet_noise_to_root=True,
                                              dirichlet_noise_alpha=dirichlet_noise_alpha,
                                              dirichlet_noise_epsilon=dirichlet_noise_epsilon)

        # if action == 81:
        #     print('Chose pass!!! as player ', prev_state.to_play)

        policy_probs_observed = torch.zeros_like(legal_moves_tensor).type(torch.float32)
        policy_probs_observed[action] = 1.0
        state, reward, done, info = env.step(action)

        # Generate mirror boards to sample data 8x faster for training
        memory_mirrors = generate_mirror_boards(prev_state_processed, legal_moves_tensor,
                                                policy_probs_observed, reward)

        if n_steps > max_episode_len_allowed - 1:
            # treat it as a draw and set done, reward 0, then break out of loop
            reward = 0
            for _mem in memory_mirrors:
                memory.append(_mem)
            break
        for _mem in memory_mirrors:
            memory.append(_mem)
        total_return += reward
        n_steps += 1

    # Lastly replace the rewards in the memory with total return
    memory_returns = [(mem[0], mem[1], mem[2], total_return) for mem in memory]
    # now we can use these immediately for training

    return memory_returns, total_return, n_steps


def train(network=None, initial_learning=True) -> nn.Module:
    # Hyperparameters
    batch_size = 5000  # 5000 is roughly every 10 games
    bootstrap_size = 60
    bootstrap_oversample_rate = 1.0  # Can't oversample bootstrap or we overfit to a few games with bad generalization. It is already effectively 8 because of mirroring boards
    bootstrap_number = int(bootstrap_oversample_rate * batch_size) // bootstrap_size  # oversample on bootstrapping
    opponent_pool_exploration_const = 0.5  # strongly favor playing best opponent
    opponent_pool_memory_maxlen = 150  # forget old opponent tallies fast
    lr = 0.0001  # reducing learning rate to improve training by preventing overfitting to a few games
    l2_regularizer = 1e-4
    num_first_moves_at_high_temp = 8
    dirichlet_noise_alpha = 0.03
    dirichlet_noise_epsilon = 0.25
    render_control = False  # if false, prevents any rendering at all.

    # Hyper-hyperparameters
    num_episodes = 600
    max_episode_len_allowed = 200  # will not train beyond this many steps. Useful to avoid getting stuck in games
    eps_per_opponent_refresh = 25  # when to update wins and losses against opponents in opponent pool
    eps_per_new_opponent = 100  # only happens after random agent is turned off.
    eps_per_debug_printing = 10
    eps_per_logging_slow = 100

    eps_turn_on_non_random_learning = 1  # when to end random play
    eps_turn_on_end_learning = 400  # update rollout number
    eps_turn_on_other_opponent_learning = 10000  # when to add other opponents. Not currently used

    training_num_rollouts_non_random = 50  # 3.5 hrs for 400 games at 50 rollouts
    training_num_rollouts_end = 150  # 4.5 more hours for 200 games at 150 rollouts

    if not initial_learning:  # slightly different training setup
        print("!!!doing continuing training!!!")
        # After we have latched an initial policy that works ok, we want to iteratively improve it
        # By many many training games with only a few rollouts.

        # v0 -> 1
        num_episodes = 200
        lr = 0.00005  # lower learning rate
        eps_turn_on_non_random_learning = 0  # when to turn off random play
        eps_turn_on_other_opponent_learning = 1  # when to add other opponents
        eps_turn_on_end_learning = 100

        training_num_rollouts_non_random = 40
        training_num_rollouts_end = 40  # 10s per game roughly

        # v1 -> 2

    training_num_rollouts = training_num_rollouts_non_random  # start with this

    if network == None:  # then initialize it, otherwise assume it's come from a loaded file for continued training
        network = alphaGoZeroNet(learning_rate=lr, l2_regularizer=l2_regularizer, device=DEVICE_NAME)
    else:  # network was loaded and can be used as is.
        # Send the network to the device
        network.to(torch.device(DEVICE_NAME))
        network.device = DEVICE_NAME
        network.learning_rate = lr
        network.l2_regularizer = l2_regularizer
        network.optimizer = torch.optim.Adam(network.parameters(),
                                             lr=network.learning_rate,
                                             weight_decay=network.l2_regularizer,
                                             )

    network.train()

    loss_fn_policy = nn.CrossEntropyLoss()
    loss_fn_value = nn.MSELoss()

    # See docstring in opponent_pool.py for more info on the opponent pool
    opponent_pool = OpponentPool(
        [
            choose_move_randomly_smart,
        ],
        memory_maxlen=opponent_pool_memory_maxlen,
        exploration_const=opponent_pool_exploration_const  # sets how much we play others
    )

    past_returns = deque(maxlen=eps_per_debug_printing)
    past_num_steps = deque(maxlen=eps_per_debug_printing)
    tbfast = SummaryWriter()
    tbslow = SummaryWriter()
    batch = []

    for ep_num in tqdm(range(num_episodes)):

        if ep_num % eps_per_opponent_refresh == eps_per_opponent_refresh - 1:
            opponent_pool.refresh_opponent_records()
        if (ep_num % eps_per_new_opponent == eps_per_new_opponent - 1) and ep_num > eps_turn_on_non_random_learning - 2:
            # Only start training against self after we have learned to deal with barriers
            checkpoint_model(network, f"checkpoint_{ep_num + 1}.pt")
            opponent_pool.add_opponent(
                ChooseMoveCheckpoint(f"checkpoint_{ep_num + 1}.pt", choose_move_training, MCTS(), training_num_rollouts)
                # adds them to the opponent pool with the current number of rollouts, and their own MCTS tree

            )
        if ep_num % eps_per_debug_printing == eps_per_debug_printing - 1:
            print(f"\n     Visit counts: {opponent_pool.visit_counts}\n",
                  f"     Prob losses: {np.round(opponent_pool.prob_losses, 3)}\n")
            print('     Avg_raw_reward last ', eps_per_debug_printing, ' episodes:',
                  sum(past_returns) / eps_per_debug_printing)

        render = False and render_control
        use_random = True  # use random initially
        if ep_num > eps_turn_on_non_random_learning:
            training_num_rollouts = training_num_rollouts_non_random
            use_random = False
            render = True and render_control
            # add some decently good opponents for end of training
        if ep_num == eps_turn_on_other_opponent_learning:
            opponent_pool.add_opponent(
                ChooseMoveCheckpoint(f"opponent_checkpoint_play_circle.pt", choose_move_training, MCTS(),
                                     training_num_rollouts=5))
            opponent_pool.add_opponent(
                ChooseMoveCheckpoint(f"opponent_checkpoint_play_well.pt", choose_move_training, MCTS(),
                                     training_num_rollouts=5))
            opponent_pool.add_opponent(
                ChooseMoveCheckpoint(f"opponent_checkpoint_350_games.pt", choose_move_training, MCTS(),
                                     training_num_rollouts=5))
        if ep_num > eps_turn_on_end_learning:
            training_num_rollouts = training_num_rollouts_end

        opponent_id = opponent_pool.select_opponent()  # test
        ep_memory, total_return, n_steps = run_episode(network, opponent_pool.get_opponent(opponent_id),
                                                       max_episode_len_allowed=max_episode_len_allowed,
                                                       num_rollouts=training_num_rollouts,
                                                       num_first_moves_at_high_temp=num_first_moves_at_high_temp,
                                                       use_random=use_random, render=render,
                                                       dirichlet_noise_alpha=dirichlet_noise_alpha,
                                                       dirichlet_noise_epsilon=dirichlet_noise_epsilon)
        print('   Return: ', total_return, ' against opponent id: ', str(opponent_id))
        opponent_pool.record_score(opponent_id, total_return)
        batch += ep_memory
        past_returns.append(int(total_return))
        past_num_steps.append(n_steps)

        if len(batch) >= batch_size:
            print('   Batch training! Num examples:', len(batch))
            network.train()  # turn on training mode

            this_batch_size = len(batch)
            # Extract data from batch
            states_processed_batch = torch.stack([item[0] for item in batch])  # (B, 17, 9, 9)
            policy_probs_observed_batch = torch.stack([item[2] for item in batch])  # (B, 82)
            policy_probs_observed_batch.requires_grad = False
            legal_moves_tensor_batch = torch.stack([item[1] for item in batch])  # (B, 82)
            legal_moves_tensor_batch.requires_grad = False
            returns_batch = torch.tensor([item[3] for item in batch], requires_grad=False, dtype=torch.float32)  # (B)

            if DEVICE_NAME == "cuda":  # send things to cuda
                states_processed_batch = states_processed_batch.cuda()
                policy_probs_observed_batch = policy_probs_observed_batch.cuda()
                legal_moves_tensor_batch = legal_moves_tensor_batch.cuda()
                returns_batch = returns_batch.cuda()

            # Iterate through the batch doing bootstrapping.
            network_params_before_update = [weight.clone().detach() for name, weight in network.named_parameters()]
            for _num_bootstrap in range(bootstrap_number):
                bootstrap_inds = np.random.choice(np.arange(len(returns_batch)), size=bootstrap_size, replace=True)

                # Grab samples
                states_processed_boot = states_processed_batch[bootstrap_inds]
                policy_probs_observed_boot = policy_probs_observed_batch[bootstrap_inds]
                legal_moves_tensor_boot = legal_moves_tensor_batch[bootstrap_inds]
                returns_boot = returns_batch[bootstrap_inds]

                # now feed batch through network
                policy_probs, pred_values, policy_logits = network.get_predictions(states_processed_boot,
                                                                                   legal_moves_tensor_boot)
                pred_values = torch.squeeze(pred_values, dim=-1)  # to match values array.

                pol_loss = loss_fn_policy(policy_logits, policy_probs_observed_boot)
                val_loss = loss_fn_value(pred_values, returns_boot)
                if _num_bootstrap == 0:
                    initial_val_loss = val_loss.item()  # recorded to check for overfitting of value
                    initial_pol_loss = pol_loss.item()
                if _num_bootstrap == bootstrap_number - 1:
                    final_val_loss = val_loss.item()  # recorded to check for overfitting of value
                    final_pol_loss = pol_loss.item()
                loss = val_loss + pol_loss

                network.optimizer.zero_grad()
                loss.backward()
                network.optimizer.step()

            network_params_after_update = [weight.clone().detach() for name, weight in network.named_parameters()]

            # reset batch
            batch = []

            # Log tensorboard fast variables
            network_update_strengths = [((network_params_before_update[_i] - network_params_after_update[_i]).std() /
                                         (network_params_before_update[_i]).std()).log10().item()
                                        for _i in range(len(network_params_before_update))]
            # pol_update_strengths = [((pol_params_before_update[_i] - pol_params_after_update[_i]).std() /
            #                          (pol_params_before_update[_i]).std()).log10().item()
            #                         for _i in range(len(pol_params_before_update))]

            for _ind, (name, weight) in enumerate(network.named_parameters()):
                tbfast.add_scalar(f"Network log_10(Delta batch of {name})",
                                  network_update_strengths[_ind] - np.log10(bootstrap_number), ep_num)

            tbfast.add_scalar("Initial Policy loss (batch)", initial_pol_loss, ep_num)
            tbfast.add_scalar("Initial Value loss (batch)", initial_val_loss, ep_num)
            tbfast.add_scalar("Final Policy loss (batch)", final_pol_loss, ep_num)
            tbfast.add_scalar("Final Value loss (batch)", final_val_loss, ep_num)
            tbfast.add_scalar('avg_episode_length', sum(past_num_steps) / len(past_num_steps), ep_num)
            tbfast.add_scalar('avg_raw_reward', sum(past_returns) / eps_per_debug_printing, ep_num)

            print("Init/final val loss: ", initial_val_loss, final_val_loss, " and init/final pol loss: ",
                  initial_pol_loss, final_pol_loss)

        # # Log tensorboard slow variables, histogram of layer outputs and weights/biases
        # if ep_num % eps_per_logging_slow == eps_per_logging_slow - 1:
        #     with torch.no_grad():
        #         # # Choose a network input tensor, and feed it through to observe activations
        #         # data_to_feed = preprocess_state(ep_memory[-1][0])  # use a random recent terminal state to rec
        #         # for layer_index, module in enumerate(policy.children()):
        #         #     data_to_feed = module(data_to_feed)
        #         #     tbslow.add_histogram(f"layer {layer_index} output", data_to_feed, ep_num)
        #         # Also record weights and activations
        #         for name, weight in network.named_parameters():
        #             tbslow.add_histogram(name, weight, ep_num)
        #         for name, weight in network.named_parameters():
        #             tbslow.add_histogram(f'{name}.grad', weight.grad, ep_num)

    tbfast.close()
    tbslow.close()

    network.eval()  # turn off training mode

    return network


def choose_move_training(
        state: State,
        network: nn.Module,
        mcts: MCTS,
        max_time_s: float = 100.0,
        num_rollouts: float = 100,
        temperature: float = 0.0,
        add_Dirichlet_noise_to_root: bool = False,
        dirichlet_noise_alpha: float = 0.03,
        dirichlet_noise_epsilon: float = 0.25
) -> int:
    start_time = time.time()

    # if MCTS is empty, then we should set current state as initial state (root node)
    # This happens if we ended a game
    if len(mcts.tree) <= 1:
        mcts.set_initial_state(state, network)

    # First prune tree based on current state,
    mcts.prune_tree(state, network)

    # Add Dirichlet noise to root node during training if necessary to encourage exploration
    if add_Dirichlet_noise_to_root:
        noise_distribution = np.random.dirichlet(alpha=([dirichlet_noise_alpha] * 82))
        current_policy_d = mcts.tree_probs[mcts.root_node.key]
        noisy_policy_d = (
                                     1.0 - dirichlet_noise_epsilon) * current_policy_d + dirichlet_noise_epsilon * noise_distribution
        mcts.tree_probs[mcts.root_node.key] = noisy_policy_d

    # now run rollouts. If we hit a time limmit, or num_rollouts, then stop
    for _i in range(num_rollouts):
        end_time = time.time()
        if end_time - start_time > max_time_s:  # we have to stop the rollouts
            # print('too much time!! ', end_time - start_time, ' after num_rollouts: ', _i)
            # print("     num rollouts:", _i)
            break
        mcts.do_rollout(network)
    action = mcts.choose_action(temperature=temperature)

    return action


def choose_move(
        state: State,
        pkl_file: Optional[Any] = None,
        mcts: Optional[MCTS] = None,
) -> int:
    """Called during competitive play. It returns a single action to play.

    Args:
        state: The current state of the go board (see state.py)
        pkl_file: The pickleable object you returned in train
        env: The current environment

    Returns:
        The action to take
    """

    network = pkl_file  # hopefully this contains the entire function calls and such it is wrapped in.

    action = choose_move_training(state, network, mcts, max_time_s=0.95,
                                  num_rollouts=10000000,  # not limited. Limited by time.
                                  temperature=0.0,
                                  add_Dirichlet_noise_to_root=False)

    return action


if __name__ == "__main__":

    training_now = False
    training_continue = False  # set to true to continue from a previous version
    render_final_game = True
    torch_load_version = 2  # which network version to load to train
    torch_save_version = torch_load_version + 1 if training_continue else 0

    if training_now:
        if not training_continue:  # make a new network
            file = train()
        else:  # we are continuing training, so load network from file and train with that
            intermediate_network = torch.load((TEAM_NAME + "_v" + str(torch_load_version) + "_torch_save.pt"),
                                              map_location=torch.device(DEVICE_NAME))
            file = train(network=intermediate_network, initial_learning=False)

        # Move to cpu and resave as next version or initial version
        print("Are we initially on GPU? ", next(file.parameters()).is_cuda)
        file.to(torch.device('cpu'))
        file.device = 'cpu'  # have to set internal device attribute of Class as well, so internal function calls are correct
        torch.save(file, (TEAM_NAME + "_v" + str(torch_save_version) + "_torch_save.pt"))
        print("Are we in the end on GPU? ", next(file.parameters()).is_cuda)

        # file = load_pkl(TEAM_NAME)
        # save_pkl(file, TEAM_NAME + ' CPU')

    elif not training_now:
        # reload file as torch.load, then resave as pkl
        my_network_torch = torch.load((TEAM_NAME + "_v" + str(torch_load_version) + "_torch_save.pt"),
                                map_location=torch.device('cpu'))
        # Resave file as pkl file and then reload it (hopefully works on cpu now)
        save_pkl(my_network_torch, TEAM_NAME)

        # Reload
        file = load_pkl(TEAM_NAME)
        my_mcts = MCTS()

        # Choose move functions when called in the game_mechanics expect only a state
        # argument, here is an example of how you can pass a pkl file and an initialized
        # mcts tree
        def choose_move_no_network(state: State) -> int:
            """The arguments in play_game() require functions that only take the state as input.

            This converts choose_move() to that format.
            """
            return choose_move(state, file, mcts=my_mcts)

        # check_submission(
        #     TEAM_NAME, choose_move_no_network
        # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

        # Play a game against your bot!
        # Left click to place a stone. Right click to pass!
        total_return = play_go(
            your_choose_move=choose_move_no_network,  # choose_move_randomly_smart,  # human_player,
            opponent_choose_move=choose_move_no_network,
            game_speed_multiplier=1,
            render=render_final_game,
            verbose= not render_final_game,
        )
        print('Total return: ', total_return)