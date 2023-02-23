from collections import Counter
import random

import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
from game_mechanics import choose_move_randomly

def preprocess_state_old(state: torch.Tensor,
                         is_learning: bool = False) -> torch.Tensor:
    """
    Process input state to feed to nn based on environment
    """
    return state

def choose_move_old(
        state: torch.Tensor,
        neural_network: nn.Module,
) -> int:  # <--------------- Please do not change these arguments!
    """Called during competitive play. It acts greedily given current state and neural network
    function dictionary. It returns a single action to take.

    Args:
        state: State of the game as a torch tensor, length = 24.
        neural_network: The pytorch network output by train().

    Returns:
        move: The move you want to give the state of the game.
                    Should be in {0,1,2,3,4,5}.
    """

    action_probs = neural_network(preprocess_state_old(state))

    return int(Categorical(action_probs).sample())

def get_extra_features_from_state(state: torch.Tensor) -> torch.Tensor:
    """
    Gets impact parameters of incoming objects in the games.
    """
    def get_relative_ship_attributes(_object: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        """

        Args:
            _object: the source object, ie my spaceship
            _target: the target object, ie the other ship

        """

        delta_x = _target[0] - _object[0]  # where is other ship
        delta_y = -(_target[1] - _object[1])  # negative because measured down from top
        object_cos = -_object[2]  # not sure whats going on but these work
        object_sin = _object[3]
        impact_param = object_cos * delta_y - object_sin * delta_x
        effective_dist = object_cos * delta_x + object_sin * delta_y
        one_hot_left = torch.heaviside(-impact_param, torch.tensor(0.0, dtype=torch.float))
        one_hot_right = torch.heaviside(impact_param, torch.tensor(0.0, dtype=torch.float))

        return torch.stack((one_hot_left, one_hot_right, effective_dist, impact_param))

    def get_bullet_trajectory(_object: torch.Tensor, _target: torch.Tensor, _bullet: torch.Tensor) -> torch.Tensor:
        """

        Args:
            _object: the place the bullet came from
            _target: the target object, ie the other ship
            _bullet: the bullet parameters

        NOTE bullets dont have angle information, just locaiton, so we have to extract it somehow

        """
        # even if bullet exists, have to check distance from source is nonzero to be able to define angle
        bullet_dx_from_source = _bullet[0] - _object[0]
        bullet_dy_from_source = -(_bullet[1] - _object[1])

        if _bullet[0] > -0.99 and _bullet[1] > -0.99:  # then the bullet exists, proceed
            delta_x = _target[0] - _bullet[0]  # where is other ship relative to bullet
            delta_y = -(_target[1] - _bullet[1])  # negative because measured down from top
            # now have to get angle of bullet. Going to assume since it came from opp, that angle is given by opp angle
            bullet_cos = bullet_dx_from_source/np.sqrt(bullet_dx_from_source**2 + bullet_dy_from_source**2 + 1e-13)
            bullet_sin = bullet_dy_from_source/np.sqrt(bullet_dx_from_source**2 + bullet_dy_from_source**2 + 1e-13)
            impact_param = bullet_cos * delta_y - bullet_sin * delta_x
            effective_dist = bullet_cos * delta_x + bullet_sin * delta_y
            one_hot_left = torch.heaviside(-impact_param, torch.tensor(0.0, dtype=torch.float))
            one_hot_right = torch.heaviside(impact_param, torch.tensor(0.0, dtype=torch.float))
        else: # object doesn't exist, set outputs to standard output
            impact_param = torch.tensor(0.0, dtype=torch.float)
            effective_dist = torch.tensor(-1.0, dtype=torch.float)  # means its effectively way past and facing away
            one_hot_left = torch.tensor(-1.0, dtype=torch.float)
            one_hot_right = torch.tensor(-1.0, dtype=torch.float)
        # print(one_hot_left, one_hot_right, effective_dist, impact_param)
        return torch.stack((one_hot_left, one_hot_right, effective_dist, impact_param))

    my_ship = state[0:4]
    op_ship = state[4:8]
    my_bul1 = state[8:12]
    my_bul2 = state[12:16]
    op_bul1 = state[16:20]
    op_bul2 = state[20:24]

    my_ship_to_opp = get_relative_ship_attributes(my_ship, op_ship)
    op_ship_to_me = get_relative_ship_attributes(op_ship,my_ship)
    op_bul1_to_me = get_bullet_trajectory(op_ship, my_ship, op_bul1)
    op_bul2_to_me = get_bullet_trajectory(op_ship, my_ship, op_bul2)
    my_bul1_to_op = get_bullet_trajectory(my_ship, op_ship, my_bul1)
    my_bul2_to_op = get_bullet_trajectory(my_ship, op_ship, my_bul2)

    feat_vec_list = [my_ship_to_opp,
                     op_ship_to_me,
                     my_bul1_to_op,
                     my_bul2_to_op,
                     op_bul1_to_me,
                     op_bul2_to_me]
    feat_vec = torch.cat(feat_vec_list)
    feat_vec_long = torch.cat((state, feat_vec))

    return feat_vec_long

def policy_turn_and_shoot(state: torch.Tensor) -> int:
    """Opponent policy to use in initial training.

    Turns toward opponent then shoots, with a bit of randomness
    """


    delta_x_unnormed = state[4].item() - state[0].item()  # where is other ship
    delta_y_unnormed = -(state[5].item() - state[1].item())  # negative because measured down from top
    my_x_unnormed = -state[2].item()  # not sure whats going on but these work
    my_y_unnormed = state[3].item()
    eps_no_div0 = 1e-13
    delta_x = delta_x_unnormed / np.sqrt(delta_x_unnormed**2 + delta_y_unnormed**2)
    delta_y = delta_y_unnormed / np.sqrt(delta_x_unnormed**2 + delta_y_unnormed**2)
    my_x = my_x_unnormed / np.sqrt(my_x_unnormed**2 + my_y_unnormed**2)
    my_y = my_y_unnormed / np.sqrt(my_x_unnormed**2 + my_y_unnormed**2)

    cross_prod = -( my_x * delta_y - my_y * delta_x)
    dot_prod = my_x * delta_x + my_y * delta_y
    # This provides a good way to decide which way to turn.
    # If it is positive, turn right, otherwise turn left

    rot_dir = 0 if cross_prod >= 0 else 1

    dot_prod_threshold = 0.99 # degrees
    need_to_rot = True if dot_prod < dot_prod_threshold else False

    # also check whether we have bullets to shoot or not
    bullet_2_state = state[12:16]
    bullet_2_fired = True if (bullet_2_state[0] != -1) else False

    # print('cross prod dot prod,', cross_prod, dot_prod)

    # now policy
    action = 2
    random_num_test = np.random.random()
    if random_num_test < 0.05:
        action = choose_move_randomly(state)
    elif random_num_test < 0.15:
        action = int(np.random.choice([4,5]))
    elif random_num_test < 0.25:
        action = 3  # shoot
    else:   # follow policy to turn, then shoot deterministically
        if bullet_2_fired:  # then strafe or move forward instead of turning to shoot
            action = int(np.random.choice([2,4,5]))
        else:  # bullet not fired, rotate toward and then fire
            if need_to_rot:
                action = rot_dir
            else:
                action = 3  # shoot

    return action

def policy_turn_and_shoot_more_random(state: torch.Tensor) -> int:
    """Opponent policy to use in initial training.

    Turns toward opponent then shoots, with a bit of randomness
    """

    delta_x_unnormed = state[4].item() - state[0].item()  # where is other ship
    delta_y_unnormed = -(state[5].item() - state[1].item())  # negative because measured down from top
    my_x_unnormed = -state[2].item()  # not sure whats going on but these work
    my_y_unnormed = state[3].item()
    eps_no_div0 = 1e-13
    delta_x = delta_x_unnormed / np.sqrt(delta_x_unnormed**2 + delta_y_unnormed**2)
    delta_y = delta_y_unnormed / np.sqrt(delta_x_unnormed**2 + delta_y_unnormed**2)
    my_x = my_x_unnormed / np.sqrt(my_x_unnormed**2 + my_y_unnormed**2)
    my_y = my_y_unnormed / np.sqrt(my_x_unnormed**2 + my_y_unnormed**2)

    cross_prod = -( my_x * delta_y - my_y * delta_x)
    dot_prod = my_x * delta_x + my_y * delta_y
    # This provides a good way to decide which way to turn.
    # If it is positive, turn right, otherwise turn left

    rot_dir = 0 if cross_prod >= 0 else 1

    dot_prod_threshold = 0.99 # degrees
    need_to_rot = True if dot_prod < dot_prod_threshold else False

    # also check whether we have bullets to shoot or not
    bullet_2_state = state[12:16]
    bullet_2_fired = True if (bullet_2_state[0] != -1) else False

    # print('cross prod dot prod,', cross_prod, dot_prod)

    # now policy
    action = 2
    random_num_test = np.random.random()
    if random_num_test < 0.1:  # act totally random
        action = choose_move_randomly(state)
    elif random_num_test < 0.4:
        action = int(np.random.choice([2, 4, 4, 5, 5]))  # strafe/move (should help avoiding walls)
    elif random_num_test < 0.5:
        action = 3  # shoot
    else:   # follow policy to turn, then shoot deterministically
        if bullet_2_fired:  # then strafe or move forward instead of turning to shoot
            action = int(np.random.choice([2, 4, 4, 5, 5]))
        else:  # bullet not fired, rotate toward and then fire
            if need_to_rot:
                action = rot_dir
            else:
                action = 3  # shoot

    return action
    # 10% of time act randomly
    # 20% of time strafe randomly
    # 10% of time shoot randomly
    # Otherwise turn toward oppponent then shoot



# def complex_policy(state: State) -> int:
#     """Opponent policy to use in initial training."""
#     check_better_than_pair = (has_two_pair(state)
#                               or has_three(state)
#                               or has_straight(state)
#                               or has_flush(state)
#                               or has_full_house(state)
#                               or has_four(state)
#                               or has_straight_flush(state))
#
#     check_better_than_triplet = (has_straight(state)
#                                  or has_flush(state)
#                                  or has_full_house(state)
#                                  or has_four(state)
#                                  or has_straight_flush(state))
#
#     # if pre-flop, call/check
#     if not state.public_cards:
#         ####################### extra stuff #######################
#         if has_pair(state) or check_better_than_pair:
#             # then sometimes randomly raise preflop
#             if np.random.random() < 0.2:
#                 if 3 in state.legal_actions:
#                     return 3
#                 if 2 in state.legal_actions:
#                     return 2
#             if np.random.random() < 0.01:
#                 if 4 in state.legal_actions:
#                     return 4
#         ####################### extra stuff #######################
#
#         if state.opponent_chips_remaining == 0:
#             # Oppo goes all-in
#             if has_pair(state):
#                 if np.random.random() < 0.5:
#                     return 1
#             elif has_face_card_in_hand(state):
#                 return random.choice([0] * 3 + [1])
#             else:
#                 # 90% chance of folding, sometimes call bluff
#                 return random.choice([0] * 9 + [1])
#         else:
#             # Sometimes randomly bet without good hand
#             if np.random.random() < 0.05:
#                 if 2 in state.legal_actions:
#                     return 2
#             if np.random.random() < 0.05:
#                 if 3 in state.legal_actions:
#                     return 3
#             # Wants to see flop
#             return 1 if 1 in state.legal_actions and state else 0
#
#     # post flop
#     # inject some randomness to bet even if you don't have a good hand
#     if not (check_better_than_pair):
#         if np.random.random() < 0.05:
#             # bet any way possible
#             return (
#                 3
#                 if 3 in state.legal_actions
#                 else 2
#                 if 2 in state.legal_actions
#                 else 1
#                 if 1 in state.legal_actions
#                 else 0
#             )
#     if has_pair(state) and not(check_better_than_pair):
#         # inject some randomness to call even if you have a good hand
#         if np.random.random() < 0.5:
#             if 1 in state.legal_actions:
#                 return 1
#         return (
#             3
#             if 3 in state.legal_actions
#             else 2
#             if 2 in state.legal_actions
#             else 1
#             if 1 in state.legal_actions
#             else 0
#         )
#     if has_two_pair(state) or has_three(state):
#         # inject some randomness to call even if you have a good hand
#         if np.random.random() < 0.15:
#             if 1 in state.legal_actions:
#                 return 1
#         return (
#             3
#             if 3 in state.legal_actions
#             else 2
#             if 2 in state.legal_actions
#             else 1
#             if 1 in state.legal_actions
#             else 0
#         )
#     if check_better_than_triplet:
#         # go all in with some probability
#         if np.random.random() < 0.05:
#             if 4 in state.legal_actions:
#                 return 4
#         # inject some randomness to call even if you have a good hand
#         if np.random.random() < 0.05:
#             if 1 in state.legal_actions:
#                 return 1
#                 # Post-flop with a pair or better, bet any way possible
#         return (
#             3
#             if 3 in state.legal_actions
#             else 2
#             if 2 in state.legal_actions
#             else 1
#             if 1 in state.legal_actions
#             else 0
#         )
#     # Other wise, we have literally nothing
#     if state.opponent_chips == state.player_chips:
#         # and opponent didn't bet.
#         # Still sometimes bet to introduce randomness
#         if np.random.random() < 0.1:
#             return (2 if 2 in state.legal_actions else 1
#                 if 1 in state.legal_actions
#                 else 0)
#         if np.random.random() < 0.05:
#             return (3 if 3 in state.legal_actions else 1
#                 if 1 in state.legal_actions
#                 else 0)
#         return (1
#             if 1 in state.legal_actions
#             else 0)
#     else:  # opponent made a bet
#         # Set up sometimes randomly calling when have nothing, and bet is small
#         bet_size = state.opponent_chips - state.player_chips
#         self_chips_left = state.player_chips_remaining
#         if bet_size / self_chips_left < 0.2:
#             if np.random.random() < 0.05:
#                 return (2 if 2 in state.legal_actions else 1
#                     if 1 in state.legal_actions
#                     else 0)
#             if np.random.random() < 0.15:
#                 return (1
#                     if 1 in state.legal_actions
#                     else 0)
#         elif bet_size / self_chips_left < 0.5:
#             if np.random.random() < 0.05:
#                 return (1
#                     if 1 in state.legal_actions
#                     else 0)
#         return 0
