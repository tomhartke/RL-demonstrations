from collections import Counter
import random

import torch
import numpy as np

from game_mechanics import State


def suited_hand(state: State) -> bool:
    return state.hand[0][0] == state.hand[1][0]


def low_card_in_hand(state: State) -> bool:
    return any(val in [state.hand[0][1], state.hand[1][1]] for val in ["2", "3", "4", "5"])


def med_card_in_hand(state: State) -> bool:
    return any(val in [state.hand[0][1], state.hand[1][1]] for val in ["6", "7", "8", "9", "T"])


def has_face_card_in_hand(state: State) -> bool:
    return any(val in [state.hand[0][1], state.hand[1][1]] for val in ["K", "Q", "J"])


def has_ace_in_hand(state: State) -> bool:
    return "A" in [state.hand[0][1], state.hand[1][1]]


def has_pair(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 2


def pair_incl_hand_card(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    most_common = Counter([card[1] for card in all_cards]).most_common(1)[0]
    return any(card[1] == most_common[0] for card in state.hand)


def pair_is_highest(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    most_common = Counter([card[1] for card in all_cards]).most_common(1)[0]

    order = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    card_indices = [order.index(card[1]) for card in all_cards]
    highest_card = order[max(card_indices)]
    return highest_card == most_common[0]


def has_two_pair(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    if len(all_cards) < 4:
        return False
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 2 and Counter([card[1] for card in all_cards]).most_common(2)[1][1] == 2


def has_three(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 3


def has_straight(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    card_indices = []
    for card in all_cards:
         card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort

    for i in range(len(card_indices) - 4):
        five_cards = card_indices[i: i + 5]
        if five_cards[4] - five_cards[0] == 4:
            return True
    return False


def my_straight_possible(state: State) -> float:
    """
    Returns the (very rough) probability that a straight is possible
     with the cards in the hand and on the table.
    """
    if not state.public_cards:
        return 0

    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    all_cards = state.hand + state.public_cards
    card_indices = []
    for card in all_cards:
        card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort
    if len(state.public_cards) < 5:
        for i in range(len(card_indices) - 3):
            four_cards = card_indices[i: i + 4]
            if four_cards[3] - four_cards[0] <= 4:
                # Having 1 card left to complete a straight is better on flop than turn
                return 1 if len(state.public_cards) == 3 else 0.5
    if len(state.public_cards) < 4:
        for i in range(len(card_indices) - 2):
            three_cards = card_indices[i: i + 3]
            if three_cards[2] - three_cards[0] <= 4:
                # ~Half as good to require turn & river cards as it is to require only 1 card
                return 0.25
    return 0


def oppo_straight_possible(state: State) -> float:
    """
    Returns the (very rough) probability that a straight is possible
     for the opponent, given the cards on the table.
    """
    if not state.public_cards:
        return 0

    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    card_indices = []
    for card in state.public_cards:
        card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort
    if len(state.public_cards) <= 5:
        for i in range(len(card_indices) - 3):
            four_cards = card_indices[i: i + 4]
            if four_cards[3] - four_cards[0] <= 4:
                return 1
        for i in range(len(card_indices) - 2):
            three_cards = card_indices[i: i + 3]
            if three_cards[2] - three_cards[0] <= 4:
                return 0.5 if len(state.public_cards) == 3 else 0.33 if len(state.public_cards) == 4 else 0.25
    return 0


def has_flush(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    return Counter([card[0] for card in all_cards]).most_common(1)[0][1] >= 5


def oppo_flush_possible(state: State) -> float:
    """
    Returns:
        1 if only 1 card required in hand
        0.5 if 2 cards required out of 4 unseen
        0.25 if 2 cards required out of 3 or 2 unseen
        0.1 if 3 cards required in unseen cards
        0 if not possible
    """
    if not state.public_cards:
        return 0

    cards_yet_to_come = 5 - len(state.public_cards)
    most_common_suit_count = Counter([card[0] for card in state.public_cards]).most_common(1)[0][1]
    max_poss_count = most_common_suit_count + cards_yet_to_come + 2
    if max_poss_count < 5 or most_common_suit_count == 1:
        # Impossible
        return 0

    if most_common_suit_count >= 4:
        return 1
    elif most_common_suit_count == 3:
        return 0.5 if cards_yet_to_come == 2 else 0.25
    elif most_common_suit_count == 2:
        return 0.1


def my_flush_possible(state: State) -> float:
    """
    Returns:
        1 if only 1 card needed at flop
        0.5 if 1 card needed turn
        0.25 if 2 cards needed flop
        0 if not possible
    """
    if not state.public_cards:
        return 0

    cards_yet_to_come = 5 - len(state.public_cards)
    all_cards = state.hand + state.public_cards
    most_common_suit_count = Counter([card[0] for card in all_cards]).most_common(1)[0][1]

    max_poss_count = most_common_suit_count + cards_yet_to_come
    if max_poss_count < 5:
        return 0
    elif most_common_suit_count >= 5:
        return 1
    elif most_common_suit_count == 4:
        return 1 if cards_yet_to_come == 2 else 0.5
    else:
        return 0.25


def has_full_house(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 3 and Counter([card[1] for card in all_cards]).most_common(2)[1][1] == 2


def has_four(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 4


def has_straight_flush(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    most_common = Counter([card[0] for card in all_cards]).most_common(1)[0] # (suit, count)
    flush_on = most_common[1] >= 5
    if not flush_on:
        return False

    suited_cards = [card for card in all_cards if card[0] == most_common[0]]
    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    card_indices = []
    for card in suited_cards:
        card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort

    for i in range(len(card_indices) - 4):
        five_cards = card_indices[i: i + 5]
        if five_cards[4] - five_cards[0] == 4:
            return True
    return False


def better_than_pair(state: State) -> bool:
    return has_three(state) or has_four(state) or has_full_house(state) or has_flush(state) or has_straight(state) or has_straight_flush(state)


############## Tom added functions ##################
def has_pair_in_hand(state: State) -> bool:
    all_cards = state.hand
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 2

def has_face_card_in_pot(state: State) -> bool:
    public_cards_vals = [card[1] for card in state.public_cards]
    return any(val in public_cards_vals for val in ["K", "Q", "J"])

def has_ace_in_pot(state: State) -> bool:
    public_cards_vals = [card[1] for card in state.public_cards]
    return "A" in public_cards_vals

def has_pair_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 2:
        return False
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 2

def has_two_pair_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 4:
        return False
    return (Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 2 and
            Counter([card[1] for card in all_cards]).most_common(2)[1][1] == 2)

def has_triplet_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 3:
        return False
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 3

def has_straight3_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 3:
        return False
    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    card_indices = []
    for card in all_cards:
         card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort

    for i in range(len(card_indices) - 2):
        three_cards = card_indices[i: i + 3]
        if three_cards[2] - three_cards[0] <= 4: # it's possible to have a straight with 2 hidden cards
            return True
    return False

def has_straight4_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 4:
        return False
    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    card_indices = []
    for card in all_cards:
         card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort

    for i in range(len(card_indices) - 3):
        four_cards = card_indices[i: i + 4]
        if four_cards[3] - four_cards[0] <= 4: # it's possible to have a straight with 1 hidden card
            return True
    return False

def has_straight_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 5:
        return False
    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    card_indices = []
    for card in all_cards:
         card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort

    for i in range(len(card_indices) - 4):
        five_cards = card_indices[i: i + 5]
        if five_cards[4] - five_cards[0] == 4:
            return True
    return False

def has_flush3_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 3:
        return False
    return Counter([card[0] for card in all_cards]).most_common(1)[0][1] == 3

def has_flush4_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 4:
        return False
    return Counter([card[0] for card in all_cards]).most_common(1)[0][1] == 4

def has_flush_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 5:
        return False
    return Counter([card[0] for card in all_cards]).most_common(1)[0][1] == 5

def has_full_house_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 5:
        return False
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 2 and Counter([card[1] for card in all_cards]).most_common(2)[1][1] == 2

def has_four_in_pot(state: State) -> bool:
    all_cards = state.public_cards
    if len(all_cards) < 4:
        return False
    return Counter([card[1] for card in all_cards]).most_common(1)[0][1] == 4

def has_straight_flush_in_pot(state: State) -> bool:
    return has_flush_in_pot(state) and has_straight_in_pot(state)

def has_straight3(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    if len(all_cards) < 3:
        return False
    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    card_indices = []
    for card in all_cards:
         card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort

    for i in range(len(card_indices) - 2):
        three_cards = card_indices[i: i + 3]
        if three_cards[2] - three_cards[0] <= 4: # it's possible to have a straight with 2 hidden cards
            return True
    return False

def has_straight4(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    if len(all_cards) < 4:
        return False
    order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # Ace is high and low
    card_indices = []
    for card in all_cards:
         card_indices += [order.index(card[1])] if card[1] != "A" else [0, 13]  # Add ace high and low
    card_indices = sorted(set(card_indices))  # Remove duplicates and sort

    for i in range(len(card_indices) - 3):
        four_cards = card_indices[i: i + 4]
        if four_cards[3] - four_cards[0] <= 4: # it's possible to have a straight with 1 hidden card
            return True
    return False

def has_flush3(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    if len(all_cards) < 3:
        return False
    return Counter([card[0] for card in all_cards]).most_common(1)[0][1] == 3

def has_flush4(state: State) -> bool:
    all_cards = state.hand + state.public_cards
    if len(all_cards) < 4:
        return False
    return Counter([card[0] for card in all_cards]).most_common(1)[0][1] == 4

############## Tom added functions ##################

def simple_policy(state: State) -> int:
    # if pre-flop, call/check
    if not state.public_cards:
        if state.opponent_chips_remaining == 0:
            # Oppo goes all-in
            if has_pair(state):
                return 1
            elif has_face_card_in_hand(state):
                return random.choice([0] * 3 + [1])
            else:
                # 90% chance of folding, sometimes call bluff
                return random.choice([0] * 9 + [1])
        else:
            # Wants to see flop
            return 1 if 1 in state.legal_actions and state else 0

    if has_pair(state) or better_than_pair(state):
        # Post-flop with a pair or better, bet any way possible
        return (
            3
            if 3 in state.legal_actions
            else 2
            if 2 in state.legal_actions
            else 1
        )
    else:
        # Limps in when it can
        return 0 if state.opponent_chips > state.player_chips else 1


def simple_policy_conservative(state: State) -> int:
    simple_pol_out = simple_policy(state)
    if simple_pol_out in [0, 1]:
        return simple_pol_out
    elif simple_pol_out == 2:
        return random.choice([1, 2, 2])
    elif simple_pol_out == 3:
        return random.choice([2 if 2 in state.legal_actions else 3, 3])
    else:
        return 3 if 3 in state.legal_actions else 2 if 2 in state.legal_actions else 1


def simple_policy_aggressive(state: State) -> int:
    # if pre-flop, call/check
    if not state.public_cards:
        return 3 if 3 in state.legal_actions else 2 if 2 in state.legal_actions else 1

    if has_pair(state) or better_than_pair(state) or has_face_card_in_hand(state):
        # Post-flop with a pair or face card in hand
        return (
            3
            if 3 in state.legal_actions
            else 4
            if 4 in state.legal_actions
            else 2
            if 2 in state.legal_actions
            else 1
        )
    else:
        # Limps in when it can
        return 0 if state.opponent_chips > state.player_chips else 1


def complex_policy(state: State) -> int:
    """Opponent policy to use in initial training."""
    check_better_than_pair = (has_two_pair(state)
                              or has_three(state)
                              or has_straight(state)
                              or has_flush(state)
                              or has_full_house(state)
                              or has_four(state)
                              or has_straight_flush(state))

    check_better_than_triplet = (has_straight(state)
                                 or has_flush(state)
                                 or has_full_house(state)
                                 or has_four(state)
                                 or has_straight_flush(state))

    # if pre-flop, call/check
    if not state.public_cards:
        ####################### extra stuff #######################
        if has_pair(state) or check_better_than_pair:
            # then sometimes randomly raise preflop
            if np.random.random() < 0.2:
                if 3 in state.legal_actions:
                    return 3
                if 2 in state.legal_actions:
                    return 2
            if np.random.random() < 0.01:
                if 4 in state.legal_actions:
                    return 4
        ####################### extra stuff #######################

        if state.opponent_chips_remaining == 0:
            # Oppo goes all-in
            if has_pair(state):
                if np.random.random() < 0.5:
                    return 1
            elif has_face_card_in_hand(state):
                return random.choice([0] * 3 + [1])
            else:
                # 90% chance of folding, sometimes call bluff
                return random.choice([0] * 9 + [1])
        else:
            # Sometimes randomly bet without good hand
            if np.random.random() < 0.05:
                if 2 in state.legal_actions:
                    return 2
            if np.random.random() < 0.05:
                if 3 in state.legal_actions:
                    return 3
            # Wants to see flop
            return 1 if 1 in state.legal_actions and state else 0

    # post flop
    # inject some randomness to bet even if you don't have a good hand
    if not (check_better_than_pair):
        if np.random.random() < 0.05:
            # bet any way possible
            return (
                3
                if 3 in state.legal_actions
                else 2
                if 2 in state.legal_actions
                else 1
                if 1 in state.legal_actions
                else 0
            )
    if has_pair(state) and not(check_better_than_pair):
        # inject some randomness to call even if you have a good hand
        if np.random.random() < 0.5:
            if 1 in state.legal_actions:
                return 1
        return (
            3
            if 3 in state.legal_actions
            else 2
            if 2 in state.legal_actions
            else 1
            if 1 in state.legal_actions
            else 0
        )
    if has_two_pair(state) or has_three(state):
        # inject some randomness to call even if you have a good hand
        if np.random.random() < 0.15:
            if 1 in state.legal_actions:
                return 1
        return (
            3
            if 3 in state.legal_actions
            else 2
            if 2 in state.legal_actions
            else 1
            if 1 in state.legal_actions
            else 0
        )
    if check_better_than_triplet:
        # go all in with some probability
        if np.random.random() < 0.05:
            if 4 in state.legal_actions:
                return 4
        # inject some randomness to call even if you have a good hand
        if np.random.random() < 0.05:
            if 1 in state.legal_actions:
                return 1
                # Post-flop with a pair or better, bet any way possible
        return (
            3
            if 3 in state.legal_actions
            else 2
            if 2 in state.legal_actions
            else 1
            if 1 in state.legal_actions
            else 0
        )
    # Other wise, we have literally nothing
    if state.opponent_chips == state.player_chips:
        # and opponent didn't bet.
        # Still sometimes bet to introduce randomness
        if np.random.random() < 0.1:
            return (2 if 2 in state.legal_actions else 1
                if 1 in state.legal_actions
                else 0)
        if np.random.random() < 0.05:
            return (3 if 3 in state.legal_actions else 1
                if 1 in state.legal_actions
                else 0)
        return (1
            if 1 in state.legal_actions
            else 0)
    else:  # opponent made a bet
        # Set up sometimes randomly calling when have nothing, and bet is small
        bet_size = state.opponent_chips - state.player_chips
        self_chips_left = state.player_chips_remaining
        if bet_size / self_chips_left < 0.2:
            if np.random.random() < 0.05:
                return (2 if 2 in state.legal_actions else 1
                    if 1 in state.legal_actions
                    else 0)
            if np.random.random() < 0.15:
                return (1
                    if 1 in state.legal_actions
                    else 0)
        elif bet_size / self_chips_left < 0.5:
            if np.random.random() < 0.05:
                return (1
                    if 1 in state.legal_actions
                    else 0)
        return 0

def complex_policy_conservative(state: State) -> int:
    simple_pol_out = complex_policy(state)
    if simple_pol_out in [0, 1]:
        return simple_pol_out
    elif simple_pol_out == 2:
        return random.choice([1, 2, 2])
    elif simple_pol_out == 3:
        return random.choice([2 if 2 in state.legal_actions else 3, 3])
    else:
        return 3 if 3 in state.legal_actions else 2 if 2 in state.legal_actions else 1


def complex_policy_agressive(state: State) -> int:
    simple_pol_out = complex_policy(state)
    if simple_pol_out == 0:
        # sometimes call anyways
        if np.random.random() < 0.1:
            if 1 in state.legal_actions:
                return 1
        return 0
    elif simple_pol_out == 1:
        # sometimes bet anyways
        if np.random.random() < 0.2:
            if 2 in state.legal_actions:
                return 2
        return simple_pol_out
    elif simple_pol_out == 2:
        # sometimes bet more anyways
        if np.random.random() < 0.2:
            if np.random.random() < 0.1:
                if 4 in state.legal_actions:
                    return 4
            if 3 in state.legal_actions:
                return 3
        return simple_pol_out
    elif simple_pol_out == 3:
        # sometimes go all in
        if np.random.random() < 0.05:
            if 4 in state.legal_actions:
                return 4
        return simple_pol_out
    else:
        return 3 if 3 in state.legal_actions else 2 if 2 in state.legal_actions else 1

def all_in_policy(state: State) -> int:
    return (4 if 4 in state.legal_actions else 3
                if 3 in state.legal_actions else 2 if 2 in state.legal_actions else 1)

def ultra_conservative_policy(state: State) -> int:
    return (2 if 2 in state.legal_actions else 1)

def can_check(state: State) -> bool:
    return state.player_chips == state.opponent_chips


def equal_stack(state: State) -> bool:
    return (state.player_chips + state.player_chips_remaining) == (state.opponent_chips + state.opponent_chips_remaining)


def bigger_stack(state: State) -> bool:
    return (state.player_chips + state.player_chips_remaining) > (state.opponent_chips + state.opponent_chips_remaining)


def much_bigger_stack(state: State) -> bool:
    return (state.player_chips + state.player_chips_remaining) >= (100 + state.opponent_chips + state.opponent_chips_remaining)


def much_smaller_stack(state: State) -> bool:
    return (state.player_chips + state.player_chips_remaining + 100) <= (state.opponent_chips + state.opponent_chips_remaining)


def opponent_all_in(state: State) -> bool: 
        # here this is conceptually whether I could win the entire game by winning the resulting hand. 
        # not just whether opponent has bet all the chips in their hand.
        # these are distinct. Opponent can bet whole stack, but if I have less than them, they're not really "all in"
        # but here this measures whether they're really all in (if I have a bigger stack than them, and they're all in)
    return state.opponent_chips_remaining == 0 if bigger_stack(state) else False


def feature_input(state: State, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Returns a 1D array of features for the state"""
    nn_input = torch.zeros(55, dtype=torch.float32, device=device)

    # Hand features
    nn_input[0] = low_card_in_hand(state)
    nn_input[1] = med_card_in_hand(state)
    nn_input[2] = has_face_card_in_hand(state)
    nn_input[3] = has_ace_in_hand(state)

    # Pair/Three features
    nn_input[4] = has_pair(state)
    nn_input[5] = pair_incl_hand_card(state)
    nn_input[6] = pair_is_highest(state)
    nn_input[7] = has_two_pair(state)
    nn_input[8] = has_three(state)

    # Straight features
    nn_input[9] = my_straight_possible(state)
    nn_input[10] = oppo_straight_possible(state)
    nn_input[11] = has_straight(state)

    # Flush features
    nn_input[12] = my_flush_possible(state)
    nn_input[13] = oppo_flush_possible(state)
    nn_input[14] = has_flush(state)

    # Top tier hand features
    nn_input[15] = has_full_house(state)
    nn_input[16] = has_four(state)
    nn_input[17] = has_straight_flush(state)

    # Chips raw
    nn_input[18] = state.player_chips / 100
    nn_input[19] = state.opponent_chips / 100
    nn_input[20] = (state.player_chips_remaining / 100) - 1
    nn_input[21] = (state.opponent_chips_remaining / 100) - 1

    # Chip features
    nn_input[22] = bool(can_check(state))
    nn_input[23] = bool(equal_stack(state))
    nn_input[24] = bool(bigger_stack(state))
    nn_input[25] = bool(much_bigger_stack(state))
    nn_input[26] = bool(much_smaller_stack(state))
    nn_input[27] = bool(opponent_all_in(state))

    # Stage features
    nn_input[28] = state.stage == 0
    nn_input[29] = state.stage == 1  # Flop
    nn_input[30] = state.stage == 2  # Turn
    nn_input[31] = state.stage == 3  # River

    # Tom added features
    nn_input[32] = (state.player_chips + 1e-13) / (state.player_chips
                                                   + state.player_chips_remaining + 1e-13)  # my frac bet
    nn_input[33] = (state.opponent_chips + 1e-13) / (state.opponent_chips + state.opponent_chips_remaining + 1e-13)
    nn_input[34] = (state.player_chips + state.opponent_chips) / 200.0  # fraction of chips in play
    nn_input[35] = suited_hand(state)
    nn_input[36] = has_pair_in_hand(state)
    nn_input[37] = has_face_card_in_pot(state)
    nn_input[38] = has_ace_in_pot(state)
    nn_input[39] = has_pair_in_pot(state)
    nn_input[40] = has_two_pair_in_pot(state)
    nn_input[41] = has_triplet_in_pot(state)
    nn_input[42] = has_straight_in_pot(state)
    nn_input[43] = has_flush_in_pot(state)
    nn_input[44] = has_full_house_in_pot(state)
    nn_input[45] = has_four_in_pot(state)
    nn_input[46] = has_straight_flush_in_pot(state)
    nn_input[47] = has_straight3_in_pot(state)
    nn_input[48] = has_straight4_in_pot(state)
    nn_input[49] = has_flush3_in_pot(state)
    nn_input[50] = has_flush4_in_pot(state)
    nn_input[51] = has_straight3(state)
    nn_input[52] = has_straight4(state)
    nn_input[53] = has_flush3(state)
    nn_input[54] = has_flush4(state)

    return nn_input