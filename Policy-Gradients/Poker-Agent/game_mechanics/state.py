from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class State:
    """
    hand: List[str] - The cards in your hand
    public_cards: List[str] - Face-up cards on the table
    player_chips: int - The number of chips you have put in the pot
    opponent_chips: int - The number of chips your opponent has put in the pot
    player_chips_remaining: int - The total number of chips in the player's stack (not including chips in pot)
    opponent_chips_remaining: int - The total number of chips your opponent has remaining
    stage: int - The stage of the game
    legal_actions: List[int] - The legal actions you can take
    """
    @staticmethod
    def from_dict(state_dict: Dict):
        assert (
            len(state_dict["all_chips"]) == 2
        ), "State class doesn't support games of more than 2 players!"
        player = state_dict["all_chips"].index(state_dict["my_chips"])
        opponent = 1 - player
        return State(
            hand=state_dict["hand"],
            public_cards=state_dict["public_cards"],
            player_chips=state_dict["my_chips"],
            opponent_chips=state_dict["all_chips"][opponent],
            player_chips_remaining=state_dict["stakes"][player],
            opponent_chips_remaining=state_dict["stakes"][opponent],
            stage=state_dict["stage"].value,
            legal_actions=[action.value for action in state_dict["legal_actions"]],
        )

    hand: List[str]
    public_cards: List[str]
    player_chips: int
    opponent_chips: int
    player_chips_remaining: int
    opponent_chips_remaining: int
    stage: int
    legal_actions: List[int]


def to_basic_nn_input(state: State) -> torch.Tensor:
    """Convert a state to a basic neural network input. Expresses the state as a 1D tensor of length
    55 where:

    - The first 52 elements are the cards visible to the player.
      If the card is in the player's hand it is 1.
      If it is in the public cards it is -1
    - The next 2 elements are the player's chips and opponent's chips
      that have been bet on that hand (i.e. the pot) (normalised between -1 and 1)
    - The final element is the total number of chips the player has remaining in the game
      (normalised between -1 and 1)
    """
    nn_input = torch.zeros(55)
    suits = ["C", "D", "H", "S"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    for card in state.hand:
        nn_input[13 * suits.index(card[0]) + ranks.index(card[1])] = 1
    for card in state.public_cards:
        nn_input[13 * suits.index(card[0]) + ranks.index(card[1])] = -1
    nn_input[52] = state.player_chips / 100 - 1
    nn_input[53] = state.opponent_chips / 100 - 1
    nn_input[54] = (state.player_chips_remaining / 100) - 1
    return nn_input
