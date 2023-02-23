import numpy as np

from delta_poker.game_mechanics.state import State, to_basic_nn_input
from rlcard.games.base import Card


def test_to_basic_nn_input():

    # Order the cards go in
    # suits = ["C", "D", "H", "S"]
    # ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

    state = State(
        hand=["HA", "SA"],
        public_cards=[],
        player_chips=100,
        opponent_chips=100,
        player_chips_remaining=500,
        opponent_chips_remaining=500,
        stage=0,
        legal_actions=[0, 1, 2, 3, 4],
    )
    nn_input = to_basic_nn_input(state)
    nn_input = nn_input.numpy()
    assert np.sum(nn_input[:52]) == 2
    assert nn_input[51] == 1
    assert nn_input[51 - 13] == 1
    assert nn_input[52] == 0
    assert nn_input[53] == 0
    assert nn_input[54] == 4
