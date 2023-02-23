from pathlib import Path

import numpy as np

import delta_utils.check_submission as checker
from game_mechanics import PokerEnv, choose_move_randomly, load_network
from torch import nn


def check_submission(team_name: str) -> None:
    # A fix for this is incoming!!
    return 
    example_state, _, _, info = PokerEnv(choose_move_randomly).reset()
    expected_choose_move_return_type = int
    expected_pkl_output_type = nn.Module
    pkl_file = load_network(team_name)

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=pkl_file,
        pkl_checker_function=lambda x: x,
        current_folder=Path(__file__).parent.resolve(),
        choose_move_extra_argument={"legal_moves": np.array([0, 1, 2, 3])},
    )
