from pathlib import Path

import delta_utils.check_submission as checker
from torch import nn

from game_mechanics import PongEnv, load_network


def check_submission(team_name: str) -> None:
    example_state, _, _, _ = PongEnv().reset()
    expected_choose_move_return_type = int
    game_mechanics_expected_hash = (
        "649cdae65301019d56379b82605c9cd5ee43da3a2d854b0117788f8985749aea"
    )
    expected_pkl_output_type = nn.Module

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=load_network(team_name),
        pkl_checker_function=lambda x: x,
        game_mechanics_hash=game_mechanics_expected_hash,
        current_folder=Path(__file__).parent.resolve(),
    )
