from pathlib import Path

from delta_utils import get_discrete_choose_move_out_checker
from delta_utils.check_submission import check_submission as _check_submission


def check_submission() -> None:
    return _check_submission(
        example_choose_move_input={"state": [], "pkl_file": {}},
        check_choose_move_output=get_discrete_choose_move_out_checker(
            possible_outputs=list(range(8))
        ),
        current_folder=Path(__file__).parent.resolve(),
    )
