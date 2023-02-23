import inspect
import time
from pathlib import Path
from typing import Callable

from delta_utils import get_discrete_choose_move_out_checker
from delta_utils.check_submission import check_submission as _check_submission
from game_mechanics import BOARD_SIZE, GoEnv, choose_move_randomly, load_pkl


def check_submission(team_name: str, choose_move_no_network: Callable) -> None:
    example_state, _, _, _ = GoEnv(choose_move_randomly).reset()
    pkl_file = load_pkl(team_name)

    max_time = 1
    t1 = time.time()
    choose_move_no_network(
        example_state,
    )
    t2 = time.time()
    assert (
        t2 - t1 < max_time
    ), f"Your choose_move_no_network function took {t2 - t1} seconds to run, which is longer than the maximum allowed time of {max_time} seconds."

    mcts = import_mcts()

    return _check_submission(
        example_choose_move_input={"state": example_state, "mcts": mcts, "pkl_file": pkl_file},
        check_choose_move_output=get_discrete_choose_move_out_checker(
            possible_outputs=list(range(BOARD_SIZE**2))
        ),
        current_folder=Path(__file__).parent.resolve(),
    )


def import_mcts():
    try:
        MCTS = getattr(__import__("main", fromlist=["None"]), "MCTS")

        return MCTS()
    except AttributeError as e:
        raise ImportError("No MCTS found in file main.py") from e
    except TypeError as e:
        print(
            "Failed to initialize MCTS class, __init__ should only take self as a non-default argument."
        )
        raise
