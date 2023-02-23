import sys
from pathlib import Path

from .go_base import all_legal_moves, score
from .go_env import (
    KOMI,
    GoEnv,
    choose_move_pass,
    choose_move_randomly,
    human_player,
    is_terminal,
    load_pkl,
    play_go,
    reward_function,
    save_pkl,
    transition_function,
)
from .state import State
from .utils import BLACK, BOARD_SIZE, EMPTY, MAX_NUM_MOVES, PASS_MOVE, WHITE
