import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .liberty_tracker import LibertyTracker
from .utils import BLACK, EMPTY_BOARD, PlayerMove


@dataclass
class State:
    """Describes the State of a game of Go.

    Args:
        board: [BOARD_SIZE, BOARD_SIZE] np array of ints
        recent_moves: a tuple of PlayerMoves, such that recent[-1] is the last move.
        to_play: BLACK or WHITE
        player_color: Keeps track of white color (BLACK or WHITE) you are playing as

        ko: a tuple (x, y) if the last move that was a ko, or None if no ko
        board_deltas: a np.array of shape (n, go.N, go.N) representing changes
            made to the board at each move (played move and captures).
            Should satisfy next_pos.board - next_pos.board_deltas[0] == pos.board
        lib_tracker: a LibertyTracker object. Used for caching available liberties for speedup.
                    Gives a speedup of 5x!
    """

    board: Optional[np.ndarray] = None
    recent_moves: Tuple[PlayerMove, ...] = tuple()
    to_play: int = BLACK

    ko: Optional[Tuple[int, int]] = None
    board_deltas: Optional[List[np.ndarray]] = None
    lib_tracker: LibertyTracker = LibertyTracker.from_board(board)

    def __deepcopy__(self, memodict: Dict) -> "State":
        new_board = np.copy(self.board)
        new_lib_tracker = copy.deepcopy(self.lib_tracker)
        return State(
            board=new_board,
            lib_tracker=new_lib_tracker,
            ko=self.ko,
            recent_moves=self.recent_moves,
            to_play=self.to_play,
            board_deltas=self.board_deltas,
        )

    def __post_init__(self):
        if self.board_deltas is None:
            self.board_deltas = []
        self.board = np.copy(EMPTY_BOARD) if self.board is None else self.board
