from typing import NamedTuple, Optional, Tuple

import numpy as np

BOARD_SIZE = 9

# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# This means that swapping colors is as simple as multiplying array by -1.
WHITE, EMPTY, BLACK, FILL, _, UNKNOWN = range(-1, 5)


MAX_NUM_MOVES = 2 * BOARD_SIZE**2 + 2


class PlayerMove(NamedTuple):
    """A hashable class representing a move made by a player.
    Can be used as a dictionary key.
    I.e the following is valid:
        d: Dict[PlayerMove, int] = {PlayerMove(color=1, move=2): 100}

    Args:
        color: BLACK or WHITE
        move: integer representing the move made
    """

    color: int
    move: int


# Represents "group not found" in the LibertyTracker object
MISSING_GROUP_ID = -1

ALL_COORDS = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
EMPTY_BOARD = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)


PASS_MOVE = BOARD_SIZE**2


def _check_bounds(coord: Tuple[int, int]) -> bool:
    return 0 <= coord[0] < BOARD_SIZE and 0 <= coord[1] < BOARD_SIZE


NEIGHBORS = {
    (x, y): list(filter(_check_bounds, [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]))
    for x, y in ALL_COORDS
}
DIAGONALS = {
    (x, y): list(
        filter(
            _check_bounds,
            [(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)],
        )
    )
    for x, y in ALL_COORDS
}


class IllegalMove(Exception):
    pass


def int_to_coord(move: int) -> Optional[Tuple[int, int]]:
    """Converts an integer move to a coordinate.

    Our choose_move() function outputs an integer and these are  converted to a tuple of (x, y)
    coordinates which is used by  go_base.
    """
    return None if move == PASS_MOVE else (move // BOARD_SIZE, move % BOARD_SIZE)
