# Delta Academy version of PettingZoo go_base. Originally from tensorflow

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code adapted from: https://github.com/tensorflow/minigo

"""A board is a NxN numpy array. A Coordinate is a tuple index into the board. A Move is a
(Coordinate c | None). A PlayerMove is a (Color, Move) tuple.

(0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.
"""
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Set, Tuple, Union

import numpy as np

from .utils import (
    BLACK,
    BOARD_SIZE,
    DIAGONALS,
    EMPTY,
    MAX_NUM_MOVES,
    NEIGHBORS,
    PASS_MOVE,
    UNKNOWN,
    WHITE,
    IllegalMove,
    PlayerMove,
    int_to_coord,
)

if TYPE_CHECKING:
    from .state import State


def place_stones(board: np.ndarray, color: int, stones: Iterable[Tuple[int, int]]) -> None:
    for s in stones:
        board[s] = color


def pass_move(state: "State", mutate: bool = False):
    pos = state if mutate else copy.deepcopy(state)
    assert pos.board_deltas is not None
    pos.recent_moves += (PlayerMove(pos.to_play, PASS_MOVE),)
    pos.board_deltas = [np.zeros([1, BOARD_SIZE, BOARD_SIZE], dtype=np.int8)] + pos.board_deltas[:6]
    pos.to_play *= -1
    pos.ko = None
    return pos


def find_reached(
    board: np.ndarray, coord: Tuple[int, int]
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Finds the stones that are connected to the stone at 'coord'. Retuns two sets of coordinates:

    - chain: the chain of stones of the same color that are connected to coord
    - reached: the empty spaces / stones of the opposite color that are adjacent to the chain
    """
    color = board[coord]
    chain = {coord}
    reached = set()
    frontier = [coord]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        for n in NEIGHBORS[current]:
            if board[n] == color and n not in chain:
                frontier.append(n)
            elif board[n] != color:
                reached.add(n)
    return chain, reached


def is_koish(board: np.ndarray, coord: Tuple[int, int]) -> Optional[int]:
    """Check if coord is surrounded on all sides by 1 color, and return that color."""
    if board[coord] != EMPTY:
        return None
    neighbors = {board[n] for n in NEIGHBORS[coord]}
    if len(neighbors) == 1 and EMPTY not in neighbors:
        return list(neighbors)[0]
    else:
        return None


def is_eyeish(board: np.ndarray, coord: Optional[Tuple[int, int]]):
    """Check if coord is an eye, for the purpose of restricting MC rollouts."""
    # pass is fine.
    if coord is None:
        return
    color = is_koish(board, coord)
    if color is None:
        return None
    diagonal_faults = 0
    diagonals = DIAGONALS[coord]
    if len(diagonals) < 4:
        diagonal_faults += 1
    for d in diagonals:
        if board[d] not in (color, EMPTY):
            diagonal_faults += 1
    return None if diagonal_faults > 1 else color


@dataclass
class Group:
    """
    stones: a frozenset of Coordinates belonging to this group
    liberties: a frozenset of Coordinates that are empty and adjacent to this group.
    color: color of this group
    """

    id: int
    stones: frozenset
    liberties: frozenset
    color: int

    def __eq__(self, other: object) -> bool:
        return (
            (
                self.stones == other.stones
                and self.liberties == other.liberties
                and self.color == other.color
            )
            if isinstance(other, Group)
            else NotImplemented
        )


def is_move_legal(
    move: Optional[Tuple[int, int]], board: np.ndarray, ko: Optional[Tuple[int, int]] = None
) -> bool:
    """Checks that a move is on an empty space, not on ko, and not suicide."""
    if move is None:
        return True
    return False if board[move] != EMPTY else move != ko


ALL_MOVES = np.ones([BOARD_SIZE, BOARD_SIZE], dtype=bool)


def all_legal_moves(board: np.ndarray, ko: Optional[Tuple[int, int]]) -> np.ndarray:
    "Returns a np.array of size go.N**2 + 1, with 1 = legal, 0 = illegal"
    # by default, every move is legal
    legal_moves = np.copy(ALL_MOVES)
    # ...unless there is already a stone there
    legal_moves[board != EMPTY] = False

    # ...and retaking ko is always illegal
    if ko is not None:
        legal_moves[ko] = False
    # Concat with pass move
    return np.arange(BOARD_SIZE**2 + 1)[legal_moves.ravel().tolist() + [True]]


def play_move(state, move: int, color=None, mutate=False):
    # Obeys CGOS Rules of Play. In short:
    # No suicides
    # Chinese/area scoring
    # Positional superko (this is very crudely approximate at the moment.)
    if color is None:
        color = state.to_play

    pos = state if mutate else copy.deepcopy(state)

    coord = int_to_coord(move)
    if coord is None:
        pos = pass_move(state, mutate=mutate)
        return pos

    if not is_move_legal(coord, state.board, state.ko):
        raise IllegalMove(
            f'{"Black" if state.to_play == BLACK else "White"} coord at {to_human_readable(coord)} is illegal: \n{state}'
        )

    potential_ko = is_koish(state.board, coord)

    place_stones(pos.board, color, [coord])
    captured_stones = pos.lib_tracker.add_stone(color, coord)
    place_stones(pos.board, EMPTY, captured_stones)

    opp_color = color * -1

    new_board_delta = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.int8)
    new_board_delta[coord] = color
    place_stones(new_board_delta, color, captured_stones)

    if len(captured_stones) == 1 and potential_ko == opp_color:
        new_ko = list(captured_stones)[0]
    else:
        new_ko = None

    pos.ko = new_ko
    pos.recent_moves += (PlayerMove(color, move),)

    # keep a rolling history of last 7 deltas - that's all we'll need to
    # extract the last 8 board states.
    pos.board_deltas = [new_board_delta.reshape(1, BOARD_SIZE, BOARD_SIZE)] + pos.board_deltas[:6]
    pos.to_play *= -1
    return pos


def game_over(recent: Tuple[PlayerMove, ...]) -> bool:
    two_passes = len(recent) >= 2 and recent[-1].move == PASS_MOVE and recent[-2].move == PASS_MOVE
    turn_limit_reached = len(recent) >= MAX_NUM_MOVES
    return two_passes or turn_limit_reached


def fill_empty_board_spaces(board: np.ndarray) -> np.ndarray:
    """Returns a copy of 'board' with all the unassigned spaces filled in with the color that
    occupies the territory."""
    working_board = np.copy(board)
    while EMPTY in working_board:
        unassigned_spaces = np.where(working_board == EMPTY)
        c = unassigned_spaces[0][0], unassigned_spaces[1][0]
        territory, borders = find_reached(working_board, c)
        border_colors = {working_board[b] for b in borders}
        X_border = BLACK in border_colors
        O_border = WHITE in border_colors
        if X_border and not O_border:
            territory_color = BLACK
        elif O_border and not X_border:
            territory_color = WHITE
        else:
            territory_color = UNKNOWN  # dame, or seki
        place_stones(working_board, territory_color, territory)
    return working_board


def score(board: np.ndarray, komi: float, return_both_colors: bool = False) -> Union[float, Tuple]:
    """Return score from Black's perspective.

    If White is winning, score is negative.
    return_both_colors: set to true to return a tuple of (black_score, white_score)
                        rather than white score = -black_score
    """

    working_board = fill_empty_board_spaces(board)
    if return_both_colors:
        return (
            np.count_nonzero(working_board == BLACK) - komi,
            np.count_nonzero(working_board == WHITE),
        )
    return (
        np.count_nonzero(working_board == BLACK) - np.count_nonzero(working_board == WHITE) - komi
    )


def result(board: np.ndarray, komi: float) -> int:
    """Return 1 if black wins, -1 if white wins, 0 if draw or the game is not over."""
    score_ = score(board, komi, return_both_colors=False)
    assert isinstance(score_, float)  # Make sure you haven't got the tuple
    if score_ > 0:
        return 1
    elif score_ < 0:
        return -1
    else:
        return 0


ALPHA_ROW = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def to_human_readable(coord: Tuple[int, int]) -> str:
    """Converts from a Minigo coordinate to a human readbale coordinate."""
    if coord is None:
        return "pass"
    y, x = coord
    return f"{ALPHA_ROW[x]}{BOARD_SIZE - y}"
