import copy
from typing import Dict, Optional, Set

import numpy as np

from .go_base import Group, find_reached, place_stones
from .utils import BLACK, BOARD_SIZE, EMPTY, FILL, MISSING_GROUP_ID, NEIGHBORS, WHITE


class LibertyTracker:
    @staticmethod
    def from_board(board: np.ndarray) -> "LibertyTracker":
        board = np.copy(board)
        curr_group_id = 0
        lib_tracker = LibertyTracker()
        for color in (WHITE, BLACK):
            while color in board:
                curr_group_id += 1
                found_color = np.where(board == color)
                coord = found_color[0][0], found_color[1][0]
                chain, reached = find_reached(board, coord)
                liberties = frozenset(r for r in reached if board[r] == EMPTY)
                new_group = Group(curr_group_id, frozenset(chain), liberties, color)
                lib_tracker.groups[curr_group_id] = new_group
                for s in chain:
                    lib_tracker.group_index[s] = curr_group_id
                place_stones(board, FILL, chain)

        lib_tracker.max_group_id = curr_group_id

        liberty_counts = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.uint8)
        for group in lib_tracker.groups.values():
            num_libs = len(group.liberties)
            for s in group.stones:
                liberty_counts[s] = num_libs
        lib_tracker.liberty_cache = liberty_counts

        return lib_tracker

    def __init__(
        self,
        group_index: Optional[np.ndarray] = None,
        groups: Optional[Dict] = None,
        liberty_cache: Optional[np.ndarray] = None,
        max_group_id: int = 1,
    ):
        """This class is used only for caching to speed up computions, it does not need to be
        understood to build a solution.

        It keeps track of the existing liberties (en.wikipedia.org/wiki/Rules_of_Go#Liberties) on
        the board so they do not have to be recomputed before each move
        """

        # group_index: a NxN numpy array of group_ids. -1 means no group
        # groups: a dict of group_id to groups
        # liberty_cache: a NxN numpy array of liberty counts
        self.group_index = (
            group_index
            if group_index is not None
            else -np.ones([BOARD_SIZE, BOARD_SIZE], dtype=np.int32)
        )
        self.groups = groups or {}
        self.liberty_cache = (
            liberty_cache
            if liberty_cache is not None
            else np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=np.uint8)
        )
        self.max_group_id = max_group_id

    def __deepcopy__(self, memo):
        new_group_index = np.copy(self.group_index)
        new_lib_cache = np.copy(self.liberty_cache)
        # shallow copy
        new_groups = copy.copy(self.groups)
        return LibertyTracker(
            new_group_index,
            new_groups,
            liberty_cache=new_lib_cache,
            max_group_id=self.max_group_id,
        )

    def add_stone(self, color, c):
        assert self.group_index[c] == MISSING_GROUP_ID
        captured_stones = set()
        opponent_neighboring_group_ids = set()
        friendly_neighboring_group_ids = set()
        empty_neighbors = set()

        for n in NEIGHBORS[c]:
            neighbor_group_id = self.group_index[n]
            if neighbor_group_id != MISSING_GROUP_ID:
                neighbor_group = self.groups[neighbor_group_id]
                if neighbor_group.color == color:
                    friendly_neighboring_group_ids.add(neighbor_group_id)
                else:
                    opponent_neighboring_group_ids.add(neighbor_group_id)
            else:
                empty_neighbors.add(n)

        one_move_remaining = np.sum(self.liberty_cache == 0) == 1
        if not one_move_remaining:
            board_becomes_full = False
        else:
            no_liberties = np.where(self.liberty_cache == 0)
            board_becomes_full = no_liberties[0] == c[0] and no_liberties[1] == c[1]

        # handle suicides
        is_suicide = not empty_neighbors and (
            not friendly_neighboring_group_ids
            or all(len(self.groups[fr].liberties) == 1 for fr in friendly_neighboring_group_ids)
            and not board_becomes_full
        )

        new_group = self._merge_from_played(
            color, c, empty_neighbors, friendly_neighboring_group_ids
        )

        # new_group becomes stale as _update_liberties and
        # _handle_captures are called; must refetch with self.groups[new_group.id]
        for group_id in opponent_neighboring_group_ids:
            neighbor_group = self.groups[group_id]
            if len(neighbor_group.liberties) == 1:
                captured = self._capture_group(group_id)
                captured_stones.update(captured)
            else:
                self._update_liberties(group_id, remove={c})

        if not captured_stones and is_suicide:
            captured_stones = self._capture_group(new_group.id)
        self._handle_captures(captured_stones)

        return captured_stones

    def _merge_from_played(self, color, played, libs, other_group_ids):
        stones = {played}
        liberties = set(libs)
        for group_id in other_group_ids:
            other = self.groups.pop(group_id)
            stones.update(other.stones)
            liberties.update(other.liberties)

        if other_group_ids:
            liberties.remove(played)
        assert stones.isdisjoint(liberties)
        self.max_group_id += 1
        result = Group(self.max_group_id, frozenset(stones), frozenset(liberties), color)
        self.groups[result.id] = result

        for s in result.stones:
            self.group_index[s] = result.id
            self.liberty_cache[s] = len(result.liberties)

        return result

    def _capture_group(self, group_id):
        dead_group = self.groups.pop(group_id)
        for s in dead_group.stones:
            self.group_index[s] = MISSING_GROUP_ID
            self.liberty_cache[s] = 0
        return dead_group.stones

    def _update_liberties(self, group_id, add: Optional[Set] = None, remove: Optional[Set] = None):
        add = add or set()
        remove = remove or set()
        group = self.groups[group_id]
        new_libs = (group.liberties | add) - remove
        self.groups[group_id] = Group(group_id, group.stones, new_libs, group.color)

        new_lib_count = len(new_libs)
        for s in self.groups[group_id].stones:
            self.liberty_cache[s] = new_lib_count

    def _handle_captures(self, captured_stones):
        for s in captured_stones:
            for n in NEIGHBORS[s]:
                group_id = self.group_index[n]
                if group_id != MISSING_GROUP_ID:
                    self._update_liberties(group_id, add={s})
