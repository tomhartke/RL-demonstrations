import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pygame
from torch import nn

from .go_base import (
    BLACK,
    BOARD_SIZE,
    PASS_MOVE,
    WHITE,
    all_legal_moves,
    game_over,
    int_to_coord,
    is_move_legal,
    play_move,
    result,
    score,
)
from .render import render_game
from .state import State

HERE = Path(__file__).parent.resolve()
MAIN_PATH = HERE.parent / "main"


ALL_POSSIBLE_MOVES = np.arange(BOARD_SIZE**2 + 1)

# Visuals
SCREEN_SIZE = (500, 500)

# The komi to use is much debated. 7.5 seems to
# generalise well for different board sizes
# lifein19x19.com/viewtopic.php?f=15&t=17750
# 7.5 is also the komi used in alpha-go vs Lee Sedol
# (non-integer means there are no draws)

KOMI = 7.5

############# Functions useful for MCTS ###############


def transition_function(state: State, action: int) -> State:
    """Returns the state that would be reached by taking 'action' in 'state'.""" ""
    return play_move(state, action, state.to_play)


def reward_function(state: State) -> int:
    """Returns the reward that received for black in the current state.

    {0, 1,-1}.
    """
    assert state.board is not None
    return result(state.board, KOMI) if game_over(state.recent_moves) else 0


def is_terminal(state: State) -> bool:
    """Returns True if the game is over, False otherwise."""
    return game_over(state.recent_moves)


def play_go(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier: float = 1.0,
    render: bool = True,
    verbose: bool = False,
) -> float:

    env = GoEnv(
        opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    state, reward, done, info = env.reset()
    while not done:
        action = your_choose_move(state=state)
        state, reward, done, info = env.step(action)
    return reward


class GoEnv:
    def __init__(
        self,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 1.0,
    ):
        """As in other environments, the step() function takes two steps.

        Hence the functions above should be used for MCTS
        """

        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        self.state = State()

        if render:
            self.init_visuals()

    def init_visuals(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("Go")
        self._render_game()

    def _render_game(
        self,
    ) -> None:
        render_game(self.state.board, screen=self.screen)

    @property
    def reward(self) -> int:
        if self.player_color == BLACK:
            return reward_function(self.state)
        else:
            return reward_function(self.state) * -1

    @property
    def done(self) -> bool:
        return is_terminal(self.state)

    def reset(self, player_black: bool = False) -> Tuple[State, float, bool, Dict]:

        # 1 is black and goes first, white is -1 and goes second
        self.player_color = BLACK if player_black else random.choice([BLACK, WHITE])
        self.color_str = "Black" if self.player_color == BLACK else "White"

        self.state = State()

        if self.verbose:
            print(
                f"Resetting Game.\nYou are playing with the {self.color_str} tiles.\nBlack plays first\n\n"
            )

        if self.state.to_play != self.player_color:
            self._step(
                self.opponent_choose_move(state=self.state),
            )

        return self.state, self.reward, self.done, {}

    def move_to_string(self, move: int) -> str:

        assert self.state.board is not None
        N = self.state.board.shape[0]
        if move == N**2:
            return "passes"
        return f"places counter at coordinate: {(move//N, move%N)}"

    def __str__(self) -> str:
        return str(self.state.board) + "\n"

    def _step(self, move: int) -> None:

        if self.verbose:
            name = "player" if self.state.to_play == self.player_color else "opponent"
            print(f"{name} {self.move_to_string(move)}")

        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert is_move_legal(
            int_to_coord(move), self.state.board, self.state.ko
        ), f"{move} is an illegal move"

        self.state = transition_function(self.state, move)

        if self.render:
            self._render_game()

    def step(self, move: int) -> Tuple[State, int, bool, Dict]:

        assert self.state.to_play == self.player_color
        self._step(move)

        if not self.done:
            self._step(self.opponent_choose_move(state=self.state))

        if self.verbose and self.done:
            self.nice_prints()  # Probably not needed

        return self.state, self.reward, self.done, {}

    def nice_prints(self):
        print(
            f"\nGame over. Reward = {self.reward}.\n"
            f"Player was playing as {self.color_str}\n"
            f"Black has {np.sum(self.state.board==1)} counters.\n"
            f"White has {np.sum(self.state.board==-1)} counters.\n"
            f"Your score is {self.player_color * score(self.state.board, KOMI)}.\n"
        )


def choose_move_randomly(state: State) -> int:
    legal_moves = all_legal_moves(state.board, state.ko)
    return legal_moves[int(random.random() * len(legal_moves))]


def choose_move_pass(state: State) -> int:
    """Always pass."""
    return PASS_MOVE


def load_pkl(team_name: str, network_folder: Path = MAIN_PATH) -> nn.Module:
    net_path = network_folder / f"{team_name}_file.pkl"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    with open(net_path, "rb") as handle:
        file = pickle.load(handle)
    return file


def save_pkl(file: Any, team_name: str) -> None:
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = MAIN_PATH / f"{team_name}_file.pkl"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            with open(net_path, "wb") as handle:
                pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            load_pkl(team_name)
            return
        except Exception:
            if attempt == n_retries - 1:
                raise


# Need to know the default screen size from petting zoo to get which square is clicked
# Will not work with a screen override
SQUARE_SIZE = SCREEN_SIZE[0] // BOARD_SIZE
LEFT = 1
RIGHT = 3


def pos_to_coord(pos: Tuple[int, int]) -> Tuple[int, int]:
    """Used in human_player only."""

    col = pos[0] // SQUARE_SIZE
    row = pos[1] // SQUARE_SIZE
    return row, col


def coord_to_int(coord: Tuple[int, int]) -> int:
    return coord[0] * BOARD_SIZE + coord[1]


def human_player(state: State) -> int:

    print("\nYour move, click to place a tile!")
    legal_moves = all_legal_moves(state.board, state.ko)
    if len(legal_moves) == 1:
        print("You have no legal moves, so you pass")
        return legal_moves[0]

    while True:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
                coord = pos_to_coord(pygame.mouse.get_pos())
                action = coord_to_int(coord)
                if action in legal_moves:
                    return action
            elif event.type == pygame.MOUSEBUTTONUP and event.button == RIGHT:
                return PASS_MOVE
