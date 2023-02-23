import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import pygame

from game_mechanics.render import (
    MOVE_MAP,
    draw_both_chip_stacks,
    draw_possible_actions,
    get_screen,
    get_screen_subsurface,
    render,
    wait_for_click,
)
from game_mechanics.state import State
from game_mechanics.utils import choose_move_randomly
from rlcard.games.nolimitholdem import Action
from rlcard.games.nolimitholdem.game import NolimitholdemGame


class PokerEnv:
    STARTING_MONEY = 100

    def __init__(
        self,
        opponent_choose_move: Callable = choose_move_randomly,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 1.0,
    ):
        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        self.game = NolimitholdemGame()
        self.player_total = self.opponent_total = self.STARTING_MONEY

        # Will be flipped on every hand reset
        self.dealer = random.choice([0, 1])

        # Want to get rid of this
        self.player_agent = 0
        self.opponent_agent = 1

        self.most_recent_move: Dict[int, Optional[str]] = {
            self.player_agent: None,
            self.opponent_agent: None,
        }

        self.env_reset = False
        if render:
            pygame.init()
            self._font = pygame.font.SysFont("arial", 18)
            self._screen = get_screen()
            self.subsurf = get_screen_subsurface(self._screen)
            self._clock = pygame.time.Clock()

    @property
    def turn(self) -> int:
        return self.game.game_pointer

    @property
    def player_state(self) -> State:
        """
        The internal state is a rich dictionary.
        We represent it to users as a State object

        Leave it up to users how to represent it to the network
        "hand": [card1, card2], in format S2 for 2 of spades (Suit, Rank) where 10 is T
            (A, 2 -> 9, T, J, Q, K)
        "public_cards": [card1, card2, card3, card4, card5]
        "all_chips": [player_chips, opponent_chips]
        "my_chips": [player_chips]
        "legal_actions": [0, 1, 2, 3, 4] as Action enum objects
        "stakes": [player_chips_remaining, opponent_chips_remaining]
        "current_player": 0 or 1
        "pot": sum of all chips in pot
        "stage": 0, 1, 2, 3, 4
        """

        return State.from_dict(self.game.get_state(self.player_agent))

    @property
    def opponent_state(self) -> State:
        return State.from_dict(self.game.get_state(self.opponent_agent))

    @property
    def done(self) -> bool:
        return self.game_over

    @property
    def hand_done(self) -> bool:
        return self.game.is_over()

    @property
    def legal_moves(self) -> List[int]:
        return [action.value for action in self.game.get_legal_actions()]

    @property
    def game_over(self) -> bool:
        return self.player_total <= 0 or self.opponent_total <= 0

    def reset(self) -> Tuple[State, float, bool, Dict]:
        """Reset the whole round."""
        self.env_reset = True
        self.player_total = self.opponent_total = self.STARTING_MONEY
        if self.verbose:
            print("\nNew round, resetting chips to 100 each")
        return self.reset_hand()

    @property
    def reward(self) -> float:
        return self.game.get_payoffs()[self.player_agent] if self.hand_done else 0

    def reset_hand(self) -> Tuple[State, float, bool, Dict]:
        """Reset game to the next hand, persisting chips."""
        reward = 0.0

        # Persist the game over screen if rendering until reset
        if self.render and self.game_over:
            return self.player_state, 0, True, {}

        self.dealer = 1 - self.dealer

        game_config = {
            "game_num_players": 2,
            "player_chips": [self.player_total, self.opponent_total],
            "dealer_id": self.dealer,
        }
        self.game.configure(game_config)
        self.game.init_game()

        self.most_recent_move = {
            self.player_agent: None,
            self.opponent_agent: None,
        }

        # Which elements of the obs vector are in the hand?
        # Probably don't need these variables
        self.hand_idx: Dict[int, List] = {
            self.player_agent: [],
            self.opponent_agent: [],
        }

        if self.verbose:
            print("starting game")

        # Take a step if opponent goes first, so step() starts with player
        if self.turn == self.opponent_agent:
            opponent_move = self.opponent_choose_move(state=self.opponent_state)
            self._step(opponent_move)
            if self.hand_done:
                reward += self.complete_hand()

        if self.render:
            # If the opponent folds on the first hand, win message
            win_message = f"You won {int(abs(self.reward))} chips" if self.done else None
            self.render_game(render_opponent_cards=win_message is not None, win_message=win_message)

        return self.player_state, reward, self.done, {}

    def print_action(self, action: int) -> None:

        player = "Player" if self.turn == self.player_agent else "Opponent"
        if action not in self.legal_moves:
            print(f"{player} made an illegal move: {action}")
        else:
            print(f"{player} {MOVE_MAP[action]}")

    def update_most_recent_move(self, move: int) -> None:
        pot_size = self.player_state.player_chips + self.player_state.opponent_chips
        if move == 2:
            move_str = f"raise {pot_size//2}"
        elif move == 3:
            move_str = f"raise {pot_size}"
        else:
            move_str = MOVE_MAP[move]

        self.most_recent_move[self.turn] = move_str

    def _step(self, move: int) -> None:
        assert self.env_reset, "You need reset the environment before taking your first step!"
        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert move in self.legal_moves, f"{move} is an illegal move"

        self.update_most_recent_move(move)

        if self.verbose:
            self.print_action(move)

        self.game.step(Action(move))
        # assert prev_turn != self.turn, "Turn did not change!"

        if self.render:
            self.render_game(render_opponent_cards=False, win_message=None)

    def step(self, move: int) -> Tuple[State, float, bool, Dict]:
        assert self.turn == 0, "Not your turn to move! This is an internal error, please report it."

        self._step(move)

        while self.turn == 1 and not self.hand_done:
            self._step(
                self.opponent_choose_move(state=self.opponent_state),
            )

        if self.hand_done:
            reward = self.complete_hand()
            return self.player_state, reward, self.done, {}

        return self.player_state, self.reward, self.done, {}

    def complete_hand(self) -> float:
        """Finishes a hand and resets, does not reset the whole env as the episod is only over when
        one player runs out of chips.

        Need to store the reward before resetting as this changes self.reward
        """

        # Store as will be changed by self.reset_hand()
        reward = self.reward

        self.player_total += int(reward)
        self.opponent_total -= int(reward)

        if reward == 0:
            win_messsage = "Draw!"
        else:
            result = "won" if reward > 0 else "lost"
            win_messsage = f"You {result} {int(abs(reward))} chips"

        if self.verbose:
            print(win_messsage)

        if self.render:
            self.render_game(render_opponent_cards=True, win_message=win_messsage)
            wait_for_click()

        if not self.game_over:
            _, extra_rew, __, ___ = self.reset_hand()
            reward += extra_rew

        return reward

    def render_game(
        self,
        render_opponent_cards: bool = False,
        win_message: Optional[str] = None,
    ) -> None:

        self._screen.fill((7, 99, 36))  # green background
        render(
            player_states={"player": self.player_state, "opponent": self.opponent_state},
            most_recent_move=self.most_recent_move,
            render_opponent_cards=render_opponent_cards,
            win_message=win_message,
            screen=self.subsurf,
            continue_hands=not self.game_over,
            turn=self.turn,
        )

        self.draw_additional()

    def draw_additional(self) -> None:

        draw_both_chip_stacks(
            self._screen,
            self.player_total - self.player_state.player_chips,
            self.opponent_total - self.opponent_state.player_chips,
            self.STARTING_MONEY * 2,
        )

        # Need to store the current raise size on the buttons, so can set it
        # to most recent action
        draw_possible_actions(self._screen, self._font, self.player_state, turn=self.turn)

        pygame.display.update()
        self._clock.tick(int(self.game_speed_multiplier))
        if self.game_speed_multiplier < 1:
            time.sleep((1 / self.game_speed_multiplier) - 1)
