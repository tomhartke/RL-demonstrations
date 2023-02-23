"""
This class simply controls who the opponent is during training.

The opponent is reselected at the start of each episode based on the
 agent's win rate. The opponent is chosen as the agent who has the best
 win rate against the training agent, although there are some exploratory
 games to check agents who we haven't played against very often before.

More possible opponents are added to the pool as the agent trains - old
 versions of the agent are added.
"""

from collections import deque
import random
from typing import Callable, List, Optional

import numpy as np

from game_mechanics import State


Opponent = Callable[[State], int]


def calculate_expl_term(big_n: int, small_n: int, const: float) -> float:
    """
    This function is related to the upper confidence bound algorithm
        big_n is the number of games played in memory
        small_n is the number of games played against a specific opponent. 
        constant is some scale factor for how often we explore
        1/sqrt(small_n) makes sense, but not sure why log(big_n) is in there?
        Ah. I think big_n is there to make you explore more in late game.
    """
    return const * np.sqrt(np.log(big_n) / small_n) if small_n > 0 else 10


def convert_to_losses(score: int) -> int:
    return 1 if score == -1 else 0


class OpponentPool:
    """
    Aim of this is to have a pool of opponents that can be used to train against.

    We want to train more against opponents that we perform poorly against, with some
        'exploratory' match-ups when we've only played a small number of games against
        an opponent.
        
    Choice of opponent is using upper confidence bound reasoning: the more often
        we play against an opponent, the uncertainty in our loss rate goes down by
        1/sqrt(number of times we played them). We choose the next opponent as the
        one with the highest loss rate + 1/sqrt(number) correction, to choose highly
        uncertain loss rates too. 
    """

    def __init__(
        self,
        opponents: Optional[List[Opponent]] = None,
        memory_maxlen: int = 10000,
        exploration_const: float = 0.75,
    ):
        self.opponents = opponents or []
        self.opponent_memory = deque(maxlen=memory_maxlen) 
            # list of (opponent,loss/no loss)

        self.loss_counts = np.zeros(len(self.opponents))
        self.visit_counts = np.zeros(len(self.opponents))
        self.explore_terms = np.zeros(len(self.opponents)) + 10
            # gets combined with prob_losses to determine who to play

        self.exploration_const = exploration_const

    def add_opponent(self, opponent: Opponent) -> None:
        """When you checkpoint an old model, add it to the pool"""
        self.opponents.append(opponent)

        self.loss_counts = np.append(self.loss_counts, 0)
        self.visit_counts = np.append(self.visit_counts, 0)
        self.explore_terms = np.append(self.explore_terms, 0)

    def record_score(self, opponent_id: int, score: int) -> None:
        """After each game, record the score, so we can use win-rate to select opponents"""
        loss_score = convert_to_losses(score)
        self.opponent_memory.append((opponent_id, loss_score))
        self.visit_counts[opponent_id] += 1
        self.loss_counts[opponent_id] += loss_score

        big_n = len(self.opponent_memory)
        small_n = self.visit_counts[opponent_id]
        self.explore_terms[opponent_id] = calculate_expl_term(
            big_n, small_n, self.exploration_const
        )

    def refresh_opponent_records(self):
        """Call this every 100 games or so to correctly update the win-rates"""
        self.loss_counts = np.zeros(len(self.opponents))
        self.visit_counts = np.zeros(len(self.opponents))
        for opponent_id, loss_score in self.opponent_memory:
            self.loss_counts[opponent_id] += loss_score
            self.visit_counts[opponent_id] += 1

        big_n = len(self.opponent_memory)
        for opponent_id in range(len(self.opponents)):
            small_n = self.visit_counts[opponent_id]
            self.explore_terms[opponent_id] = calculate_expl_term(
                big_n, small_n, self.exploration_const
            )

    @property
    def prob_losses(self) -> np.ndarray:
        return self.loss_counts / (self.visit_counts + 1e-8)

    def get_opponent(self, opponent_id: int) -> Opponent:
        return self.opponents[opponent_id]

    def select_opponent(self) -> int:
        """Select an opponent to play against"""
        ucb_values = self.prob_losses + self.explore_terms
        max_value = np.max(ucb_values)
        max_values = np.where(ucb_values == max_value)[0]
        return max_values[int(random.random() * len(max_values))]