from delta_poker.game_mechanics import PokerEnv, State, choose_move_randomly


def raiser(state: State) -> int:
    """Always raises."""
    return 3


def test_fold_loses():
    env = PokerEnv(opponent_choose_move=raiser)
    env.reset()
    # Raise
    _, reward, done, _ = env.step(3)
    assert reward == 0

    # Fold
    _, reward, done, _ = env.step(0)
    assert reward < 0


def test_random_games_returns():
    env = PokerEnv(verbose=True)
    for _ in range(100):
        state, reward, done, info = env.reset()
        total_reward = reward

        while not env.done:
            action = choose_move_randomly(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if action == 0:
                assert reward <= 0

        assert total_reward in {-100, 100}


def test_raise_returns():
    env = PokerEnv(lambda state: 3 if 3 in state.legal_actions else 1)
    for _ in range(10):
        state, reward, done, info = env.reset()
        total_reward = reward

        while not env.done:
            state, reward, done, info = env.step(3 if 3 in state.legal_actions else 1)
            total_reward += reward

        assert total_reward in {-100, 100}
