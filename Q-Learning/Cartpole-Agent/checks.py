import torch
from torch import nn
from time import sleep
import math
from env import CartPoleEnv


def test_solution(network):

    # Render does not work on replit.
    # Download your solution locally to see your
    # network in action!
    render = True

    env = CartPoleEnv()
    assert isinstance(
        network, nn.Module
    ), "Your solution should be a torch neural network!"
    state, reward, done, _ = env.reset()
    if render:
        env.render()

    n_steps = 0
    while not done:
        q_values = network(torch.tensor(state))
        action = torch.argmax(q_values).item()
        state, _, done, _ = env.step(action)
        if render:
            env.render()
        sleep(0.05)
        n_steps += 1

    assert (
        n_steps >= 200
    ), f"Oh no the pole fell after {n_steps} steps, you need a better network!"
    print(f"Congratulations, you kept the pole up for {n_steps} steps! You're a cartpole champion!")
