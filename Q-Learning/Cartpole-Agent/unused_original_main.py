import torch 
from torch import nn
from checks import test_solution
import random
from env import CartPoleEnv
from memory import ExperienceReplayMemory
from tqdm import tqdm
import numpy as np


"""
You should train a pytorch network with DQN & Experience 
Replay that takes a cartpole state (numpy array with 
shape (4,)) as an input and has two output nodes representing
the value of each action (0 and 1).
"""
env = CartPoleEnv()

# Hyperparameters - given you some to narrow the search space
gamma = 0.9
epsilon = 0.05
batch_size = 50  # Can be lowered if this is slow to run
lr = 0.005
max_num_episodes = 1000  # Shouldn't need more than this
max_memory_size = 100000  # This shouldn't matter

n_neurons = 64

Q = nn.Sequential(
    nn.Linear(4, n_neurons),
    nn.ReLU(),
    nn.Linear(n_neurons, 2),
)

optim = torch.optim.Adam(Q.parameters(), lr=lr)
loss_fn = nn.MSELoss()
memory = ExperienceReplayMemory(max_length=max_memory_size)

num_wins, num_losses, num_steps, ep_losses = 0, 0, [], []
moving_avg = 0

for episode_num in tqdm(range(max_num_episodes)):
    # This moving average is an early stopping condition: aim is 
    #  200 steps, so average of 200 with epsilon-greedy will solve
    if moving_avg > 200:
        break
    # Counts number of steps this episode
    n_steps = 0
    successor_state, reward, done, _ = env.reset()

    while not done:
        prev_state = successor_state.copy()
        if np.random.random() < epsilon:
            action = np.random.randint(0, 1)
        else:
            with torch.no_grad():
                action = torch.argmax(Q(torch.tensor(successor_state))).item()

        successor_state, reward, done, _ = env.step(action)

        memory.append(prev_state, action, reward, successor_state, done)

        if len(memory.rewards) >= batch_size:
            s1, a1, r1, s2, is_terminals = memory.sample(batch_size)

            # Update steps
            q_values_chosen_1 = Q(s1)[range(len(s1)), a1]
            with torch.no_grad():
                chosen_successor_action = torch.argmax(Q(s2), dim=1)
                max_q_successor = (
                    Q(s2)[range(len(s2)), chosen_successor_action] * ~is_terminals
                )

            loss = loss_fn(q_values_chosen_1, r1 + gamma * max_q_successor)

            # Updates the parameters!
            optim.zero_grad()
            loss.backward()
            optim.step()

        n_steps += 1

    # Used in debugging
    num_steps.append(n_steps)
    moving_avg = 0.9 * moving_avg + 0.1 * n_steps

    # Debugging
    if episode_num % 10 == 9:
        print("Avg steps:", sum(num_steps) / len(num_steps))
        ep_losses, num_steps = [], []
    

# Test your solution below: over 200 timesteps is completion!
test_solution(Q)
