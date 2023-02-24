# Space Shooter - Proximal Policy Optimization

## Summary
This is an example of using policy gradients to solve a two-player game, with many
degrees of freedom and potentially complex optimal strategies.
Given only the current state of the board (positions of players), 
we train a neural network policy for the actions to take. 

Details of reinforcement learning strategies included:
* Proximal policy optimization (clip gradient updates in some cases)
  * A learned baseline for the value function to reduce variance.
  * Entropy regularization of the action probabilities to encourage exploration
* Generalized advantage estimation of the returns to stabilize updates. 
* A pool of opponents to play against from past learning to ensure global strategy convergence.

## The game
Two players live in an arena, each controlling a space ship.
Each ship can turn and move forward or sideways, and shoot bullets.

The goal is the shoot the opponent before being shot. 

> The arena at the start of the game
![Space shooter starting position](images/space-shooter.png)
> The arena midgame, with players moving and shooting bullets
![Space shooter starting position](images/SpaceShooterIngame.jpg)
Agents learn to hide behind barriers and duck out to shoot.

## Main files
Basically everything is in the delta_shooter folder, in the main.py file. 
This is the file used for training the agent. If you want to retrain the agent, 
you should just have to run main.py

A second file, run_test_model.py is a more simple file which loads the network, and 
plays the final trained bot against itself for a few games.
It should be able to be directly run if you just want to see the agent play.

## The architecture and training process
All the basic things are the same as the Poker agent.
* Policy gradients with a learned baseline, and entropy regularization. 
* There is an opponent pool of trained agents (checkpointed over time) which the 
agent plays against. 
* Many things are logged in tensorboard (action probabilities, update magnitudes, rewards, etc)

In addition, a few aspects of the training make this a stronger agent:
1. We use generalized advantage estimation to set the expected returns from a given state (lamda return).
This lets us tradeoff the high variance/quick update of Monte-Carlo returns with a few step TD-learning look ahead. 
2. Instead of a normal policy gradient, we use proximal policy optimization. 
   * For each training step, we gather data from a number of games. 
   * Then we run a batch of gradient descent updates. If during those updates, 
   a certain predicted policy probability changes too much from the anchor policy, 
   then it gets dropped from the loss function. 
   * This stabilizes the learning to not change the policy too drastically.   
3. The training process is split into multiple stages to gradually allow the agent
to learn certain skills.
   1. First we learn on a small board, with no obstacles/barriers 
      (this teaches the agent to at least turn and shoot the other agent).
   2. Then we go to a full size board.
   3. Then we introduce the barriers. 

## Takeaways 
Main takeaway is that getting the agent to initially learn can be hard,
and you need sufficiently detailed readouts to check if certain strategies are being learned. 
You also need to have a sufficiently complex network to capture the game mechanics 
(it's better to start out with too complex of a network that is slow to learn, than to 
start with a too small network which isn't possible to learn enough. 
Detecting slow learning is possible, but figuring out why nothing works at all is harder.)

A specific quirk of space shooter is the existence of the barriers in the arena.
I think without introducing the barriers later in training (ie if they were introduced immediately),
the agent would really never learn anything, because there would be no end of game, and no reward. 

However, once learning happens, using an opponent pool is a very powerful way to generate strategies that
perform well against general opponents. Here I think the strategy continually improved over time.