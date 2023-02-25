# AlphaGo Zero Implementation

## Summary
This is an example implementation of Monte-Carlo tree search (MCTS) guided by a learned 
neural network which estimates the optimal actions to take (policy network) and the
expected returns (value network) in the AlphaGo Zero (AGZ) framework. 

It has essentially everything in the AlphaGo Zero paper (https://doi.org/10.1038/nature24270).
* For details, refer to that paper, but I will describe the basic architecture below.
* The only things missing from this implementation that are in the paper (as far as I know), are
  * This code only runs one Go game at a time (and is set up for 1 GPU). For really good training, I think we 
  would have to parallelize training for multiple games at a time, to get enough games. 
  * This code is only for a 9x9 board instead of 19x19, and the depth of the neural network is a bit smaller than the AGZ paper.

  
## The game of Go
This agent learns to play the board game Go (https://en.wikipedia.org/wiki/Go_(game)).
For details of the rules, see Wikipedia.

The basic idea is that it is a board game, 
and the agent can see where all of its tiles are placed on the board, and the opponents
tiles, and then decides where to play the next tile. 
* There is no hidden information
* The game rules are deterministic, and the agent knows them (ie it can plan ahead). 
There are similar versions of this algorithm where the agent also learns the rules of the game over time,
but I am not implementing that here.

Example snapshot of a self play game:
![](images/ExampleGoGame.jpg)

## Main files
All relevant files are in the folder "main".
The master file is main.py. 
Various aspects of the network and training are also implemented in files:
agzNetwork.py, mcts_class.py, and opponent_pool.py. 

To run the network and observe a game, you need only run main.py.

To train the network, again you need only run main.py, but with the "training_now" variable inside 
the main section set to True instead of False. (You will need to train on a GPU to get any reasonable speed.)

Within main.py, there are a few other options to be aware of:
* You can set variables to choose whether to train now, whether to load a previous version for training,
and which version to load. 
  * I made it this way so that I could do training in stages on a GPU, and then stop and restart later.
* If you want to play against the bot, you can change the arguments to "play_go" near the bottom
of main.py to be "your_choose_move=human_player" (and turn off training).

## The setup
### The network

In brief, the architecture is:
1. The agent sees the state of the board, and past 8 turns. 
   * These are fed in as a bunch of images to the convolutional network.
2. A convolutional network with many layers and residual connections extracts features
    * The network is defined in the file agzNetwork.py
    * It is a multi-layer convolutional network with Batch normalization and 
   residual connections, exactly as in the AGZ paper (though the network size is different, and kernel number).
    * Followed by two fully-connected heads (policy and value)
4. These features are fed to a policy network head and a value network head (one network, two heads)
4. The policy network predicts all possible actions with certain probabilities (and certain illegal moves are masked)
5. The value network predicts the expected returns. 


### Monte-Carlo tree search
This is standard Monte-Carlo tree search, guided by the output of the policy network.
The tree search is implemented through a python Class in a file called mcts_class.py. 

Roughly, what it does is:

* We observe the game state, choose an action based on essentially the upper confidence bound prediction
for the expected returns
  * Exploration is encouraged in the usual way by adding an "upper confidence bound" style term to nodes, which
  is larger for unexplored nodes. 
  * As in the AGZ paper, this exploration term is guided by the policy network (it is made larger for actions 
  that are predicted by the policy network)
* We search the tree using this algorithm until we find an unexplored node, then expand, 
and query the value network for the expected returns.
* Then we back up the value returned by the value network to the base of the tree search, and begin again. 

During training, the policy of the root node is made a bit more random by introducing Dirichlet-distributed noise. 

We usually do around 100 rollouts per action choice. 

### The training process 

We train by playing against an opponent pool of past selves. For details see the opponent_pool.py file.

We augment the training data by reflecting and flipping the input boards (taking advantage of the
symmetries of the board.)

The network is trained in a supervised way to predict the value and actions during self play.
Past games are sampled in a random way (bootstrapping) and trained repeatedly. 

### Pitfalls 
I originally masked out illegal actions wrong, forcing bad actions. 
Specifically, I originally made it only possible to pass when there are 0 other options. 
This is bad because there needs to be a way for the game to end so that we aren’t 
forced to fill in our own squares (sometimes passing is optimal, when you have an option to play). 
If we fill in our own squares because we're forced to not pass, that can remove protection from being 
surrounded by the opponent (removing "eyes" in the language of Go) and 
cause the final outcome of the game to be pretty random (basically whoever happens to win once all eyes are filled). 
This prevents effective training. 

One must be careful about signs of terms in the game mechanics (who is which player, and what perspective
is the reward evaluated from). Similarly, one has to be careful about 
sign flips in the MCTS perspective when choosing which action to take in simulated planning. 

I found it was useful to predict the actual action probabilities of the network, rather than discrete action actually chosen
(which depends on whether we have a high temperature or a low temperature for the conversion of MCTS visit count 
to action probability). This ensures that we learn to act more randomly at the beginning of the game 
(because of the added dirichlet noise) rather than converging to some randomly chosen initial action 
via the policy network. It also seems necessary to add the Dirichlet noise only to the set of masked possible
actions (not to all 82 possible actions equally). This ensures you get more randomization/search later in the game,
as the set of available options dwindles (and therefore each gets more relative noise).

The PUCT exploration coefficient is very important. I noticed that having it too high freezes in 
the original (random) policy, since the exploration term in MCTS always dominates, and you only explore the existing policy. That is not good. 
A higher level description of why this happens can be found in this paper (https://arxiv.org/abs/2007.12509),
which describes the AGZ MCTS algorithm as regularized tree search, with a penalty for deviating from the policy network
(and the coefficient of that penalty is the PUCT coefficient, roughly, so that high 
PUCT coefficient penalizes deviating from the initial policy).

In a similar way, the number of rollouts has to be more than something like 
10 otherwise you will just blindly follow whatever policy is currently present, 
which will reinforce that policy.

Be mindful of bottlenecks in the speed of the code. I had to change some of the mechanics of the MCTS
node class to reduce computation time. 

It is useful to improve sample efficiency by reflecting/flipping the boards during training, and treating
that as independent samples. 

Be careful about sampling the games too much so that samples are correlated. 
Your action samples need to be pretty independent 
to avoid collapsing the policy to suboptimal things. 
For example, I ended up using batch training over last 15 games to get more uncorrelated 
samples for training, and then trained every 5 games on 2.5 games worth of data. 

A good way to debug is to: 
Print which player the bot is (black or white) and predicted Q value of the chosen action after MCTS, 
while watching games run in real time.
Seeing this predicted value correlate with visual swings up or down in who will win the game shows learning. 


### Suggestions for steps to rebuild this code from scratch
Here is a list of the steps I planned to take and eventually took, 
roughly in order, to build this code. 

I started with the basic game mechanics, then just made sure that behaved correctly given the outputs of the network.
1. Then I set up value function training just based on the outcome of random games (and checked slight learning). 
2. Then I set up MCTS to get a slightly improved policy (and could see that improved things). 
I just used the UCT algorithm at first, instead of policy-guided UCT.
3. Then I set up the policy network to predict the action given past data from games (and checked learning). 
4. Then I added the policy network into the MCTS algorithm to guide search. 
5. Then I made the opponent pool and checkpointing past versions of self.
6. Then I made sure the runtime wasn't particularly bad on a GPU.
7. Then I tuned the hyperparameters to optimize learning, including:
    * Gamma for reward decay within episodes (not important, though at one point it seemed that large decay helped, 
   since it reduced variance in the early-game learned predicted reward.)
    * PUCT exploration constant (not too large, or random policy freezes)
    * Network structure (depth and kernel number, wasn't too important)
    * Learning rates (didn't explore significantly, perhas I should be annealing of learning rate?)
    * L2 regularization during learning (didn't explore significantly)
    * Number of rollouts during training (must be long, even if it's slow, otherwise there's no policy 
   improvement loop through self play)
    * Exploration rate during training 
      * Temperature for exploration of first N plays (didn't explore much) 
      * Noise level for noise added to policy network root node during mcts (didn't explore much)
    * Batch size for training (how many games between training sessions) (has to be sufficiently non-correlated samples)
    * Self play number of games before checkpoint (didn't explore much)
    * Exploration constant of opponent pool (didn't explore much)

### Future optimization
Probably the biggest gain would be to train longer and parallelize the training process (more games).
The original AGZ paper had 5 million training games (I think implying 1500 games run in parallel over 3 days).
I only had a few hundred training games (since each takes a minute or two).

I didn't get a chance to really explore hyperparameter optimization. Probably the best approaches are: 
* Try learning rate decay 
* Try adjusted puct coefficient
* Try adjusting rollout number 
* Maybe more fully connected value neurons in the value head of the network
* Deeper network (more layers or more kernels)

There's definitely still room for improvement, since 
visual inspection/play against human shows that the agent still 
hasn't learned to block the opponent in crucial situations, or to 
connect it's own groups when crucial (likely these phenomena are correlated, since within
the self-play framework, there's no feedback for them to learn to block the opponent (which is themselves) until
the opponent learns to make crucial connection moves).