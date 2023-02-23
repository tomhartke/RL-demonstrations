# Cartpole-Agent

The Cartpole-Agent implements Q learning.
More specifically it includes:
- Deep Q learning (double network for slowing down the update rate)
- Experience replay to randomly sample the past set of actions, and not just learn from recent experience.
- Otherwise, just pretty basic Q learning.

For this agent, all training and evaluation was done in a Jupyter notebook called "main.ipynb".

The notebook also keeps track of the training process (updates to network weights and biases, 
and the input/output values to layers) to visualize the training process.

At the end, you can run the visualization to see it perform.