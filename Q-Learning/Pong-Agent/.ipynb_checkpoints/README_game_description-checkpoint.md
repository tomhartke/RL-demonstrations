# Competition Finale: Pong  :ping_pong:

![The basic setup of Pong.](./images/pongGame.jpeg)

## Rules of Pong :scroll:

Pong is a **two-player** video game. Players control a paddle at one end of the arena. 

The player has three options: move up, move down, or stay still.

A ball travels across the arena. You move your paddle to hit the ball back.

This game is **very challenging** to train an RL algorithm for! However, it is absolutely possible to train an agent that performs better than you do!

# Competition Rules :scroll:
1. You must build a **Deep Reinforcement Learning** agent (using a neural network)
2. You can only write code in `main.py`
    - You can only store data to be used in a competition in a `.pt` file by `save_network()`
    - In the competition, your agent will call the `choose_move()` function in `main.py` to select a move (`choose_move()` may call other functions in `main.py`)
    - Any code not in `main.py` will not be used.    
2. Submission deadline: **5pm UTC, Sunday**.
    - You can update your code after submitting, but **not after the deadline**.
    - **Check your submission is valid with `check_submission()`**

## Competition Format :crossed_swords:

The competition is a knockout tournament where your AI will play other teams' AIs 1-v-1. 

Each 1-v-1 matchup consists of a single **first to 5 points** game 

The competition & discussion will be in [Gather Town](https://app.gather.town/app/nJwquzJjD4TLKcTy/Delta%20Academy) at **6pm UTC on Sunday** (60 mins after submission deadline)!

Since this is the final week, this week's points in the competition **count double**!

1. First place gets 20 points :trophy:
2. Second 18 points
3. Third: 16, Fourth: 14, 5th: 12
4. **Everyone else who competes gets 10 points** :muscle:

## Technical Details :hammer:

### Need to Know


🚨🚨🚨 The `states_per_step` input to `PongEnv` on initialization controls the speed of the game (how far the balls and paddles move in one timestep). It's highly recommended to increase this greater than `1` during training to reduce the number of timesteps between rewards. Be aware that the tournament will be played with this variable set to `1` 🚨🚨🚨.


### Rewards :moneybag:

You receive `+10` for winning and `-10` for losing. 

You also recieve `+1` every time you successfully hit the ball with the paddle.

### Observations each timestep :mag:
The **tuple** returned from each `env.step()` has:
- A **numpy array** of length 6 describing the current game state:
    - `ball_position_x`: the current x coordinate of the ball
    - `ball_position_y`: the current y coordinate of the ball
    - `ball_direction`: the angle of the ball’s motion, relative to the y-axis, measured clockwise (start pointing up, then rotate clockwise until you reach the direction) **in degrees**
    - `paddle_position`: the y-position of the center of your paddle
    - `opponent_paddle_position`: The y-position of the center of your opponent’s paddle
    - `ball_speed`: The speed of the ball. This increases each time it hits a paddle.
- The reward for each timestep
- Whether the point is done (boolean)
- Extra information

## Court Layout :world_map:
    
The court is `COURT_WIDTH = 900` wide (along x-axis) and `COURT_HEIGHT = 600` tall (along y-axis).
    
**Positions:** bottom left corner is at `(0, 0)`, the bottom right corner is at `(COURT_WIDTH, 0)` and the top right corner is at `(COURT_WIDTH, COURT_HEIGHT)`.
    
**Your paddle** (the left paddle in green) moves up and down along the y axis at `x = 0` (on the left hand side!).
    
**The opponent’s paddle** (the right paddle in grey) moves up and down along the y axis at `x = 900` (on the right hand side!).

Both paddles are `PADDLE_HEIGHT = 120` tall.

The ball has radius `BALL_RADIUS = 5`.

![Pong with all the inputs to `choose_move()` annotated](/images/pong_annotation.png
)
### Controlling the Paddle :tennis:
    
You control the left-hand paddle and have 3 possible moves:
    
1. `1` Move up 👆
2. `0` Stay still ✋
3. `-1` Move down 👇

The distance you move up or down by in the competition is `1`, but during training, you can change how long is between each step with the `steps_per_state` parameter, input to `PongEnv` at initialization.

### Ball Mechanics :baseball:

If the ball hits the wall at the top or bottom of the court, it bounces off.

The ball’s speed is fixed and increases slightly every time it bounces off **a paddle**, but not the top or bottom of the court.
    
When bouncing off the top or bottom of the court, the angle of incidence equals the angle of reflection

![Angles when the ball bounces off the top or bottom of the court](images/anglee.png)

## Paddle Mechanics
    
**2 rules** govern how the ball bounces off the paddle:
    
1. Every time the ball bounces off the paddle, the **ball speed increases** slightly
2. The **location on the paddle** that the ball bounces **affects the angle it leaves the paddle at**


### There are **3 regions** on the paddle:

### 1. **Sweet Spot** :lollipop:
(blue below) - ball bounces off at a pretty straight angle (perpendicular to the paddle)! This area is `2 * SWEET_SPOT_RADIUS` in size, where `SWEET_SPOT_RADIUS = 10`

### 2. **Corners** :blue_square:
(red below) - ball bounces off at a large angle! (this extends beyond the paddle as the ball has a radius of `BALL_RADIUS` as well)


### 3. **Rest of paddle** 🏓
(black below) - angle of incidence equals angle of reflection (like bouncing off the walls)

#### All 3 have some randomness added to the angle!


![Paddle regions shown](/images/paddle.png)


## Functions you write :point_left:

<details>
<summary><code style="white-space:nowrap;">  train()</code></summary>
Write this to train your network from experience in the environment. 
<br />
<br />
Return the trained network so it can be saved.
</details>
<details>
<summary><code style="white-space:nowrap;">  choose_move()</code></summary>
This acts greedily given the state and network.

In the competition, the choose_move() function is called to make your next move. Takes the state as input and outputs an action.

</details>

## Existing Code :pray:
    
<details>
<summary><code style="white-space:nowrap;">  Env</code> class</summary>
The environment class controls the game and runs the opponent. It should be used for training your agent.
<br />
<br />
See example usage in <code style="white-space:nowrap;">play_pong()</code>.
<br />
<br />
The opponent's <code style="white-space:nowrap;">choose_move</code> function is input at initialisation (when <code style="white-space:nowrap;">Env(opponent_choose_move)</code> is called). Every time you call <code style="white-space:nowrap;">Env.step()</code>, both moves are taken - yours your opponent's. Then the ball moves - if it hits a paddle then it will bounce off. Your opponent sees a 'mirrored' version of the arena, so from each player's perspective, the arena mechanics are the same. The <code style="white-space:nowrap;">env</code> also has 
    <br />
    <br />

The env also has a  <code style="white-space:nowrap;">render</code> arugment, whether or not to render the game graphically.
    
<code style="white-space:nowrap;">  Env.step()</code> has a <code style="white-space:nowrap;">  verbose</code> arguments which prints the game to the console when set to <code style="white-space:nowrap;">True</code>. Verbose visualises the ongoing progress of the game in the console. The bat controlled by your choose move function is on the left hand side, and your opponent is on the right.
</details>

<details>
<summary><code style="white-space:nowrap;">  robot_choose_move()</code></summary>
A basic pong bot that moves the bat up if the ball is above it, and down if the ball is below it. Your first goal is to learn to beat this robot!
<br />
<br />
Takes the state as input and outputs an action.
</details>

<details>
<summary><code style="white-space:nowrap;">  play_pong()</code></summary>
Plays a game of Pong, which can be rendered through pygame in the console (if <code style="white-space:nowrap;">render=True</code>) and printed to the terminal (if <code style="white-space:nowrap;">verbose=True</code>). 

You can play against your own bot if you set <code style="white-space:nowrap;">your_choose_move</code> to <code style="white-space:nowrap;">human_player</code>!
<br />
<br />
Inputs:
    
<code style="white-space:nowrap;">your_choose_move</code>: Function that takes the state and outputs the action for your agent.

<code style="white-space:nowrap;">opponent_choose_move</code>: Function that takes the state and outputs the action for the opponent.

<code style="white-space:nowrap;">num_points</code>: How many points before game over.

<code style="white-space:nowrap;">game_speed_multiplier</code>: controls the gameplay speed. High numbers mean fast games, low numbers mean slow games.


<code style="white-space:nowrap;">verbose</code>: whether to print to console each move and the corresponding board states.

<code style="white-space:nowrap;">render</code>: whether to render the match through pygame
</details>


## Suggested Approach :+1:

1. Discuss which RL algorithm to use and what the network should look like
2. **Write `train()`**, borrowing from past exercises
3. **Print out important values** - otherwise bugs in your code may slip through the cracks :astonished:
4. Ponder what you want to set `timesteps_per_step` to. What are the tradeoffs when it's really large? What about when it's really small? Think about how this might affect the values and how gamma's (the time discount factor) meaning might change.
5. Think about what behaviour maximizes the reward function. Based on this, what rewards should you give your agent during training?
6. Test out how fast it is to train in the environment and think about whether online or episodic updates are better. Perhaps play with both if you have time.
