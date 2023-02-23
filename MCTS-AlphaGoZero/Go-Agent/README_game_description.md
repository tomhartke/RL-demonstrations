# Week four: Go :white_circle: :black_circle: :white_circle: :black_circle:

![](images/go_game.webp)

## Rules of Go

Go is a two player board game. The aim is to capture territory from your opponent :guardsman:. The player that has the most territory at the end of the game is the winner.

The rules of Go are complex and many different rule-sets exist. [Wikipedia](https://en.wikipedia.org/wiki/Rules_of_Go) has a good description of the rules so we won't rehash them here.

The rule-set we will use here are the [CGOS](http://www.yss-aya.com/cgos/) rules that allow two bots to play each other (as other forms of Go rely on agreements between human players). One amendment to the CGOS rules is that instead of 'Positional SuperKo', we only check for simple Ko.

Professional go is played on a 19x19 board, but we will use a 9x9 board so you don't need your own GPU farm :pig:!

Do not fret if you do not understand the rules fully, we will provide your bots with a list of legal moves for each turn!

### Starting and ending the game

Go starts with an empty board and black places the first stone. We will randomise which player is black on each game.

The game ends when both players pass.

### Scoring

As in the [CGOS](http://www.yss-aya.com/cgos/) rules, we will use [area scoring](<https://en.wikipedia.org/wiki/Go_(game)#Scoring_rules>). So a player's score is the number of stones they have on the board + the number of empty territories they surround at the end of the game.

The score also includes a [Komi](<https://en.wikipedia.org/wiki/Go_(game)#Komi>) which is a handicap given to black as it is advantageous to lay the first stone. We will use a komi of `7.5` which is subtracted from black's score at the end of the game.

# Competition Rules :scroll:

1. You must build an agent to play Go using either a **reinforcement learning** algorithm or a **planning algorithm** (such as monte carlo tree search :deciduous_tree:) or a mix of **both**!

2. You can only write code in the `main` folder. As previously, the entry point to your code will be `main.py`, but you can add, and import from, other custom python files in the `main` folder.

   - You can only store data to be used in a competition in a `.pkl` file by `save_file()`.
   - You can pkl anything you want, even a dictionary of pytorch networks (or nothing)! Just make sure your choose_move can read it.
   - In the competition, your agent will call the `choose_move()` function in `main.py` to select a move (`choose_move()` may call other functions in the `main` folder)
   - We provide an `MCTS` class for you to implement. This class is passed to your choose_move and will persist across calls to choose_move, allowing you to prune the tree. In the competition we will initialise this class for you, so please do not add any arguments to `__init__()`. (You do not have to use this class).
   - Any code not in the `main` folder will not be used.

3. Your choose_move function will have a limit of 1 second to run!

4. Submission deadline: **2:30pm UTC, Sunday**.

- You can update your code after submitting, but **not after the deadline**.
- **Check your submission is valid with `check_submission()`**

## Competition Format :crossed_swords:

Each matchup will consist of one game of Go between two players with the winning player progressing to the later rounds. The Komi (handicap) controls for the fact that it's an advantage to going first

The competition & discussion will be in [Gather Town](https://app.gather.town/app/nJwquzJjD4TLKcTy/Delta%20Academy) at **3:30pm UTC on Sunday** (60 mins after submission deadline)!

## Technical Details :hammer:

### Rewards :moneybag:

| Reward |           |
| ------ | --------- |
| `+1`   | You win   |
| `-1`   | You lose  |
| `0`    | Otherwise |

(A draw is not possible as the Komi is 7.5)

### State at each timestep :mag:

The `tuple` returned from each `env.step()` has:

- The `State` object (defined in `state.py`). This is described in detail further down
- The `reward` for each timestep
- Whether the point is `done` (boolean)
- The `info` dictionary
  - This contains a key `legal_moves` with a numpy array containing all legal moves. This is the legal moves that can be taken on the next turn.

### Actions :muscle:

Valid actions are integers in the range `0-81` (inclusive). Each position on the board has is deinfed by an integer - e.g. (row 1, col 0) = 10. You can convert an integer action a to its corresponding board coordinate through the `int_to_coord()` function.

The integer `81` is the pass action. Two consecutive passes ends the game. If you do not return this action the game may never end!

## Code you write :point_left:

<details>
<summary><code style="white-space:nowrap;">  train()</code></summary>
(Optional)
Write this to train your algorithm from experience in the environment.
<br />
<br />
(Optional) Returns a pickelable object for your choose_move to use
</details>

<details>
<summary><code style="white-space:nowrap;">  choose_move()</code></summary>
This acts greedily given the state and network.

In the competition, the choose_move() function is called to make your next move. Takes inputs of `state`, `pkl_file` and `mcts` (see below).

</details>

<details>
<summary><code style="white-space:nowrap;">  MCTS()</code></summary>
The skeleton of a class that you can use to implement mcts. Use this to persist your mcts tree between steps so it can be pruned.
</details>

## Existing Code :pray:

<details>
<summary><code style="white-space:nowrap;">  GoEnv</code></summary>
The environment class controls the game and runs the opponents. It should be used for training your agent.
<br />
<br />
See example usage in <code style="white-space:nowrap;">play_go()</code>.
<br />
<br />
The opponents' <code style="white-space:nowrap;">choose_move</code> functions are input at initialisation (when <code style="white-space:nowrap;">Env(opponent_choose_moves)</code> is called). Every time you call <code style="white-space:nowrap;">Env.step()</code>, both players make a move according to their choose_move function. Players view the board from their own perspective (i.e player1_board = -player2_board).
    <br />
    <br />

<code style="white-space:nowrap;">GoEnv</code> has a <code style="white-space:nowrap;"> verbose</code> argument which prints the information about the game to the console when set to <code style="white-space:nowrap;">True</code>. <code style="white-space:nowrap;"> GoEnv</code> also has a render argument which visualises the game in pygame when set to <code style="white-space:nowrap;">True</code>. This allows you to visualise your AI's skills. You can play against your agent using the <code style="white-space:nowrap;">human_player()</code> function!

</details>

<details>
<summary><code style="white-space:nowrap;"> choose_move_randomly()</code></summary>
A basic go playing bot that makes legal random moves, learn to beat this first!
<br />
<br />
Takes the state as input and outputs an action.
</details>

<details>
<summary><code style="white-space:nowrap;">  play_go()</code></summary>
Plays a game of Go, which can be rendered through pygame (if <code style="white-space:nowrap;">render=True</code>).
You can play against your own bot if you set <code style="white-space:nowrap;">your_choose_move</code> to <code style="white-space:nowrap;">human_player</code>!
<br />
<br />

Inputs:

<code style="white-space:nowrap;">your_choose_move</code>: Function that takes the state and outputs the action for your agent.

<code style="white-space:nowrap;">opponent_choose_move</code>: Function that takes the state and outputs the action for the opponent.

<code style="white-space:nowrap;">game_speed_multiplier</code>: controls the gameplay speed. High numbers mean fast games, low numbers mean slow games.

<code style="white-space:nowrap;">verbose</code>: whether to print info to the console.

<code style="white-space:nowrap;">render</code>: whether to render the match through pygame

</details>

<details>
<summary><code style="white-space:nowrap;"> human_player()</code></summary>
Use this in place of a choose_move function to play against your bot yourself!
Left click the board to place a stone, right click to pass.
<br />
<br />
Takes the state as input and outputs an action.
</details>

<details>
<summary><code style="white-space:nowrap;"> State</code></summary>

This is a big dataclass. Hold onto your hats.

However there are only 3 important attributes you _need_ to know about:

- `board`: a (board size x board size) numpy array containing the board state. The board is represented as follows:

  - `-1` = white stone
  - `0` = empty
  - `1` = black stone
  - There are other possible values, but these aren't important

- `recent_moves`: a tuple of all `PlayerMove`s made in the game so far. This is useful for keeping track of the game history & **as a unique identifier for a state**. :wink:

- `to_play`: signifies whose turn it is to play at the current state. Either `BLACK` or `WHITE`.

The other attributes are explained in the docstring, although can be ignored (unless building a pro-level Go AI).

</details>

<details>
<summary><code style="white-space:nowrap;">  int_to_coord()</code></summary>

A function that converts from an integer to a coordinate tuple (or None, if the pass move).

</details>

<details>
<summary><code style="white-space:nowrap;">  PlayerMove</code></summary>

A dataclass that simply represents a move made by a player.

It has 2 attributes:

<code style="white-space:nowrap;">color</code>: either <code style="white-space:nowrap;"> WHITE</code> or <code style="white-space:nowrap;">BLACK</code>

<code style="white-space:nowrap;"> move</code>: the move made by the player. This is either an integer in the range <code style="white-space:nowrap;">0-81</code> (inclusive) or <code style="white-space:nowrap;">None</code> if the player passes.

</details>

<details>
<summary><code style="white-space:nowrap;">  reward_function()</code></summary>
Gives the reward that would be recieved in the State for the player playing as Black. This reward \* -1 is the reward recieved by the player playing as White. `1` if black wins, `-1` if white wins, `0` otherwise.
</details>

<details>
<summary><code style="white-space:nowrap;">  transition_funcion()</code></summary>

Gives the successor `State` object given the current `State` and the action `int` made by the player whose turn it is to play.

</details>

<details>
<summary><code style="white-space:nowrap;">is_terminal()</code></summary>

Returns `True` if the game is over, `False` otherwise.

Takes the `State` as input.

</details>
