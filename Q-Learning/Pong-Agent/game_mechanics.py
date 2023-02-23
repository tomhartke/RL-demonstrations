import math
import random
from pathlib import Path
from time import sleep
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pygame
import pygame.gfxdraw
import torch
from mypy_extensions import Arg
from torch import nn

# Constants used in the game
COURT_WIDTH = 900
COURT_HEIGHT = 600
PADDLE_HEIGHT = COURT_HEIGHT // 5
SWEET_SPOT_RADIUS = 10
BALL_RADIUS = 5

HERE = Path(__file__).parent.resolve()

VALID_ACTIONS = set([-1, 0, 1])


def load_network(team_name: str) -> nn.Module:
    net_path = HERE / f"{team_name}_network.pt"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    model = torch.load(net_path, map_location=torch.device("cpu"))
    model.eval()
    return model


def save_network(network: nn.Module, team_name: str) -> None:
    assert isinstance(
        network, nn.Module
    ), f"train() function outputs an network type: {type(network)}"
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = HERE / f"{team_name}_network.pt"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            torch.save(network, net_path)
            load_network(team_name)
            return
        except Exception:
            if attempt == n_retries - 1:
                raise


def play_pong(
    your_choose_move: Callable[[np.ndarray], int],
    opponent_choose_move: Callable[[Arg(np.ndarray, "state")], int],
    n_points: int = 5,
    game_speed_multiplier: float = 1,
    verbose: bool = False,
    render: bool = False,
) -> int:
    """Play a game of pong where bat movement is controlled by `your_choose_move()`
    and `opponent_choose_move()`. You can display the game by setting verbose = True

    Args:
        your_choose_move: function that chooses move.
        opponent_choose_move: function that picks your opponent's next move
        n_points: how many points to play against your opponent
        game_speed_multiplier: multiplies the speed of the game. High == fast
        verbose: whether to print board states to console. Useful for debugging
        render: whether to render the game graphically with pygame

    Returns: total_return, which is the sum of return from the game
    """
    total_return = 0
    game = PongEnv(opponent_choose_move, render=render)

    for _ in range(n_points):
        observation, reward, done, info = game.reset()
        n = 0
        while not done:
            n += 1
            action = your_choose_move(observation)
            observation, reward, done, info = game.step(action)
            total_return += reward

            # Don't visualise the game on every timestep - there are too many!
            if verbose and n % 2 == 0:
                print(game)
            sleep(0.05 / game_speed_multiplier)

        if verbose:
            print(
                "Final positions...\nBall y:",
                round(game.ball.y, 2),
                "\nPaddle 1 y:",
                game.paddle_2.centre_y,
                "\nPaddle 2 y:",
                game.paddle_1.centre_y,
                "\nNumber of bounces off paddles:",
                game.ball.num_paddle_bounces,
            )

    return total_return


def robot_choose_move(state: np.ndarray) -> int:
    """Returns an action for the paddle to take.

    Args:
        state: State of the game as a np array, length = 5.
    Returns:
        move (int): The move you want to given the state of the game.
                    Should be in {-1,0,1}.
    """

    return 1 if state[1] > state[3] else -1


def will_hit_top_edge(paddle_center_y: float, ball_y_at_paddle: float) -> bool:
    """Return True if the ball will hit the top edge of the paddle."""
    assert isinstance(
        paddle_center_y, (int, float)
    ), f"The paddle_center_y input to will_hit_top_edge should be a number, instead it's {paddle_center_y}"

    assert isinstance(
        ball_y_at_paddle, (int, float)
    ), f"The ball_y_at_paddle input to will_hit_top_edge should be a number, instead it's {ball_y_at_paddle}"

    return abs(ball_y_at_paddle - (paddle_center_y + (PADDLE_HEIGHT / 2))) < BALL_RADIUS


def will_hit_bottom_edge(paddle_center_y: float, ball_y_at_paddle: float) -> bool:
    """Return True if the ball will hit the bottom edge of the paddle."""
    return abs(ball_y_at_paddle - (paddle_center_y - (PADDLE_HEIGHT / 2))) < BALL_RADIUS


def will_hit_edge(paddle_center_y: float, ball_y_at_paddle: float) -> bool:
    """Return True if the ball will hit the top or bottom edge of the paddle."""
    return will_hit_top_edge(paddle_center_y, ball_y_at_paddle) or will_hit_bottom_edge(
        paddle_center_y, ball_y_at_paddle
    )


def will_hit_sweet_spot(paddle_center_y: float, ball_y_at_paddle: float) -> bool:
    """Return True if the ball will hit the sweet spot of the paddle."""
    return abs(paddle_center_y - ball_y_at_paddle) <= SWEET_SPOT_RADIUS


# ############################################################
# ############################################################
# ##### BELOW CODE DOESN'T HAVE TO BE UNDERSTOOD BY YOU #####
# ############################################################
# ############################################################


class PongBall:
    """Class representing the ball in the game."""

    def __init__(self, radius: float, steps_per_state: float):
        """__init__() is called when you 'initialize' a new instance of any class.

        Args:
            radius: The radius of the ball.
            steps_per_state: The number of timesteps that pass for each call to env.step()
        """
        # The radius of the ball
        self.radius = radius

        # The direction the ball is travelling in from the start in degrees
        self.direction_degrees = random.uniform(90, 135)
        # Position of the ball in x & y coordinates
        self.x = random.uniform(50, (COURT_WIDTH // 2) - 50)
        self.y = random.uniform(100, (COURT_HEIGHT // 2) - 100)

        if random.choice([True, False]):
            self.direction_degrees = 360 - self.direction_degrees
            self.x = COURT_WIDTH - self.x
            self.y = COURT_HEIGHT - self.y

        self.steps_per_state = steps_per_state

        # Initial speed of the ball
        self.speed = 3.2 * self.steps_per_state

        # Number of times the ball has bounced off the paddle
        self.num_paddle_bounces = 0

    def update(self, paddles: Tuple["PongPaddle", "PongPaddle"]):
        """Updates the ball based on paddle positions, current speed and direction.

        Args:
            paddles: A tuple of the left and right paddles in
                the format: (left_paddle, right_paddle)
        """
        paddle_1, paddle_2 = paddles

        # Find the ball's x-position after this timestep if it doesn't hit a paddle
        new_x = self.x + self.speed * math.sin(self.direction_radians)

        y_at_paddle = (
            self.y - self.x / math.tan(self.direction_radians)
            if new_x <= 0
            else self.y + (COURT_WIDTH - self.x) / math.tan(self.direction_radians)
        )
        # Check if the new x position is outside the arena
        if new_x <= 0 or new_x >= COURT_WIDTH:
            # Has it been hit by the paddle on that side?
            if self.has_hit_paddle(paddle_1 if new_x <= 0 else paddle_2, y_at_paddle):
                self.bounce_off_paddle(
                    paddle_1 if new_x <= 0 else paddle_2, y_at_paddle
                )
            else:  # Paddle missed - game over!
                self.x = 0 if new_x <= 0 else COURT_WIDTH
                self.y = y_at_paddle
        else:  # Still inside the arena
            self.x = new_x
            self.y += self.speed * math.cos(self.direction_radians)

        # Bounce off top and bottom walls
        if self.y <= 0 or self.y >= COURT_HEIGHT:
            self.direction_degrees = (180 - self.direction_degrees) % 360
            self.y = -self.y if self.y <= 0 else COURT_HEIGHT - (self.y - COURT_HEIGHT)

        # Ensure reasonable values
        assert (
            0 < self.y < COURT_HEIGHT
        ), "Internal error in game - ignore this and rerun!"
        assert (
            15 <= self.direction_degrees <= 165 or 195 <= self.direction_degrees <= 345
        ), "Internal error in game - ignore this and rerun!"

    def has_hit_paddle(self, paddle: "PongPaddle", ball_y_at_paddle: float) -> bool:
        """Has the ball hit the paddle at the given y-coordinate?"""
        return (
            (paddle.bottom - self.radius)
            <= ball_y_at_paddle
            <= (paddle.top + self.radius)
        )

    @property
    def direction_radians(self) -> float:
        """Get the angle between the y-axis and the direction the ball is moving in (clockwise), in
        radians."""
        return math.radians(self.direction_degrees)

    def bounce_off_paddle(self, paddle: "PongPaddle", ball_y_at_paddle: float) -> None:
        """Updates the ball's speed, position and direction based on it bouncing off the paddle."""
        is_paddle_1 = self.x + self.speed * math.sin(self.direction_radians) <= 0
        paddle_x = 0 if is_paddle_1 else COURT_WIDTH

        # This is the distance to the end of the court from where the ball is at the start of the timestep
        x_dist_pre_bounce = self.x if is_paddle_1 else COURT_WIDTH - self.x
        proportion_time_pre_bounce = x_dist_pre_bounce / (
            self.speed * math.sin(self.direction_radians)
        )

        # Each time the ball hits a paddle, it increases in speed slightly!
        self.increase_speed()

        # Hit the sweet spot of the paddle!
        if will_hit_sweet_spot(paddle.centre_y, ball_y_at_paddle):
            self.direction_degrees = 90.0 if is_paddle_1 else 270.0
            self.direction_degrees += 22.5 * (random.random() - 0.5) * 2
        # Hit top of the paddle - go off at an angle
        elif will_hit_top_edge(paddle.top, ball_y_at_paddle):
            self.direction_degrees = max(
                45 - 5 * (ball_y_at_paddle - (paddle.top - self.radius)) ** 2, 22.5
            )
            self.direction_degrees = (
                self.direction_degrees if is_paddle_1 else 360 - self.direction_degrees
            )
            self.direction_degrees += 5 * (random.random() - 0.5) * 2
        # Hit bottom of the paddle - go off at an angle
        elif will_hit_bottom_edge(paddle.centre_y, ball_y_at_paddle):
            self.direction_degrees = min(
                135 + 5 * (ball_y_at_paddle - (paddle.top - self.radius)) ** 2, 157.5
            )
            self.direction_degrees = (
                self.direction_degrees if is_paddle_1 else 360 - self.direction_degrees
            )
            self.direction_degrees += 5 * (random.random() - 0.5) * 2
        # Hit normal part of the paddle
        else:
            # Mirror direction of travel of ball
            self.direction_degrees = 360 - self.direction_degrees
            self.direction_degrees += 22.5 * (random.random() - 0.5) * 2
            self.direction_degrees = self.direction_degrees % 360
            self.direction_degrees = (
                max(min(self.direction_degrees, 165), 15)
                if self.direction_degrees < 180
                else max(min(self.direction_degrees, 345), 195)
            )

        # Update x & y
        self.x = paddle_x + (1 - proportion_time_pre_bounce) * self.speed * math.sin(
            self.direction_radians
        )
        self.y = ball_y_at_paddle + (
            1 - proportion_time_pre_bounce
        ) * self.speed * math.cos(self.direction_radians)

        # Update the number of paddle bounces counter
        self.num_paddle_bounces += 1

    @property
    def position(self) -> Tuple[float, float]:
        """Get the position of the ball as a tuple (x, y)"""
        return self.x, self.y

    def increase_speed(self):
        """Increases the speed of the ball slightly.

        This happens each time the ball hits a paddle.
        """
        self.speed += 0.2 * self.steps_per_state


class PongPaddle:
    """A paddle for Pong."""

    def __init__(self, paddle_height: float) -> None:
        """
        Args:
            paddle_height: The height of the paddle.
        """
        self.height = paddle_height
        self.centre_y = COURT_HEIGHT / 2

    def update(self, action: int, steps_per_state: float) -> None:
        """Updates the paddle's position based on the action taken."""
        self.centre_y = min(
            max(self.centre_y + (2 * action * steps_per_state), self.height / 2),
            COURT_HEIGHT - self.height / 2,
        )

    @property
    def top(self) -> float:
        """y-coordinate of the top of the paddle."""
        return self.centre_y + self.height / 2

    @property
    def bottom(self) -> float:
        """y-coordinate of the bottom of the paddle."""
        return self.centre_y - self.height / 2


class PongEnv:
    """A class that encapsulates the mechanics of Pong."""

    viz_dims: Tuple[int, int] = (50, 10)

    def __init__(
        self,
        # mypy sad without Arg
        opponent_choose_move: Callable[
            [Arg(np.ndarray, "state")], int
        ] = robot_choose_move,
        steps_per_state: float = 1,
        render: bool = False,
    ):
        self.steps_per_state = steps_per_state

        self.ball = PongBall(BALL_RADIUS, steps_per_state)

        self.paddle_1 = PongPaddle(PADDLE_HEIGHT)
        self.paddle_2 = PongPaddle(PADDLE_HEIGHT)

        self.done = False
        self.timesteps = 0
        self.prev_observation = self.observation
        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.player_score = 0
        self.opponent_score = 0
        if render:
            self.visuals = PongVisuals()

    def reset(self) -> Tuple[np.ndarray, int, bool, Optional[Dict]]:

        self.ball = PongBall(BALL_RADIUS, self.steps_per_state)

        self.paddle_1 = PongPaddle(PADDLE_HEIGHT)
        self.paddle_2 = PongPaddle(PADDLE_HEIGHT)
        self.done = False
        self.timesteps = 0
        self.prev_observation = self.observation

        return self.observation, 0, self.done, None

    def step(
        self, player_action: int, verbose: bool = False
    ) -> Tuple[np.ndarray, int, bool, Optional[Dict]]:
        """Updates the state of the game after 1 timestep."""

        assert (
            player_action in VALID_ACTIONS
        ), f"Invalid action, action must be in {VALID_ACTIONS}"
        self.paddle_1.update(player_action, self.steps_per_state)

        player2_state = np.array(
            [
                COURT_WIDTH - self.ball.position[0],
                self.ball.position[1],
                (360 - self.ball.direction_degrees) % 360,
                self.paddle_2.centre_y,
                self.paddle_1.centre_y,
                self.ball_speed,
            ]
        )

        opponent_action = self.opponent_choose_move(state=player2_state)
        assert (
            opponent_action in VALID_ACTIONS
        ), f"Invalid action, action must be in {VALID_ACTIONS}"

        self.paddle_2.update(opponent_action, self.steps_per_state)
        self.ball.update((self.paddle_1, self.paddle_2))
        self.done = not 0 < self.ball.x < COURT_WIDTH
        self.timesteps += 1

        if self.render and self.timesteps % 10 == 0:
            self.visuals.draw_game(self)

        # Point lost
        if self.ball.x <= 0:
            self.opponent_score += 1
            reward = -10
        # Point won
        elif self.ball.x >= COURT_WIDTH:
            self.player_score += 1
            reward = 10
        # Ball has bounced off paddle
        elif 180 < self.prev_observation[2] < 360 and 0 < self.observation[2] < 180:
            reward = 1
        else:
            reward = 0

        self.prev_observation = self.observation

        if verbose and self.timesteps % 10 == 0:
            print(self)

        return self.observation, reward, self.done, None

    @property
    def ball_speed(self) -> float:
        return self.ball.speed / self.ball.steps_per_state

    @property
    def observation(self) -> np.ndarray:
        return np.array(
            [
                self.ball.position[0],
                self.ball.position[1],
                self.ball.direction_degrees,
                self.paddle_1.centre_y,
                self.paddle_2.centre_y,
                self.ball_speed,
            ]
        )

    def __repr__(self):
        """
        Prints game state like below:
         __________________________________________________


                            *


        |                                                  |




         __________________________________________________
        """
        # Find print coordinates of ball and paddle
        ball_x = math.floor((self.ball.x / COURT_WIDTH) * self.viz_dims[0])
        paddle_1_y = math.floor(
            (self.paddle_1.centre_y / COURT_HEIGHT) * self.viz_dims[1]
        )
        paddle_2_y = math.floor(
            (self.paddle_2.centre_y / COURT_HEIGHT) * self.viz_dims[1]
        )
        ball_y = math.floor((self.ball.y / COURT_HEIGHT) * self.viz_dims[1])

        # Top wall
        viz_string = " " + "_" * self.viz_dims[0] + "\n"

        for y in reversed(range(self.viz_dims[1])):
            # Add paddle_1
            viz_string += "|" if y == paddle_1_y else " "
            # Add the ball and spaces
            if y == ball_y:
                viz_string += (
                    " " * (ball_x - 1) + "*" + " " * (self.viz_dims[0] - ball_x)
                )
            else:
                viz_string += " " * self.viz_dims[0]
            # Add paddle_1
            viz_string += "|" if y == paddle_2_y else " "
            # Add newline
            viz_string += "\n"
        # Bottom wall
        viz_string += " " + "_" * self.viz_dims[0]

        return viz_string


BLUE_COLOR = (23, 93, 222)
YELLOW_COLOR = (255, 240, 0)
RED_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (19, 72, 162)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
LIGHT_GRAY_COLOR = (200, 200, 200)
EXTRA_WIDTH = 10


class PongVisuals:
    def __init__(self) -> None:

        self.screen, self.font, self.origin = init_pygame()

    def draw_game(self, env: PongEnv) -> None:
        """Draws match game on self.screen.

        Args:
            game: The game to draw
        """

        self.screen.fill(WHITE_COLOR)
        pixel_game_width = COURT_WIDTH + EXTRA_WIDTH
        pixel_game_height = COURT_HEIGHT

        # Draw background of the board
        pygame.gfxdraw.box(
            self.screen,
            pygame.Rect(
                self.origin[0],
                self.origin[1],
                pixel_game_width,
                pixel_game_height,
            ),
            LIGHT_GRAY_COLOR,
        )

        # Draw the paddles
        paddle_width = EXTRA_WIDTH
        pygame.gfxdraw.box(
            self.screen,
            pygame.Rect(
                # Maybe change signs below!
                self.origin[0],
                self.origin[1] + env.paddle_1.bottom,
                paddle_width,
                round(env.paddle_1.height),
            ),
            YELLOW_COLOR,
        )
        pygame.gfxdraw.box(
            self.screen,
            pygame.Rect(
                self.origin[0] + round((COURT_WIDTH)),
                self.origin[1] + round(env.paddle_2.bottom),
                paddle_width,
                round(env.paddle_2.height),
            ),
            RED_COLOR,
        )

        # Draw the ball
        # Anti-aliased circle drawing
        pygame.gfxdraw.aacircle(
            self.screen,
            self.origin[0] + round((EXTRA_WIDTH + env.ball.x)),
            self.origin[1] + round(env.ball.y),
            int(1.5 * env.ball.radius),  # 1.5 is to make it look bigger
            BLACK_COLOR,
        )

        pygame.gfxdraw.filled_circle(
            self.screen,
            self.origin[0] + round(EXTRA_WIDTH + env.ball.x),
            self.origin[1] + round(env.ball.y),
            int(1.5 * env.ball.radius),
            BLACK_COLOR,
        )

        # Draw the walls
        pygame.gfxdraw.rectangle(
            self.screen,
            pygame.Rect(
                self.origin[0],
                self.origin[1],
                pixel_game_width,
                pixel_game_height,
            ),
            BLACK_COLOR,
        )

        # Draw the score
        img = self.font.render(
            f"{env.player_score}", True, YELLOW_COLOR, LIGHT_GRAY_COLOR
        )
        rect = img.get_rect()
        rect.center = (
            self.origin[0] - self.font.get_height() // 2,
            self.origin[1] + self.font.get_height() // 2,
        )
        self.screen.blit(img, rect)

        img = self.font.render(
            f"{env.opponent_score}", True, RED_COLOR, LIGHT_GRAY_COLOR
        )
        rect = img.get_rect()
        rect.center = (
            self.origin[0] + pixel_game_width + self.font.get_height() // 2,
            self.origin[1] + self.font.get_height() // 2,
        )
        self.screen.blit(img, rect)

        img = self.font.render(
            f"Ball speed: {round(env.ball.speed, 2)}", True, BLACK_COLOR, None
        )
        rect = img.get_rect()
        rect.center = (
            self.origin[0] + pixel_game_width // 2,
            self.origin[1] + pixel_game_height + self.font.get_height() // 2,
        )
        self.screen.blit(img, rect)

        pygame.display.update()


def init_pygame():

    pygame.init()
    screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE, 32) # pygame.FULLSCREEN, 32) # had to change this
    pygame.init()
    pygame.display.set_caption("Pong")
    screen.fill(WHITE_COLOR)
    font = pygame.font.Font(None, 64)
    pygame.display.flip()
    return screen, font, (150, 50)


def human_player(*args, **kwargs) -> int:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            quit()
    is_key_pressed = pygame.key.get_pressed()
    if is_key_pressed[pygame.K_UP]:
        return -1
    elif is_key_pressed[pygame.K_DOWN]:
        return 1

    return 0
