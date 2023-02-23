from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import pygame
import pygame.gfxdraw
from game_mechanics.state import State

BLACK_COLOR = (21, 26, 26)
WHITE_COLOR = (255, 255, 255)
GREY_COLOR = (150, 159, 167)

# Graphics consts
FULL_WIDTH = 400
FULL_HEIGHT = 600

# Table is the game rendered by PettingZoo
TABLE_WIDTH = 375
TABLE_HEIGHT = 500
TABLE_ORIGIN = ((FULL_WIDTH - TABLE_WIDTH) // 2, 0)

BG_COLOR = (7, 99, 36)
WHITE = (255, 255, 255)
RED = (217, 30, 61)

PARENT_DIR = Path(__file__).parent.parent.resolve()
SPRITE_PATH = PARENT_DIR / "sprites"


MOVE_MAP = {
    0: "fold",
    1: "call/check",
    2: "raise 1/2 pot",
    3: "raise full pot",
    4: "all-in",
}


BUTTON_MOVE_MAP: Dict[int, str] = {
    0: "fold",
    1: "call\ncheck",
    2: "raise\n",  # Half pot, to be appended with the amount
    3: "raise\n",  # Full pot, to be appended with the amount
    4: "all\nin",
}

# Clickable buttons
N_BUTTONS = len(BUTTON_MOVE_MAP)
GAP_BETWEEN_BUTTONS = 10
BUTTON_MARGIN_HORIZONTAL = (FULL_WIDTH - TABLE_WIDTH) // 2
BUTTON_MARGIN_VERTICAL = 10
BUTTON_DIM = (
    FULL_WIDTH - BUTTON_MARGIN_HORIZONTAL * 2 - N_BUTTONS * GAP_BETWEEN_BUTTONS
) // N_BUTTONS


@dataclass
class ChipInfo:
    value: int
    img: str
    number: int


CHIPS: List[ChipInfo] = [
    ChipInfo(value=10000, img="ChipOrange", number=0),
    ChipInfo(value=5000, img="ChipPink", number=0),
    ChipInfo(value=1000, img="ChipYellow", number=0),
    ChipInfo(value=100, img="ChipBlack", number=0),
    ChipInfo(value=50, img="ChipBlue", number=0),
    ChipInfo(value=25, img="ChipGreen", number=0),
    ChipInfo(value=10, img="ChipLightBlue", number=0),
    ChipInfo(value=5, img="ChipRed", number=0),
    ChipInfo(value=1, img="ChipWhite", number=0),
]


def get_button_origins(idx: int) -> Tuple[int, int]:
    return (
        BUTTON_MARGIN_HORIZONTAL + idx * (BUTTON_DIM + GAP_BETWEEN_BUTTONS),
        TABLE_ORIGIN[1] + TABLE_HEIGHT + BUTTON_MARGIN_VERTICAL,
    )


def click_in_button(pos: Tuple[int, int], idx: int) -> bool:
    x_pos, y_pos = get_button_origins(idx)
    return x_pos < pos[0] < x_pos + BUTTON_DIM and y_pos < pos[1] < y_pos + BUTTON_DIM


def get_image(image_name: str, alpha: int = 255) -> pygame.surface.Surface:
    image = pygame.image.load(SPRITE_PATH / f"{image_name}.png").copy()
    image.fill((255, 255, 255, alpha), None, pygame.BLEND_RGBA_MULT)
    return image


def get_screen() -> pygame.Surface:
    return pygame.display.set_mode((FULL_WIDTH, FULL_HEIGHT))


def get_screen_subsurface(screen: pygame.Surface):
    return screen.subsurface(
        (
            TABLE_ORIGIN[0],
            TABLE_ORIGIN[1],
            TABLE_WIDTH,
            TABLE_HEIGHT,
        )
    )


def get_tile_size(screen_height) -> float:
    return screen_height / 5


def draw_player_cards(
    screen: pygame.Surface,
    state: State,
    name: str,
    idx: int,
    render_opponent_cards: bool,
    tile_size: float,
    screen_width: int,
    screen_height: int,
    alpha: int = 255,
) -> None:
    for j, card in enumerate(state.hand):
        if not render_opponent_cards and name == "opponent":
            card_img = get_image("Card", alpha=alpha)
        else:
            card_img = get_image(card, alpha=alpha)
        card_img = pygame.transform.scale(card_img, (int(tile_size * (142 / 197)), int(tile_size)))
        # Players with even id go above public cards
        if name == "opponent":
            screen.blit(
                card_img,
                (
                    (
                        calculate_card_width(
                            idx=idx, screen_width=screen_width, tile_size=tile_size
                        )
                        - calculate_offset(state.hand, j, tile_size)
                    ),
                    calculate_height(screen_height, 4, 1, tile_size, -1),
                ),
            )
        # Players with odd id go below public cards
        else:
            screen.blit(
                card_img,
                (
                    (
                        calculate_card_width(
                            idx=idx, screen_width=screen_width, tile_size=tile_size
                        )
                        - calculate_offset(state.hand, j, tile_size)
                    ),
                    calculate_height(screen_height, 4, 3, tile_size, 0),
                ),
            )


def render(
    screen: pygame.Surface,
    player_states: Dict[str, State],
    most_recent_move: Dict[int, str],
    render_opponent_cards: bool = True,
    continue_hands: bool = True,
    win_message: Optional[str] = None,
    turn: Optional[int] = None,
) -> None:
    """player_states is a dict of player_id: player_state."""

    # Use me for external screen injection
    # mode = "human"
    screen_height = screen.get_height()
    screen_width = screen.get_width()
    tile_size = get_tile_size(screen_height)

    # Setup dimensions for card size and setup for colors
    screen.fill(BG_COLOR)

    # Load and blit all images for each card in each player's hand
    move_to_str: Dict[Optional[int], str] = {
        None: "",
        0: "fold",
        1: "call/check",
        2: "raise half",
        3: "raise full",
        4: "all in",
    }

    font = pygame.font.SysFont("arial", 22)
    for idx, (name, state) in enumerate(player_states.items()):
        alpha = 128 if turn != idx and win_message is None else 255
        # Load and blit text for player name

        # Only render the most recent move when you are not about to make a new move (or if you've just folded)
        move = most_recent_move[idx] if turn != idx or most_recent_move[idx] == "fold" else ""

        text = font.render(move, True, WHITE)
        textRect = text.get_rect()
        if name == "opponent":
            textRect.center = (
                (screen_width / (np.ceil(len(player_states) / 2) + 1) * np.ceil((idx + 1) / 2)),
                calculate_height(screen_height, 4, 1, tile_size, -(22 / 20)),
            )
        else:
            textRect.center = (
                (screen_width / (np.ceil(len(player_states) / 2) + 1) * np.ceil((idx + 1) / 2)),
                calculate_height(screen_height, 4, 3, tile_size, (23 / 20)),
            )
        screen.blit(text, textRect)

        x_pos, y_pos = get_player_chip_position(
            player_idx=0 if name == "opponent" else 1,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        draw_chips(screen=screen, x_pos=x_pos, y_pos=y_pos, n_chips=state.player_chips)

        draw_player_cards(
            screen=screen,
            state=state,
            name=name,
            idx=idx,
            render_opponent_cards=render_opponent_cards,
            tile_size=tile_size,
            screen_width=screen_width,
            screen_height=screen_height,
            alpha=alpha,
        )

        draw_player_cards(
            screen=screen,
            state=state,
            name=name,
            idx=idx,
            render_opponent_cards=render_opponent_cards,
            tile_size=tile_size,
            screen_width=screen_width,
            screen_height=screen_height,
            alpha=alpha,
        )

    # Load and blit public cards
    public_cards = player_states["player"].public_cards
    for idx, card in enumerate(public_cards):
        card_img = get_image(card)
        card_img = pygame.transform.scale(card_img, (int(tile_size * (142 / 197)), int(tile_size)))
        if len(public_cards) <= 3:
            screen.blit(
                card_img,
                (
                    (
                        (
                            ((screen_width / 2) + (tile_size * 31 / 616))
                            - calculate_offset(public_cards, idx, tile_size)
                        ),
                        calculate_height(screen_height, 2, 1, tile_size, -(1 / 2)),
                    )
                ),
            )
        elif idx <= 2:
            screen.blit(
                card_img,
                (
                    (
                        (
                            ((screen_width / 2) + (tile_size * 31 / 616))
                            - calculate_offset(state.public_cards[:3], idx, tile_size)
                        ),
                        calculate_height(screen_height, 2, 1, tile_size, -21 / 20),
                    )
                ),
            )
        else:
            screen.blit(
                card_img,
                (
                    (
                        (
                            ((screen_width / 2) + (tile_size * 31 / 616))
                            - calculate_offset(public_cards[3:], idx - 3, tile_size)
                        ),
                        calculate_height(screen_height, 2, 1, tile_size, 1 / 20),
                    )
                ),
            )

    if win_message is not None:
        # Load and blit text for player name
        font = pygame.font.SysFont("arial", 22)
        text = font.render(win_message, True, RED)
        textRect = text.get_rect()
        textRect.center = (screen_width // 2, int(screen_height * 0.45))
        pygame.draw.rect(screen, WHITE, textRect)
        screen.blit(text, textRect)

        second_line = "Click to continue" if continue_hands else None

        text = font.render(second_line, True, RED)
        textRect = text.get_rect()
        textRect.center = (
            screen_width // 2,
            int(screen_height * 0.55),
        )
        pygame.draw.rect(screen, WHITE, textRect)
        screen.blit(text, textRect)

    # pygame.display.update()


def calculate_card_width(idx: int, screen_width: int, tile_size: float, n_agents: int = 2) -> int:
    return int(
        (screen_width / (np.ceil(n_agents / 2) + 1) * np.ceil((idx + 1) / 2)) + tile_size * 31 / 616
    )


def get_player_chip_position(
    player_idx: int, screen_width: int, screen_height: int
) -> Tuple[int, int]:
    tile_size = get_tile_size(screen_height)

    if player_idx % 2 == 0:
        offset = -0.2
        multiplier = 1
    else:
        offset = 0.5
        multiplier = 3

    x_pos = calculate_card_width(
        player_idx, screen_width=screen_width, tile_size=tile_size
    ) + tile_size * (8 / 10)
    y_pos = calculate_height(screen_height, 4, multiplier, tile_size, offset)
    return int(x_pos), y_pos


def calculate_height(
    screen_height: int, divisor: float, multiplier: float, tile_size: float, offset: float
) -> int:
    return int(multiplier * screen_height / divisor + tile_size * offset)


def calculate_offset(hand: List, j: int, tile_size: float):
    return int((len(hand) * (tile_size * 23 / 56)) - ((j) * (tile_size * 23 / 28)))


def draw_chips(screen: pygame.surface.Surface, n_chips: int, x_pos: int, y_pos: int):

    font = pygame.font.SysFont("arial", 20)
    text = font.render(str(n_chips), True, WHITE)
    textRect = text.get_rect()
    tile_size = get_tile_size(screen.get_height())

    # Calculate number of each chip
    height = 0

    # Draw the chips
    for chip_info in CHIPS:
        num = n_chips / chip_info.value
        chip_info.number = int(num)
        n_chips %= chip_info.value

        chip_img = get_image(chip_info.img, alpha=255)
        chip_img = pygame.transform.scale(chip_img, (int(tile_size / 2), int(tile_size * 16 / 45)))

        for j in range(chip_info.number):
            height_offset = (j + height) * tile_size / 15
            screen.blit(
                chip_img,
                (x_pos, y_pos - height_offset),
            )
        height += chip_info.number

    # Blit number of chips
    textRect.center = (
        x_pos + tile_size // 4,
        y_pos - ((height + 1) * tile_size / 15),
    )
    screen.blit(text, textRect)


def draw_both_chip_stacks(
    screen: pygame.Surface, player_total: int, opponent_total: int, max_num_chips: int
) -> None:
    player_chips = int(min(max(0, player_total), max_num_chips))
    opponent_chips = int(min(max(0, opponent_total), max_num_chips))

    draw_chips(
        screen=screen,
        n_chips=opponent_chips,
        x_pos=int(screen.get_width() * 0.02),
        y_pos=int(FULL_HEIGHT * 0.2),
    )
    draw_chips(
        screen=screen,
        n_chips=player_chips,
        x_pos=int(screen.get_width() * 0.02),
        y_pos=int(FULL_HEIGHT * 0.66),
    )


def render_multi_line_centre(
    screen, font, string: str, x: int, y: int, fsize: int, color: Tuple[int, int, int]
) -> None:
    """Render centre aligned 'string' with line breaks on '\n'."""
    lines = string.splitlines()
    for i, l in enumerate(lines):
        text = font.render(l, False, color)
        text_rect = text.get_rect(center=(x, y + fsize * i))
        screen.blit(text, text_rect)


def draw_action(screen, font, action: str, idx: int, legal: bool) -> None:
    x_pos, y_pos = get_button_origins(idx)

    rect = pygame.Rect(
        x_pos,
        y_pos,
        BUTTON_DIM,
        BUTTON_DIM,
    )
    color = WHITE_COLOR if legal else GREY_COLOR
    pygame.gfxdraw.rectangle(
        screen,
        rect,
        color,
    )

    text = font.render(action, True, color)

    # Centre the text in the middle of the button
    x_pos += BUTTON_DIM // 2
    text_height = text.get_height() * action.count("\n")
    y_pos += (BUTTON_DIM - text_height) // 2

    render_multi_line_centre(screen, font, action, x_pos, y_pos, font.get_height(), color=color)


def draw_possible_actions(screen, font, state: State, turn) -> None:
    pot_size = state.player_chips + state.opponent_chips

    for idx, action in BUTTON_MOVE_MAP.items():
        if idx == 2:
            action += str(pot_size // 2)
        elif idx == 3:
            action += str(pot_size)
        draw_action(screen, font, action, idx, idx in state.legal_actions and turn == 0)


def wait_for_click() -> None:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                return


def human_player(state: State) -> int:
    print("Your move, click to choose!")
    LEFT = 1
    while True:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
                pos = pygame.mouse.get_pos()
                for idx in range(N_BUTTONS):
                    if click_in_button(pos, idx) and idx in state.legal_actions:
                        return idx
