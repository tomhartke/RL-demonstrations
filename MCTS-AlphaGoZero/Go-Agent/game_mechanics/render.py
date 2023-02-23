"""Just renders the game visually, does not need to be understood to build a solution."""
from pathlib import Path

import numpy as np

import pygame

from .go_base import BLACK, BOARD_SIZE, WHITE

HERE = Path(__file__).parent.resolve()


def get_image(image_name: str) -> pygame.surface.Surface:

    sprite_path = HERE / "sprites"

    image = pygame.image.load(sprite_path / image_name)
    surface = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    surface.blit(image, (0, 0))
    return surface


def render_game(
    board: np.ndarray, screen: pygame.surface.Surface, update_display: bool = True
) -> None:
    """Render the board to the screen.

    If running from delta_live_tournaments then set update_display to False
    """

    screen_width = screen.get_width()

    # Load and scale all of the necessary images
    tile_size = (screen_width) / BOARD_SIZE

    black_stone = get_image("GoBlackPiece.png")
    black_stone = pygame.transform.scale(
        black_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6)))
    )

    white_stone = get_image("GoWhitePiece.png")
    white_stone = pygame.transform.scale(
        white_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6)))
    )

    tile_img = get_image("GO_Tile0.png")
    tile_img = pygame.transform.scale(
        tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6)))
    )

    # blit board tiles
    for i in range(1, BOARD_SIZE - 1):
        for j in range(1, BOARD_SIZE - 1):
            screen.blit(tile_img, ((i * (tile_size)), int(j) * (tile_size)))

    for i in range(1, 9):
        tile_img = get_image(f"GO_Tile{str(i)}.png")
        tile_img = pygame.transform.scale(
            tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6)))
        )
        for j in range(1, BOARD_SIZE - 1):
            if i == 1:
                screen.blit(tile_img, (0, int(j) * (tile_size)))
            elif i == 2:
                screen.blit(tile_img, ((int(j) * (tile_size)), 0))
            elif i == 3:
                screen.blit(tile_img, ((BOARD_SIZE - 1) * (tile_size), int(j) * (tile_size)))
            elif i == 4:
                screen.blit(tile_img, ((int(j) * (tile_size)), (BOARD_SIZE - 1) * (tile_size)))
        if i == 5:
            screen.blit(tile_img, (0, 0))
        elif i == 6:
            screen.blit(tile_img, ((BOARD_SIZE - 1) * (tile_size), 0))
        elif i == 7:
            screen.blit(tile_img, ((BOARD_SIZE - 1) * (tile_size), (BOARD_SIZE - 1) * (tile_size)))
        elif i == 8:
            screen.blit(tile_img, (0, (BOARD_SIZE - 1) * (tile_size)))

    offset = tile_size * (1 / 6)
    # Blit the necessary chips and their positions
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == BLACK:
                screen.blit(
                    black_stone,
                    ((j * (tile_size) + offset), int(i) * (tile_size) + offset),
                )
            elif board[i][j] == WHITE:
                screen.blit(
                    white_stone,
                    ((j * (tile_size) + offset), int(i) * (tile_size) + offset),
                )

    # If the tournament flashed it's this
    if update_display:
        pygame.display.update()
