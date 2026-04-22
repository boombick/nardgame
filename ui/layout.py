from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class BoardLayout:
    """Pure-math layout for the board. Renderer reads these rects and the
    `checker_positions` helper to draw; no pygame imports here so the whole
    module stays unit-testable."""

    screen_w: int = 1280
    screen_h: int = 800
    board_width: int = 1024
    board_height: int = 640
    bar_width: int = 32
    point_count_per_quadrant: int = 6

    @property
    def board_left(self) -> int:
        return (self.screen_w - self.board_width) // 2

    @property
    def board_top(self) -> int:
        return (self.screen_h - self.board_height) // 2

    @property
    def quadrant_width(self) -> int:
        return (self.board_width - self.bar_width) // 2

    @property
    def point_width(self) -> int:
        return self.quadrant_width // self.point_count_per_quadrant

    @property
    def row_height(self) -> int:
        return self.board_height // 2

    def point_rect(self, point: int) -> Rect:
        """Return the triangle bounding rect for absolute point 1..24."""
        top_row = point >= 13
        if top_row:
            y = self.board_top
            if 13 <= point <= 18:
                col = point - 13  # 0..5 left->right in the top-left quadrant
                x = self.board_left + col * self.point_width
            else:  # 19..24 (top-right quadrant, left-to-right)
                col = point - 19
                x = (self.board_left + self.quadrant_width + self.bar_width
                     + col * self.point_width)
        else:
            y = self.board_top + self.row_height
            if 7 <= point <= 12:
                col = 12 - point  # bottom-left quadrant: pt 12 is leftmost
                x = self.board_left + col * self.point_width
            else:  # 1..6 (bottom-right quadrant: pt 6 is leftmost)
                col = 6 - point
                x = (self.board_left + self.quadrant_width + self.bar_width
                     + col * self.point_width)
        return Rect(x=x, y=y, w=self.point_width, h=self.row_height)


# Dice slot geometry. Pure math so we can unit-test it without pygame.
# Slot size and spacing match the render in GameScreen._draw_dice — if one
# changes, the other must follow.
DICE_SIZE = 64
DICE_SPACING = 80     # vertical distance between stacked slots (non-double)
DICE_COL_SPACING = 80  # horizontal distance between columns (double grid)


def dice_slot_count(dice: Tuple[int, int]) -> int:
    """Doubles are played 4 times; everything else 2 times."""
    return 4 if dice[0] == dice[1] else 2


def dice_slot_offsets(dice: Tuple[int, int]) -> List[Tuple[int, int]]:
    """(dx, dy) offsets of each slot relative to the dice-group anchor.

    Non-double: two slots stacked vertically at the same x.
    Double: 2×2 grid (two columns × two rows). Visit order is row-major
    (top-left, top-right, bottom-left, bottom-right) — the same order we
    dim slots as moves are played, so the user's eye tracks "which die has
    been spent" left-to-right, top-to-bottom."""
    if dice[0] != dice[1]:
        return [(0, 0), (0, DICE_SPACING)]
    return [(0, 0),
            (DICE_COL_SPACING, 0),
            (0, DICE_SPACING),
            (DICE_COL_SPACING, DICE_SPACING)]


def dice_slot_values(dice: Tuple[int, int]) -> List[int]:
    """Face value shown in each slot, in the same order as
    `dice_slot_offsets`. Doubles repeat the value four times."""
    if dice[0] != dice[1]:
        return [dice[0], dice[1]]
    return [dice[0]] * 4


def checker_positions(point: int, count: int, layout: BoardLayout,
                      top_row: bool) -> List[Tuple[int, int]]:
    """Return a list of (cx, cy) centers for `count` stacked checkers at
    `point`. Top row stacks downward from the upper edge; bottom row stacks
    upward from the lower edge. Overflow (count > visible rows) compresses
    spacing so all checkers still fit in the rect."""
    rect = layout.point_rect(point)
    radius = min(rect.w // 2 - 2, 22)
    cx = rect.x + rect.w // 2
    max_visible = max(1, rect.h // (radius * 2))
    step = (min(radius * 2, rect.h // max(count, 1))
            if count > max_visible else radius * 2)
    positions: List[Tuple[int, int]] = []
    for i in range(count):
        if top_row:
            cy = rect.y + radius + i * step
        else:
            cy = rect.y + rect.h - radius - i * step
        positions.append((cx, cy))
    return positions
