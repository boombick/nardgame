from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class Color(Enum):
    WHITE = "white"
    BLACK = "black"


def opposite(color: "Color") -> "Color":
    return Color.BLACK if color == Color.WHITE else Color.WHITE


@dataclass
class PointState:
    count: int = 0
    color: Optional[Color] = None


def point_to_step(point: int, color: Color) -> int:
    """Absolute point (1..24) -> player-relative step (0..23).
    Step 0 = head, step 23 = deepest home, steps 18..23 = home."""
    if color == Color.WHITE:
        return 24 - point
    if point <= 12:
        return 12 - point
    return 36 - point


def step_to_point(step: int, color: Color) -> int:
    """Inverse of point_to_step."""
    if color == Color.WHITE:
        return 24 - step
    if step <= 11:
        return 12 - step
    return 36 - step


@dataclass
class Board:
    """Immutable-layout board state for a two-player long backgammon game."""

    points: list[PointState] = field(default_factory=list, init=False)
    borne_off: dict[Color, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.points = [PointState() for _ in range(25)]  # index 0 unused
        self.points[24] = PointState(count=15, color=Color.WHITE)
        self.points[12] = PointState(count=15, color=Color.BLACK)
        self.borne_off = {Color.WHITE: 0, Color.BLACK: 0}

    # Queries ------------------------------------------------------------

    def is_head(self, point: int, color: Color) -> bool:
        """Return True if point is the starting head point for color."""
        return point == (24 if color == Color.WHITE else 12)

    def is_home(self, point: int, color: Color) -> bool:
        """Return True if point lies within color's home board."""
        return 18 <= point_to_step(point, color) <= 23

    def all_in_home(self, color: Color) -> bool:
        """Return True if every remaining checker of color is in its home board."""
        for pt in range(1, 25):
            ps = self.points[pt]
            if ps.color == color and ps.count > 0 and not self.is_home(pt, color):
                return False
        return True

    def count_at(self, point: int, color: Color) -> int:
        """Return the number of color's checkers at point (0 if occupied by opponent)."""
        ps = self.points[point]
        return ps.count if ps.color == color else 0

    def pip_count(self, color: Color) -> int:
        """Sum of pips-to-off across all of color's checkers currently on
        the board. Lower is better — starting position has every side at
        15 checkers × 24 pips = 360. Borne-off checkers contribute 0."""
        total = 0
        for pt in range(1, 25):
            ps = self.points[pt]
            if ps.color == color and ps.count > 0:
                total += (24 - point_to_step(pt, color)) * ps.count
        return total

    # Mutators -----------------------------------------------------------

    def remove_one(self, point: int, color: Color) -> None:
        """Remove one checker of color from point; raise AssertionError if none present."""
        ps = self.points[point]
        assert ps.color == color and ps.count > 0, f"No {color} checker at {point}"
        ps.count -= 1
        if ps.count == 0:
            ps.color = None

    def place_one(self, point: int, color: Color) -> None:
        """Place one checker of color on point; raise AssertionError if occupied by opponent."""
        ps = self.points[point]
        assert ps.color in (None, color), f"Point {point} occupied by opponent"
        ps.color = color
        ps.count += 1

    def bear_off_one(self, point: int, color: Color) -> None:
        """Remove one checker of color from point and add it to the borne-off tally."""
        self.remove_one(point, color)
        self.borne_off[color] += 1

    def clone(self) -> "Board":
        """Return an independent deep copy of this board."""
        new = Board.__new__(Board)
        new.points = [PointState(p.count, p.color) for p in self.points]
        new.borne_off = dict(self.borne_off)
        return new
