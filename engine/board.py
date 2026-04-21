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
    points: list = field(default_factory=list)
    borne_off: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = [PointState() for _ in range(25)]  # index 0 unused
        self.points[24] = PointState(count=15, color=Color.WHITE)
        self.points[12] = PointState(count=15, color=Color.BLACK)
        self.borne_off = {Color.WHITE: 0, Color.BLACK: 0}

    # Queries ------------------------------------------------------------

    def is_head(self, point: int, color: Color) -> bool:
        return point == (24 if color == Color.WHITE else 12)

    def is_home(self, point: int, color: Color) -> bool:
        return 18 <= point_to_step(point, color) <= 23

    def all_in_home(self, color: Color) -> bool:
        for pt in range(1, 25):
            ps = self.points[pt]
            if ps.color == color and ps.count > 0 and not self.is_home(pt, color):
                return False
        return True

    def count_at(self, point: int, color: Color) -> int:
        ps = self.points[point]
        return ps.count if ps.color == color else 0

    # Mutators -----------------------------------------------------------

    def remove_one(self, point: int, color: Color) -> None:
        ps = self.points[point]
        assert ps.color == color and ps.count > 0, f"No {color} checker at {point}"
        ps.count -= 1
        if ps.count == 0:
            ps.color = None

    def place_one(self, point: int, color: Color) -> None:
        ps = self.points[point]
        assert ps.color in (None, color), f"Point {point} occupied by opponent"
        ps.color = color
        ps.count += 1

    def bear_off_one(self, point: int, color: Color) -> None:
        self.remove_one(point, color)
        self.borne_off[color] += 1

    def clone(self) -> "Board":
        new = Board.__new__(Board)
        new.points = [PointState(p.count, p.color) for p in self.points]
        new.borne_off = dict(self.borne_off)
        return new
