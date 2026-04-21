# Nardgame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a long backgammon (nardgame) engine, pygame UI, notation system, and model integration for training a local ML model.

**Architecture:** Monolithic project with strict module boundaries. Engine is independent of pygame. Player-perspective step system (0-23) normalizes movement for both players.

**Tech Stack:** Python 3, Pygame, requests (OpenRouter), pytest

---

### Task 1: Project scaffold + Board data structure

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/board.py`
- Create: `tests/__init__.py`
- Create: `tests/test_board.py`
- Create: `requirements.txt`

**Key design decision — player-relative steps (0-23):**

Both players use the same step range internally. Step 0 = head, step 23 = deepest home point. Steps 18-23 = home. This makes all movement rules uniform.

```
White: step = 24 - point (pt 24=step 0, pt 1=step 23)
Black: step = 12 - point (if point<=12), else 36 - point (pt 12=step 0, pt 13=step 23)
```

- [ ] **Step 1: Create project structure and requirements.txt**

```bash
mkdir -p engine tests
touch engine/__init__.py tests/__init__.py
```

`requirements.txt`:
```
pygame>=2.5
requests>=2.31
pytest>=8.0
```

- [ ] **Step 2: Write failing tests for Board**

`tests/test_board.py`:
```python
import pytest
from engine.board import Board, Color, point_to_step, step_to_point


class TestBoardInit:
    def test_initial_white_checkers(self):
        board = Board()
        assert board.points[24].count == 15
        assert board.points[24].color == Color.WHITE

    def test_initial_black_checkers(self):
        board = Board()
        assert board.points[12].count == 15
        assert board.points[12].color == Color.BLACK

    def test_initial_empty_points(self):
        board = Board()
        for pt in range(1, 24):
            if pt not in (12, 24):
                assert board.points[pt].count == 0

    def test_initial_borne_off(self):
        board = Board()
        assert board.borne_off[Color.WHITE] == 0
        assert board.borne_off[Color.BLACK] == 0

    def test_total_white_checkers(self):
        board = Board()
        on_board = sum(p.count for p in board.points if p.color == Color.WHITE)
        assert on_board + board.borne_off[Color.WHITE] == 15

    def test_total_black_checkers(self):
        board = Board()
        on_board = sum(p.count for p in board.points if p.color == Color.BLACK)
        assert on_board + board.borne_off[Color.BLACK] == 15


class TestStepConversion:
    def test_white_head_is_step_0(self):
        assert point_to_step(24, Color.WHITE) == 0

    def test_white_home_edge_is_step_18(self):
        assert point_to_step(6, Color.WHITE) == 18

    def test_white_deepest_home_is_step_23(self):
        assert point_to_step(1, Color.WHITE) == 23

    def test_black_head_is_step_0(self):
        assert point_to_step(12, Color.BLACK) == 0

    def test_black_home_edge_is_step_18(self):
        assert point_to_step(18, Color.BLACK) == 18

    def test_black_deepest_home_is_step_23(self):
        assert point_to_step(13, Color.BLACK) == 23

    def test_roundtrip_white(self):
        for pt in range(1, 25):
            assert step_to_point(point_to_step(pt, Color.WHITE), Color.WHITE) == pt

    def test_roundtrip_black(self):
        for pt in range(1, 25):
            assert step_to_point(point_to_step(pt, Color.BLACK), Color.BLACK) == pt
```

- [ ] **Step 3: Run tests — expect failure**

Run: `cd /Users/sinitsyn-as/Documents/own_paws/nn && python -m pytest tests/test_board.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement Board**

`engine/board.py`:
```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class Color(Enum):
    WHITE = "white"
    BLACK = "black"


@dataclass
class PointState:
    count: int = 0
    color: Optional[Color] = None


def point_to_step(point: int, color: Color) -> int:
    """Convert absolute point (1-24) to player-relative step (0-23).
    Step 0 = head, step 23 = deepest home, steps 18-23 = home."""
    if color == Color.WHITE:
        return 24 - point
    else:
        if point <= 12:
            return 12 - point
        else:
            return 36 - point


def step_to_point(step: int, color: Color) -> int:
    """Convert player-relative step (0-23) back to absolute point (1-24)."""
    if color == Color.WHITE:
        return 24 - step
    else:
        if step <= 11:
            return 12 - step
        else:
            return 36 - step


@dataclass
class Board:
    points: list = field(default_factory=lambda: [PointState() for _ in range(25)])
    borne_off: dict = field(default_factory=lambda: {Color.WHITE: 0, Color.BLACK: 0})

    def __post_init__(self):
        # Reset and set initial position
        self.points = [PointState() for _ in range(25)]
        self.points[24] = PointState(count=15, color=Color.WHITE)
        self.points[12] = PointState(count=15, color=Color.BLACK)
        self.borne_off = {Color.WHITE: 0, Color.BLACK: 0}

    def is_head(self, point: int, color: Color) -> bool:
        """Check if point is the head for this color."""
        if color == Color.WHITE:
            return point == 24
        return point == 12

    def is_home(self, point: int, color: Color) -> bool:
        """Check if point is in the home zone for this color."""
        step = point_to_step(point, color)
        return 18 <= step <= 23

    def all_in_home(self, color: Color) -> bool:
        """Check if all checkers of this color are in their home zone."""
        for pt in range(1, 25):
            ps = self.points[pt]
            if ps.color == color and ps.count > 0:
                if not self.is_home(pt, color):
                    return False
        return True
```

- [ ] **Step 5: Run tests — expect pass**

Run: `python -m pytest tests/test_board.py -v`
Expected: all 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add engine/__init__.py engine/board.py tests/__init__.py tests/test_board.py requirements.txt
git commit -m "feat: add Board data structure with player-relative step system"
```
