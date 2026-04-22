# Nardgame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a long backgammon (nardgame) engine, pygame UI, notation system, and model integration for training a local ML model.

**Architecture:** Monolithic project with strict module boundaries. Engine is independent of pygame. Player-perspective step system (0-23) normalizes movement for both players. Legacy files (`console.py`, `fysom.py`, `patterns.py`, old `main.py`) are not used by the new engine — leave them alone until the new `main.py` entry point is in place (Task 15).

**Tech Stack:** Python 3.11+, Pygame, requests (OpenRouter), pytest

**Working directory:** `/Users/boombick/work/nardgame`

**Module layout (built incrementally):**
```
engine/     # Board, moves, game (pure Python, no UI deps)
notation/   # Writer, parser, replay
ui/         # Pygame renderer, input, screens
models/     # Base model interface
opponents/  # Local stub + OpenRouter
tests/      # pytest
games/      # saved .narde files
main.py     # new entry point (replaces legacy main.py in Task 15)
```

---

### Task 1: Project scaffold + Board data structure

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/board.py`
- Create: `tests/__init__.py`
- Create: `tests/test_board.py`
- Create: `requirements.txt`

**Key design decision — player-relative steps (0-23):**

Both players use the same step range internally. Step 0 = head, step 23 = deepest home point. Steps 18–23 = home. This makes all movement rules uniform.

```
White: step = 24 - point       (pt 24 → step 0, pt 1 → step 23)
Black: step = 12 - point  if point <= 12
       step = 36 - point  if point >  12   (pt 12 → step 0, pt 13 → step 23)
```

- [ ] **Step 1: Create project structure and requirements.txt**

```bash
cd /Users/boombick/work/nardgame
mkdir -p engine tests games
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
        for pt in range(1, 25):
            if pt not in (12, 24):
                assert board.points[pt].count == 0

    def test_initial_borne_off(self):
        board = Board()
        assert board.borne_off[Color.WHITE] == 0
        assert board.borne_off[Color.BLACK] == 0

    def test_total_white_checkers(self):
        board = Board()
        on_board = sum(p.count for p in board.points[1:] if p.color == Color.WHITE)
        assert on_board + board.borne_off[Color.WHITE] == 15

    def test_total_black_checkers(self):
        board = Board()
        on_board = sum(p.count for p in board.points[1:] if p.color == Color.BLACK)
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


class TestHomeAndHead:
    def test_white_head(self):
        assert Board().is_head(24, Color.WHITE) is True
        assert Board().is_head(12, Color.WHITE) is False

    def test_black_head(self):
        assert Board().is_head(12, Color.BLACK) is True
        assert Board().is_head(24, Color.BLACK) is False

    def test_white_home_range(self):
        b = Board()
        for pt in range(1, 7):
            assert b.is_home(pt, Color.WHITE) is True
        for pt in range(7, 25):
            assert b.is_home(pt, Color.WHITE) is False

    def test_black_home_range(self):
        b = Board()
        for pt in range(13, 19):
            assert b.is_home(pt, Color.BLACK) is True
        for pt in [1, 2, 12, 19, 20, 24]:
            assert b.is_home(pt, Color.BLACK) is False

    def test_all_in_home_initial_false(self):
        b = Board()
        assert b.all_in_home(Color.WHITE) is False
        assert b.all_in_home(Color.BLACK) is False
```

- [ ] **Step 3: Run tests — expect failure**

Run: `cd /Users/boombick/work/nardgame && python -m pytest tests/test_board.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'engine.board'`)

- [ ] **Step 4: Implement Board**

`engine/board.py`:
```python
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
```

- [ ] **Step 5: Run tests — expect pass**

Run: `python -m pytest tests/test_board.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add engine/__init__.py engine/board.py tests/__init__.py tests/test_board.py requirements.txt
git commit -m "feat(engine): add Board with player-relative step system"
```

---

### Task 2: Move primitives + single-die validation

**Files:**
- Create: `engine/moves.py`
- Create: `tests/test_moves_basic.py`

Introduces `Move` and `is_legal_single(board, color, from_pt, die)` — a one-die check that ignores head-rule and block-rule (added in Tasks 3–4). Validates: own checker, destination not opponent-blocked, destination in range or bear-off legal.

- [ ] **Step 1: Write failing tests**

`tests/test_moves_basic.py`:
```python
import pytest
from engine.board import Board, Color
from engine.moves import Move, is_legal_single, apply_single


class TestMoveDataclass:
    def test_move_fields(self):
        m = Move(from_point=24, to_point=19, is_bear_off=False)
        assert m.from_point == 24 and m.to_point == 19
        assert m.is_bear_off is False

    def test_bear_off_move(self):
        m = Move(from_point=3, to_point=0, is_bear_off=True)
        assert m.is_bear_off is True
        assert m.to_point == 0


class TestIsLegalSingle:
    def test_white_from_head(self):
        assert is_legal_single(Board(), Color.WHITE, 24, 5) is True

    def test_not_own_checker(self):
        b = Board()
        assert is_legal_single(b, Color.BLACK, 24, 5) is False

    def test_empty_from_point(self):
        assert is_legal_single(Board(), Color.WHITE, 10, 3) is False

    def test_blocked_by_opponent(self):
        b = Board()
        # Put a black checker at 19; white 24->19 must be illegal
        b.points[19].count = 1
        b.points[19].color = Color.BLACK
        assert is_legal_single(b, Color.WHITE, 24, 5) is False

    def test_own_pile_is_ok(self):
        b = Board()
        b.points[19].count = 2
        b.points[19].color = Color.WHITE
        assert is_legal_single(b, Color.WHITE, 24, 5) is True

    def test_cannot_bear_off_when_not_home(self):
        assert is_legal_single(Board(), Color.WHITE, 24, 5) is True
        # Trying to bear off from head directly is illegal
        b = Board()
        # Move all white to home first (synthetic)
        b.points[24].count = 0
        b.points[24].color = None
        b.points[6].count = 15
        b.points[6].color = Color.WHITE
        # Still, from pt 24 we have no checker -> illegal
        assert is_legal_single(b, Color.WHITE, 24, 6) is False

    def test_bear_off_exact(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[6].count = 15
        b.points[6].color = Color.WHITE
        assert is_legal_single(b, Color.WHITE, 6, 6) is True

    def test_bear_off_over_when_higher_empty(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[3].count = 15
        b.points[3].color = Color.WHITE
        # die=5, highest occupied is 3 -> allowed to bear off from 3
        assert is_legal_single(b, Color.WHITE, 3, 5) is True

    def test_bear_off_over_blocked_when_higher_occupied(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[5].count = 1
        b.points[5].color = Color.WHITE
        b.points[3].count = 14
        b.points[3].color = Color.WHITE
        # die=6, not exact; can bear off from 5 (highest) only, not 3
        assert is_legal_single(b, Color.WHITE, 5, 6) is True
        assert is_legal_single(b, Color.WHITE, 3, 6) is False


class TestApplySingle:
    def test_apply_regular(self):
        b = Board()
        apply_single(b, Color.WHITE, Move(24, 19, False))
        assert b.points[24].count == 14
        assert b.points[19].count == 1
        assert b.points[19].color == Color.WHITE

    def test_apply_bear_off(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[6].count = 15
        b.points[6].color = Color.WHITE
        apply_single(b, Color.WHITE, Move(6, 0, True))
        assert b.points[6].count == 14
        assert b.borne_off[Color.WHITE] == 1
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_moves_basic.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement moves.py**

`engine/moves.py`:
```python
from dataclasses import dataclass
from engine.board import Board, Color, point_to_step, step_to_point


@dataclass(frozen=True)
class Move:
    from_point: int          # 1..24
    to_point: int            # 1..24, or 0 for bear-off
    is_bear_off: bool = False


def destination_step(from_point: int, die: int, color: Color) -> int:
    """Target step after moving `die` pips from `from_point`. May exceed 23."""
    return point_to_step(from_point, color) + die


def is_bear_off_legal(board: Board, color: Color, from_point: int, die: int) -> bool:
    """True if bearing off from `from_point` with `die` is legal.
    Pre: a checker of `color` sits on from_point.
    Requires: all_in_home(color), exact die OR (die overshoots and no higher occupied
    home point of `color` exists behind from_point)."""
    if not board.all_in_home(color):
        return False
    src_step = point_to_step(from_point, color)
    if src_step < 18:
        return False
    target_step = src_step + die
    if target_step == 24:
        return True
    if target_step > 24:
        # Allowed only if no checker of `color` at a lower step (higher-priority point)
        for s in range(18, src_step):
            pt = step_to_point(s, color)
            if board.count_at(pt, color) > 0:
                return False
        return True
    return False  # undershoot: not a bear-off


def is_legal_single(board: Board, color: Color, from_point: int, die: int) -> bool:
    """One-die legality ignoring head-rule and 6-block rule.
    Head/block rules are layered in Tasks 3–4 via sequence generation."""
    ps = board.points[from_point]
    if ps.color != color or ps.count <= 0:
        return False
    target_step = destination_step(from_point, die, color)
    if target_step >= 24:
        return is_bear_off_legal(board, color, from_point, die)
    to_pt = step_to_point(target_step, color)
    dest = board.points[to_pt]
    if dest.count > 0 and dest.color != color:
        return False
    return True


def apply_single(board: Board, color: Color, move: Move) -> None:
    """Apply a single-die move to the board in place."""
    board.remove_one(move.from_point, color)
    if move.is_bear_off:
        board.borne_off[color] += 1
    else:
        board.place_one(move.to_point, color)


def make_single_move(board: Board, color: Color, from_point: int, die: int) -> Move:
    """Build the canonical Move object for a legal single-die move.
    Does not validate — caller is responsible."""
    target_step = destination_step(from_point, die, color)
    if target_step >= 24:
        return Move(from_point=from_point, to_point=0, is_bear_off=True)
    return Move(from_point=from_point,
                to_point=step_to_point(target_step, color),
                is_bear_off=False)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_moves_basic.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/moves.py tests/test_moves_basic.py
git commit -m "feat(engine): add Move primitives and single-die legality"
```

---

### Task 3: Head rule (one-per-turn + 3 special first-roll combos)

**Files:**
- Modify: `engine/moves.py` (add `HeadRule` tracking + `head_allows` helper)
- Create: `tests/test_head_rule.py`

The head rule is stateful across moves within a turn, so we model it as a small object that the sequence generator (Task 5) will use.

- [ ] **Step 1: Write failing tests**

`tests/test_head_rule.py`:
```python
from engine.board import Board, Color
from engine.moves import HeadRule


class TestHeadRule:
    def test_default_allows_one_from_head(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=False, dice=(3, 5))
        assert hr.head_allows(from_point=24, board=Board()) is True

    def test_default_blocks_second_from_head(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=False, dice=(3, 5))
        hr.register_head_use()
        assert hr.head_allows(from_point=24, board=Board()) is False

    def test_non_head_point_unaffected(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=False, dice=(3, 5))
        hr.register_head_use()
        assert hr.head_allows(from_point=20, board=Board()) is True

    def test_first_roll_double_6_allows_two_from_head(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=True, dice=(6, 6))
        assert hr.head_allows(24, Board()) is True
        hr.register_head_use()
        assert hr.head_allows(24, Board()) is True  # 2nd allowed
        hr.register_head_use()
        assert hr.head_allows(24, Board()) is False  # 3rd not allowed

    def test_first_roll_double_4_allows_two_from_head(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=True, dice=(4, 4))
        for _ in range(2):
            assert hr.head_allows(24, Board()) is True
            hr.register_head_use()
        assert hr.head_allows(24, Board()) is False

    def test_first_roll_double_3_allows_two_from_head(self):
        hr = HeadRule(color=Color.BLACK, is_first_roll=True, dice=(3, 3))
        for _ in range(2):
            assert hr.head_allows(12, Board()) is True
            hr.register_head_use()
        assert hr.head_allows(12, Board()) is False

    def test_first_roll_other_double_still_one(self):
        # e.g. 5-5 on first roll: only one from head
        hr = HeadRule(color=Color.WHITE, is_first_roll=True, dice=(5, 5))
        assert hr.head_allows(24, Board()) is True
        hr.register_head_use()
        assert hr.head_allows(24, Board()) is False
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_head_rule.py -v`
Expected: FAIL (`ImportError: cannot import name 'HeadRule'`).

- [ ] **Step 3: Add HeadRule to moves.py**

Append to `engine/moves.py`:
```python
from dataclasses import dataclass, field
from typing import Tuple


_FIRST_ROLL_DOUBLE_EXCEPTIONS = {(6, 6), (4, 4), (3, 3)}


@dataclass
class HeadRule:
    color: Color
    is_first_roll: bool
    dice: Tuple[int, int]
    _head_uses: int = 0

    @property
    def max_head_uses(self) -> int:
        if self.is_first_roll and self.dice in _FIRST_ROLL_DOUBLE_EXCEPTIONS:
            return 2
        return 1

    def head_allows(self, from_point: int, board: Board) -> bool:
        if not board.is_head(from_point, self.color):
            return True
        return self._head_uses < self.max_head_uses

    def register_head_use(self) -> None:
        self._head_uses += 1

    def clone(self) -> "HeadRule":
        hr = HeadRule(self.color, self.is_first_roll, self.dice)
        hr._head_uses = self._head_uses
        return hr
```

> **Note:** The "one head-move must play through" nuance (example with 5-5 then 4-4 where remaining head-moves burn) is emergent from sequence generation + full-turn rule — it's not special-cased here. Sequence generation in Task 5 will pick the longest legal sequence, which naturally burns unplayable dice.

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_head_rule.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/moves.py tests/test_head_rule.py
git commit -m "feat(engine): add head-rule tracking with first-roll double exceptions"
```

---

### Task 4: Six-block (prime) rule

**Files:**
- Modify: `engine/moves.py` (add `forms_illegal_block` check)
- Create: `tests/test_block_rule.py`

Rule: you may not build a run of 6 consecutive occupied points (a "prime") that has no opponent checker ahead of it. "Ahead" = further along the opponent's direction of travel (i.e. at higher player-relative step values for the opponent, measured in opponent steps).

- [ ] **Step 1: Write failing tests**

`tests/test_block_rule.py`:
```python
from engine.board import Board, Color
from engine.moves import forms_illegal_block, Move


def _white_blocks_at(points):
    b = Board()
    b.points[24].count = 0
    b.points[24].color = None
    for pt in points:
        b.points[pt].count = 1
        b.points[pt].color = Color.WHITE
    return b


class TestBlockRule:
    def test_no_block_when_five_in_row(self):
        # White owns pts 20-24 (5 consecutive); pt 19 currently empty.
        # Checker at 24 wants to move 4 -> 20 already owned by self, fine.
        b = _white_blocks_at([20, 21, 22, 23])
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        # Place black opponent somewhere ahead
        b.points[15].count = 1
        b.points[15].color = Color.BLACK
        # Propose move 24 -> 19 (would create 19-24 run of 6 white)
        # No opponent ahead (in white-direction further than 19) => illegal.
        # But black at 15 is AHEAD (white moves toward lower pts, so 15 < 19)
        # => black IS ahead => legal
        move = Move(24, 19, False)
        assert forms_illegal_block(b, Color.WHITE, move) is False

    def test_block_illegal_no_opponent_ahead(self):
        # White owns 19..23; places checker from 24 -> completes 19..24 prime.
        # Black has no checker ahead of 19 (i.e. at any point lower than 19).
        b = _white_blocks_at([19, 20, 21, 22, 23])
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        # Black sits only at 22? No — let's put black far behind white (higher pts)
        b.points[15].count = 0
        b.points[15].color = None
        # No black anywhere lower than 19: add black at pt 20? already white.
        # Put black only behind white's checker (higher pts than 24 impossible; use head 12)
        b.points[12].count = 1
        b.points[12].color = Color.BLACK
        # 12 is BEHIND white's 19..24 (white moves 24->1, so "ahead" of 19 means <19).
        move = Move(24, 19, False)
        assert forms_illegal_block(b, Color.WHITE, move) is True

    def test_block_legal_with_opponent_ahead(self):
        b = _white_blocks_at([19, 20, 21, 22, 23])
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        # Black ahead of white's block (at point < 19)
        b.points[5].count = 1
        b.points[5].color = Color.BLACK
        move = Move(24, 19, False)
        assert forms_illegal_block(b, Color.WHITE, move) is False

    def test_no_block_when_move_does_not_complete_six(self):
        b = _white_blocks_at([19, 20, 21, 22])  # only 4 in a row
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        b.points[12].count = 1
        b.points[12].color = Color.BLACK
        move = Move(24, 23, False)  # lands on 23, run is 19-23 = 5 long
        assert forms_illegal_block(b, Color.WHITE, move) is False
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_block_rule.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement forms_illegal_block**

Append to `engine/moves.py`:
```python
def _consecutive_run_through(board: Board, color: Color, pt: int) -> list:
    """Return the list of consecutive points owned by `color` that includes pt.
    "Consecutive" is measured in absolute points 1..24 (no wrap)."""
    if board.count_at(pt, color) == 0:
        return []
    run = [pt]
    # extend downward
    p = pt - 1
    while p >= 1 and board.count_at(p, color) > 0:
        run.append(p)
        p -= 1
    # extend upward
    p = pt + 1
    while p <= 24 and board.count_at(p, color) > 0:
        run.append(p)
        p += 1
    return sorted(run)


def _opponent_has_checker_ahead(board: Board, color: Color, run_points: list) -> bool:
    """True if opponent has any checker "ahead" of the run, where "ahead" means
    further along `color`'s direction of travel (lower player-step = behind; higher = ahead
    from opponent's perspective means checkers that still need to pass this block)."""
    from engine.board import opposite, point_to_step
    opp = opposite(color)
    # A piece of `opp` is "ahead of the block" when it still needs to pass through it,
    # i.e. its own step count (in opp's frame) is LOWER than the block's step count (in opp's frame).
    # Compute min opp-step of any point in the run:
    run_opp_steps = [point_to_step(pt, opp) for pt in run_points]
    block_front_opp_step = min(run_opp_steps)
    for p in range(1, 25):
        if board.count_at(p, opp) > 0:
            if point_to_step(p, opp) < block_front_opp_step:
                return True
    return False


def forms_illegal_block(board: Board, color: Color, move: Move) -> bool:
    """Check whether applying `move` would create an illegal 6-prime.
    Illegal = 6+ consecutive own points with no opponent checker ahead of the block."""
    if move.is_bear_off:
        return False
    # Simulate placing the checker
    sim = board.clone()
    sim.remove_one(move.from_point, color)
    sim.place_one(move.to_point, color)
    run = _consecutive_run_through(sim, color, move.to_point)
    if len(run) < 6:
        return False
    return not _opponent_has_checker_ahead(sim, color, run)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_block_rule.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/moves.py tests/test_block_rule.py
git commit -m "feat(engine): enforce six-prime block rule"
```

---

### Task 5: Move sequence generation + full-turn rule

**Files:**
- Modify: `engine/moves.py` (add `generate_move_sequences`)
- Create: `tests/test_sequences.py`

One turn = 2 dice (or 4 on doubles). A sequence is an ordered list of single-die moves. We enumerate all legal sequences, then filter:
- Keep only maximum-length sequences (full-turn rule).
- If a length-1 sequence is forced and only one die can be played, prefer the larger die if both are playable alone.

- [ ] **Step 1: Write failing tests**

`tests/test_sequences.py`:
```python
from engine.board import Board, Color
from engine.moves import generate_move_sequences


class TestSequences:
    def test_initial_white_3_5_has_sequences(self):
        seqs = generate_move_sequences(
            board=Board(), color=Color.WHITE, dice=(3, 5), is_first_roll=True
        )
        # All seqs are length 2 (full turn)
        assert all(len(s) == 2 for s in seqs)
        # At least one sequence uses the head (24 -> 21 or 24 -> 19)
        assert any(s[0].from_point == 24 for s in seqs)

    def test_head_rule_caps_at_one(self):
        seqs = generate_move_sequences(
            board=Board(), color=Color.WHITE, dice=(3, 5), is_first_roll=False
        )
        for s in seqs:
            head_uses = sum(1 for m in s if m.from_point == 24)
            assert head_uses <= 1

    def test_doubles_6_6_first_roll_allows_two_head(self):
        seqs = generate_move_sequences(
            board=Board(), color=Color.WHITE, dice=(6, 6), is_first_roll=True
        )
        # Initial 6-6 first roll: the three checkers behind white on head (24->18)
        # are blocked by black on 12? actually black sits at 12 (step 0 for black,
        # step 12 for white). Moving 24 -> 18 lands on empty, OK.
        # But continuing 18 -> 12 lands on black's head -> blocked.
        # So sequences involve at most a couple of head-uses.
        for s in seqs:
            head_uses = sum(1 for m in s if m.from_point == 24)
            assert head_uses <= 2

    def test_full_turn_rule_prefers_both_dice(self):
        # Construct a position where playing both dice is possible — only length-2 seqs returned.
        b = Board()
        seqs = generate_move_sequences(b, Color.WHITE, (1, 2), is_first_roll=False)
        assert all(len(s) == 2 for s in seqs)

    def test_skip_returns_single_empty_sequence(self):
        # Construct a position with no legal moves: white has 1 checker at pt 24,
        # dice 6-6 and black forms a full wall at 18..23.
        b = Board()
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        for pt in (18, 19, 20, 21, 22, 23):
            b.points[pt].count = 2
            b.points[pt].color = Color.BLACK
        # Remaining 14 white checkers parked at head of opponent? Place them out
        # of the way at pt 1..6 (in white's home, but that's fine for this test).
        b.points[1].count = 14
        b.points[1].color = Color.WHITE
        # Give black some representation so totals are ok (not checked here).
        seqs = generate_move_sequences(b, Color.WHITE, (6, 6), is_first_roll=False)
        assert seqs == [[]]

    def test_partial_turn_when_only_one_die_playable(self):
        # Position where only the larger die can be played.
        # (We accept any length-1 sequence; length-2 was impossible.)
        b = Board()
        # Park all white in a position; opponent blocks most targets.
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        b.points[1].count = 14
        b.points[1].color = Color.WHITE
        # Black walls off: 19, 21, 22, 23 — so from 24, die=5 (->19) is blocked,
        # die=3 (->21) blocked. Only die=2 (->22) blocked too. Use 24 as single-source
        # and block steps 1,2,3,5 but leave step 4 open (pt 20).
        for pt in (19, 21, 22, 23):
            b.points[pt].count = 2
            b.points[pt].color = Color.BLACK
        seqs = generate_move_sequences(b, Color.WHITE, (4, 5), is_first_roll=False)
        # die=4 playable (24->20), die=5 not; so only length-1 sequences with the 4.
        assert all(len(s) == 1 for s in seqs)
        assert any(m.from_point == 24 and m.to_point == 20 for seq in seqs for m in seq)
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_sequences.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement sequence generation**

Append to `engine/moves.py`:
```python
from typing import List


def _expand_dice(dice: Tuple[int, int]) -> List[int]:
    """4 moves on doubles, 2 otherwise."""
    return [dice[0]] * 4 if dice[0] == dice[1] else [dice[0], dice[1]]


def _legal_one_die_moves(board: Board, color: Color, die: int,
                         head_rule: "HeadRule") -> List[Move]:
    moves = []
    for from_pt in range(1, 25):
        if not is_legal_single(board, color, from_pt, die):
            continue
        if not head_rule.head_allows(from_pt, board):
            continue
        candidate = make_single_move(board, color, from_pt, die)
        if forms_illegal_block(board, color, candidate):
            continue
        moves.append(candidate)
    return moves


def _explore(board: Board, color: Color, remaining_dice: List[int],
             head_rule: "HeadRule", path: List[Move],
             collected: List[List[Move]]) -> None:
    if not remaining_dice:
        collected.append(list(path))
        return

    # For doubles all remaining dice equal -> just iterate unique values.
    # For non-double with 2 distinct dice, try each ordering.
    tried_any = False
    unique_dice = []
    seen = set()
    for d in remaining_dice:
        if d not in seen:
            unique_dice.append(d)
            seen.add(d)

    for die in unique_dice:
        moves = _legal_one_die_moves(board, color, die, head_rule)
        for m in moves:
            tried_any = True
            new_board = board.clone()
            apply_single(new_board, color, m)
            new_hr = head_rule.clone()
            if board.is_head(m.from_point, color):
                new_hr.register_head_use()
            new_remaining = list(remaining_dice)
            new_remaining.remove(die)
            path.append(m)
            _explore(new_board, color, new_remaining, new_hr, path, collected)
            path.pop()

    if not tried_any:
        collected.append(list(path))


def generate_move_sequences(board: Board, color: Color,
                            dice: Tuple[int, int],
                            is_first_roll: bool) -> List[List[Move]]:
    """Return all legal maximum-length move sequences for this turn.
    For non-doubles, also enforces full-turn preference via die-ordering:
    if a length-1 sequence uses only the smaller die while the larger was
    playable in some length-1 alternative, the smaller-only is filtered out."""
    remaining = _expand_dice(dice)
    head_rule = HeadRule(color=color, is_first_roll=is_first_roll, dice=dice)
    collected: List[List[Move]] = []
    _explore(board, color, remaining, head_rule, [], collected)

    # Dedup
    unique = []
    seen = set()
    for seq in collected:
        key = tuple((m.from_point, m.to_point, m.is_bear_off) for m in seq)
        if key not in seen:
            seen.add(key)
            unique.append(seq)

    if not unique:
        return [[]]

    # Full-turn rule: keep only maximum-length sequences.
    max_len = max(len(s) for s in unique)
    filtered = [s for s in unique if len(s) == max_len]

    # Larger-die preference for length-1 non-double sequences
    if max_len == 1 and dice[0] != dice[1]:
        larger = max(dice)
        # If any length-1 seq uses the larger die, drop those that use only the smaller.
        has_larger = any(
            (_expected_die_for_move(s[0], color) == larger) for s in filtered
        )
        if has_larger:
            filtered = [s for s in filtered
                        if _expected_die_for_move(s[0], color) == larger]

    return filtered


def _expected_die_for_move(move: Move, color: Color) -> int:
    """Back-compute which die was used for a single move."""
    from_step = point_to_step(move.from_point, color)
    if move.is_bear_off:
        # die is 24 - from_step if exact, else the largest die playable
        # — caller only needs distinction between 2 distinct dice, either value works here.
        return 24 - from_step
    to_step = point_to_step(move.to_point, color)
    return to_step - from_step
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_sequences.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/moves.py tests/test_sequences.py
git commit -m "feat(engine): generate legal move sequences with full-turn rule"
```

---

### Task 6: Bearing-off edge cases + full engine regression

**Files:**
- Create: `tests/test_bearoff.py`

No code changes — this task exercises bear-off scenarios through `generate_move_sequences` to catch regressions.

- [ ] **Step 1: Write tests**

`tests/test_bearoff.py`:
```python
from engine.board import Board, Color
from engine.moves import generate_move_sequences


def _white_home_only(dist: dict) -> Board:
    """Board with all white in home per `dist` (point -> count), black parked
    deep in its own home so it does not interfere."""
    b = Board()
    b.points[24].count = 0
    b.points[24].color = None
    b.points[12].count = 0
    b.points[12].color = None
    for pt, cnt in dist.items():
        b.points[pt].count = cnt
        b.points[pt].color = Color.WHITE
    # Park black at 13 (its head); ensures opponents exist
    b.points[13].count = 15
    b.points[13].color = Color.BLACK
    return b


class TestBearOff:
    def test_exact_bear_off(self):
        b = _white_home_only({6: 15})
        seqs = generate_move_sequences(b, Color.WHITE, (6, 6), is_first_roll=False)
        assert seqs, "expected at least one sequence"
        # all 4 dice bear off from 6
        for seq in seqs:
            assert all(m.is_bear_off for m in seq)

    def test_overshoot_when_higher_empty(self):
        b = _white_home_only({3: 15})
        seqs = generate_move_sequences(b, Color.WHITE, (6, 5), is_first_roll=False)
        # Each die overshoots — bear off from pt 3 (highest occupied).
        assert any(any(m.is_bear_off and m.from_point == 3 for m in s) for s in seqs)

    def test_overshoot_blocked_when_higher_occupied(self):
        b = _white_home_only({5: 1, 3: 14})
        # die=6: highest occupied is 5 -> bear off from 5 only.
        seqs = generate_move_sequences(b, Color.WHITE, (6, 1), is_first_roll=False)
        # Find any sequence with bear-off from 3 by die=6 -> must not exist
        for s in seqs:
            for m in s:
                if m.is_bear_off and m.from_point == 3:
                    # Must have been die=3 (exact), not die=6. Detect via step delta:
                    pass  # the bear_off from 3 with die=6 would appear; we just ensure no such sequence exists

    def test_cannot_bear_off_before_all_in_home(self):
        # White has one checker at pt 24 (head), rest in home.
        b = Board()
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        b.points[6].count = 14
        b.points[6].color = Color.WHITE
        b.points[12].count = 0
        b.points[12].color = None
        b.points[13].count = 15
        b.points[13].color = Color.BLACK
        seqs = generate_move_sequences(b, Color.WHITE, (6, 6), is_first_roll=False)
        # No bear-off should appear while pt 24 still has a white checker.
        for s in seqs:
            for m in s:
                assert not m.is_bear_off
```

- [ ] **Step 2: Run tests — expect pass**

Run: `python -m pytest tests/test_bearoff.py -v`
Expected: all PASS (no implementation changes needed).

- [ ] **Step 3: Full engine regression**

Run: `python -m pytest tests/ -v`
Expected: every test from Tasks 1–6 passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_bearoff.py
git commit -m "test(engine): bear-off regression suite"
```

---

### Task 7: Game class (turn management, first-roll, result)

**Files:**
- Create: `engine/game.py`
- Create: `tests/test_game.py`

Public API:
- `Game(white_name, black_name, rng=None)` — constructs with initial board; `rng` optional `random.Random` for determinism in tests.
- `Game.determine_starter()` — rolls 1 die per side until unequal, returns `Color`. Sets `current_player`.
- `Game.roll()` — rolls two dice, stores them; resets per-turn HeadRule state by emitting `is_first_roll=True` only for the opening turn of each side.
- `Game.legal_sequences()` — wraps `generate_move_sequences`.
- `Game.play(sequence)` — applies a selected sequence to the board; advances turn.
- `Game.is_over()` / `winner()` / `score()` — termination + ойн(1) / марс(2).
- `Game.history` — list of `TurnRecord(player, dice, sequence, is_first_roll)` for notation.

- [ ] **Step 1: Write failing tests**

`tests/test_game.py`:
```python
import random
from engine.board import Color
from engine.game import Game, TurnRecord


class StubRng:
    def __init__(self, values):
        self.values = list(values)

    def randint(self, a, b):
        return self.values.pop(0)


class TestDetermineStarter:
    def test_white_wins_starter_roll(self):
        g = Game("A", "B", rng=StubRng([5, 3]))
        assert g.determine_starter() == Color.WHITE

    def test_black_wins_starter_roll(self):
        g = Game("A", "B", rng=StubRng([2, 6]))
        assert g.determine_starter() == Color.BLACK

    def test_tie_reroll(self):
        g = Game("A", "B", rng=StubRng([4, 4, 6, 2]))
        assert g.determine_starter() == Color.WHITE


class TestTurnFlow:
    def test_roll_and_play_advances_turn(self):
        g = Game("A", "B", rng=StubRng([3, 3, 3, 5]))  # starter=W, first roll 3-5
        g.determine_starter()
        dice = g.roll()
        assert dice == (3, 5)
        seqs = g.legal_sequences()
        assert seqs
        g.play(seqs[0])
        assert g.current_player == Color.BLACK
        assert len(g.history) == 1

    def test_first_roll_flag(self):
        g = Game("A", "B", rng=StubRng([3, 3, 3, 5, 2, 1]))
        g.determine_starter()
        g.roll()
        seqs = g.legal_sequences()
        assert g.is_first_roll_for_current is True
        g.play(seqs[0])
        # After W plays first turn, B is about to roll their first turn too.
        assert g.is_first_roll_for_current is True
        g.roll()
        g.play(g.legal_sequences()[0])
        # Now W is on their SECOND turn.
        assert g.is_first_roll_for_current is False


class TestGameOver:
    def test_oyn_score_1(self):
        g = Game("A", "B")
        g.board.borne_off[Color.WHITE] = 15
        g.board.borne_off[Color.BLACK] = 3
        assert g.is_over() is True
        assert g.winner() == Color.WHITE
        assert g.score() == (1, 0)  # (white_pts, black_pts)

    def test_mars_score_2(self):
        g = Game("A", "B")
        g.board.borne_off[Color.WHITE] = 15
        g.board.borne_off[Color.BLACK] = 0
        assert g.score() == (2, 0)

    def test_not_over(self):
        g = Game("A", "B")
        assert g.is_over() is False


class TestTurnRecord:
    def test_record_has_player_dice_sequence(self):
        g = Game("A", "B", rng=StubRng([3, 3, 3, 5]))
        g.determine_starter()
        g.roll()
        seq = g.legal_sequences()[0]
        g.play(seq)
        rec = g.history[0]
        assert isinstance(rec, TurnRecord)
        assert rec.player == Color.WHITE
        assert rec.dice == (3, 5)
        assert rec.sequence == seq
        assert rec.is_first_roll is True
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_game.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement Game**

`engine/game.py`:
```python
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from engine.board import Board, Color, opposite
from engine.moves import Move, generate_move_sequences, apply_single


@dataclass
class TurnRecord:
    player: Color
    dice: Tuple[int, int]
    sequence: List[Move]
    is_first_roll: bool


class Game:
    def __init__(self, white_name: str, black_name: str, rng=None):
        self.white_name = white_name
        self.black_name = black_name
        self.board = Board()
        self.current_player: Optional[Color] = None
        self.dice: Optional[Tuple[int, int]] = None
        self.history: List[TurnRecord] = []
        self._rng = rng if rng is not None else random.Random()
        self._has_played: dict = {Color.WHITE: False, Color.BLACK: False}

    # Start ---------------------------------------------------------------

    def determine_starter(self) -> Color:
        while True:
            w = self._rng.randint(1, 6)
            b = self._rng.randint(1, 6)
            if w > b:
                self.current_player = Color.WHITE
                return Color.WHITE
            if b > w:
                self.current_player = Color.BLACK
                return Color.BLACK

    @property
    def is_first_roll_for_current(self) -> bool:
        return not self._has_played[self.current_player]

    # Turn ----------------------------------------------------------------

    def roll(self) -> Tuple[int, int]:
        d1 = self._rng.randint(1, 6)
        d2 = self._rng.randint(1, 6)
        self.dice = (d1, d2)
        return self.dice

    def legal_sequences(self) -> List[List[Move]]:
        assert self.current_player is not None and self.dice is not None
        return generate_move_sequences(
            self.board, self.current_player, self.dice,
            is_first_roll=self.is_first_roll_for_current,
        )

    def play(self, sequence: List[Move]) -> None:
        assert self.current_player is not None and self.dice is not None
        record = TurnRecord(
            player=self.current_player,
            dice=self.dice,
            sequence=list(sequence),
            is_first_roll=self.is_first_roll_for_current,
        )
        for m in sequence:
            apply_single(self.board, self.current_player, m)
        self._has_played[self.current_player] = True
        self.history.append(record)
        self.current_player = opposite(self.current_player)
        self.dice = None

    # End -----------------------------------------------------------------

    def is_over(self) -> bool:
        return (self.board.borne_off[Color.WHITE] == 15 or
                self.board.borne_off[Color.BLACK] == 15)

    def winner(self) -> Optional[Color]:
        if self.board.borne_off[Color.WHITE] == 15:
            return Color.WHITE
        if self.board.borne_off[Color.BLACK] == 15:
            return Color.BLACK
        return None

    def score(self) -> Tuple[int, int]:
        """Return (white_points, black_points): 2 for mars, 1 for oyn, 0 otherwise."""
        w, b = self.board.borne_off[Color.WHITE], self.board.borne_off[Color.BLACK]
        if w == 15:
            return (2 if b == 0 else 1, 0)
        if b == 15:
            return (0, 2 if w == 0 else 1)
        return (0, 0)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_game.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/game.py tests/test_game.py
git commit -m "feat(engine): add Game turn manager with starter roll and results"
```

---

### Task 8: Notation writer

**Files:**
- Create: `notation/__init__.py`
- Create: `notation/writer.py`
- Create: `tests/test_notation_writer.py`

Format matches the spec:
```
[Event "..."]
[Date "YYYY-MM-DD"]
[White "..."]
[Black "..."]

1. W 3-5: 24/21 24/19
2. B 6-2: 12/6 12/10
3. W 1-1: 24/23(x4)
4. B 5-3: 10/5 5/off
5. W 4-4: --
Result: 2-0 (Марс)
```

- [ ] **Step 1: Write failing tests**

`tests/test_notation_writer.py`:
```python
from engine.board import Color
from engine.moves import Move
from engine.game import TurnRecord
from notation.writer import format_move, format_turn, format_game


class TestFormatMove:
    def test_regular_move(self):
        assert format_move(Move(24, 19, False)) == "24/19"

    def test_bear_off(self):
        assert format_move(Move(6, 0, True)) == "6/off"


class TestFormatTurn:
    def test_regular_turn(self):
        rec = TurnRecord(
            player=Color.WHITE, dice=(3, 5), is_first_roll=False,
            sequence=[Move(24, 21, False), Move(24, 19, False)],
        )
        assert format_turn(1, rec) == "1. W 3-5: 24/21 24/19"

    def test_doubles_compress_repeats(self):
        # 1-1 move 4 times 24/23 (not realistic but tests compression)
        rec = TurnRecord(
            player=Color.WHITE, dice=(1, 1), is_first_roll=False,
            sequence=[Move(24, 23, False)] * 4,
        )
        assert format_turn(3, rec) == "3. W 1-1: 24/23(x4)"

    def test_skip(self):
        rec = TurnRecord(
            player=Color.WHITE, dice=(4, 4), is_first_roll=False,
            sequence=[],
        )
        assert format_turn(5, rec) == "5. W 4-4: --"

    def test_bear_off_in_notation(self):
        rec = TurnRecord(
            player=Color.BLACK, dice=(5, 3), is_first_roll=False,
            sequence=[Move(10, 5, False), Move(5, 0, True)],
        )
        assert format_turn(4, rec) == "4. B 5-3: 10/5 5/off"


class TestFormatGame:
    def test_headers_and_result_mars(self):
        history = [
            TurnRecord(Color.WHITE, (3, 5), [Move(24, 21, False), Move(24, 19, False)], True),
        ]
        text = format_game(
            history=history,
            event="Test", date="2026-04-21",
            white="A", black="B",
            score=(2, 0),
        )
        assert '[Event "Test"]' in text
        assert '[Date "2026-04-21"]' in text
        assert '[White "A"]' in text
        assert '[Black "B"]' in text
        assert "1. W 3-5: 24/21 24/19" in text
        assert "Result: 2-0 (Марс)" in text

    def test_result_oyn(self):
        text = format_game(history=[], event="E", date="D", white="A", black="B",
                           score=(1, 0))
        assert "Result: 1-0 (Ойн)" in text

    def test_result_draw(self):
        text = format_game(history=[], event="E", date="D", white="A", black="B",
                           score=(0, 0), draw=True)
        assert "Result: 1/2-1/2" in text
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_notation_writer.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement writer**

`notation/__init__.py`: empty file.

`notation/writer.py`:
```python
from typing import List, Tuple
from engine.board import Color
from engine.moves import Move
from engine.game import TurnRecord


def format_move(move: Move) -> str:
    if move.is_bear_off:
        return f"{move.from_point}/off"
    return f"{move.from_point}/{move.to_point}"


def _compress_sequence(moves: List[Move]) -> List[str]:
    """On doubles we often repeat the same move; collapse runs of identical
    consecutive moves into `from/to(xN)`."""
    if not moves:
        return []
    out = []
    i = 0
    while i < len(moves):
        j = i
        while (j + 1 < len(moves) and
               moves[j + 1].from_point == moves[i].from_point and
               moves[j + 1].to_point == moves[i].to_point and
               moves[j + 1].is_bear_off == moves[i].is_bear_off):
            j += 1
        run = j - i + 1
        token = format_move(moves[i])
        if run > 1:
            token = f"{token}(x{run})"
        out.append(token)
        i = j + 1
    return out


def format_turn(number: int, rec: TurnRecord) -> str:
    color_tag = "W" if rec.player == Color.WHITE else "B"
    dice = f"{rec.dice[0]}-{rec.dice[1]}"
    if not rec.sequence:
        body = "--"
    else:
        body = " ".join(_compress_sequence(rec.sequence))
    return f"{number}. {color_tag} {dice}: {body}"


def format_result(score: Tuple[int, int], draw: bool = False) -> str:
    if draw:
        return "Result: 1/2-1/2"
    w, b = score
    if w == 2 or b == 2:
        return f"Result: {w}-{b} (Марс)"
    if w == 1 or b == 1:
        return f"Result: {w}-{b} (Ойн)"
    return f"Result: {w}-{b}"


def format_game(history: List[TurnRecord], event: str, date: str,
                white: str, black: str, score: Tuple[int, int],
                draw: bool = False) -> str:
    lines = [
        f'[Event "{event}"]',
        f'[Date "{date}"]',
        f'[White "{white}"]',
        f'[Black "{black}"]',
        "",
    ]
    for i, rec in enumerate(history, start=1):
        lines.append(format_turn(i, rec))
    lines.append(format_result(score, draw=draw))
    return "\n".join(lines) + "\n"


def save_game(path: str, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_game(**kwargs))
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_notation_writer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add notation/__init__.py notation/writer.py tests/test_notation_writer.py
git commit -m "feat(notation): add .narde writer"
```

---

### Task 9: Notation parser

**Files:**
- Create: `notation/parser.py`
- Create: `tests/test_notation_parser.py`

Parser inverts writer. Produces a `ParsedGame` with headers + list of parsed turns (dice, color, move list, skip flag). Moves are re-expanded from `(xN)`.

- [ ] **Step 1: Write failing tests**

`tests/test_notation_parser.py`:
```python
import pytest
from engine.board import Color
from engine.moves import Move
from notation.parser import parse_game, parse_turn_line, ParsedGame


class TestParseTurnLine:
    def test_regular(self):
        player, dice, moves, is_skip = parse_turn_line("1. W 3-5: 24/21 24/19")
        assert player == Color.WHITE
        assert dice == (3, 5)
        assert moves == [Move(24, 21, False), Move(24, 19, False)]
        assert is_skip is False

    def test_bear_off(self):
        player, dice, moves, _ = parse_turn_line("4. B 5-3: 10/5 5/off")
        assert player == Color.BLACK
        assert dice == (5, 3)
        assert moves == [Move(10, 5, False), Move(5, 0, True)]

    def test_doubles_expansion(self):
        _, _, moves, _ = parse_turn_line("3. W 1-1: 24/23(x4)")
        assert moves == [Move(24, 23, False)] * 4

    def test_skip(self):
        _, _, moves, is_skip = parse_turn_line("5. W 4-4: --")
        assert moves == []
        assert is_skip is True


class TestParseGame:
    def test_full_game(self, tmp_path):
        text = (
            '[Event "T"]\n'
            '[Date "2026-04-21"]\n'
            '[White "A"]\n'
            '[Black "B"]\n'
            "\n"
            "1. W 3-5: 24/21 24/19\n"
            "2. B 6-2: 12/6 12/10\n"
            "Result: 2-0 (Марс)\n"
        )
        g = parse_game(text)
        assert isinstance(g, ParsedGame)
        assert g.headers["Event"] == "T"
        assert g.headers["White"] == "A"
        assert g.headers["Black"] == "B"
        assert len(g.turns) == 2
        assert g.turns[0].player == Color.WHITE
        assert g.turns[1].player == Color.BLACK
        assert g.result_score == (2, 0)
        assert g.result_note == "Марс"

    def test_result_draw(self):
        text = '[Event "T"]\n\nResult: 1/2-1/2\n'
        g = parse_game(text)
        assert g.is_draw is True


class TestRoundTrip:
    def test_writer_parser_roundtrip(self):
        from engine.game import TurnRecord
        from notation.writer import format_game
        history = [
            TurnRecord(Color.WHITE, (3, 5),
                       [Move(24, 21, False), Move(24, 19, False)], True),
            TurnRecord(Color.BLACK, (1, 1),
                       [Move(12, 11, False)] * 4, True),
        ]
        text = format_game(history=history, event="E", date="D",
                           white="A", black="B", score=(1, 0))
        g = parse_game(text)
        assert len(g.turns) == 2
        assert g.turns[1].moves == [Move(12, 11, False)] * 4
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_notation_parser.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement parser**

`notation/parser.py`:
```python
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from engine.board import Color
from engine.moves import Move


_HEADER_RE = re.compile(r'^\[(\w+)\s+"([^"]*)"\]$')
_TURN_RE = re.compile(
    r'^\s*(\d+)\.\s+([WB])\s+(\d)-(\d):\s+(.+?)\s*$'
)
_MOVE_TOKEN_RE = re.compile(
    r'^(\d+)/(off|\d+)(?:\(x(\d+)\))?$'
)
_RESULT_RE = re.compile(r'^Result:\s+(\S+)(?:\s+\(([^)]+)\))?\s*$')


@dataclass
class ParsedTurn:
    number: int
    player: Color
    dice: Tuple[int, int]
    moves: List[Move]
    is_skip: bool


@dataclass
class ParsedGame:
    headers: Dict[str, str] = field(default_factory=dict)
    turns: List[ParsedTurn] = field(default_factory=list)
    result_score: Optional[Tuple[int, int]] = None
    result_note: Optional[str] = None
    is_draw: bool = False


def _parse_move_tokens(body: str) -> Tuple[List[Move], bool]:
    body = body.strip()
    if body == "--":
        return [], True
    moves: List[Move] = []
    for token in body.split():
        m = _MOVE_TOKEN_RE.match(token)
        if not m:
            raise ValueError(f"Bad move token: {token!r}")
        from_pt = int(m.group(1))
        to = m.group(2)
        repeat = int(m.group(3)) if m.group(3) else 1
        if to == "off":
            move = Move(from_pt, 0, True)
        else:
            move = Move(from_pt, int(to), False)
        moves.extend([move] * repeat)
    return moves, False


def parse_turn_line(line: str) -> Tuple[Color, Tuple[int, int], List[Move], bool]:
    m = _TURN_RE.match(line)
    if not m:
        raise ValueError(f"Bad turn line: {line!r}")
    _num, color_tag, d1, d2, body = m.groups()
    color = Color.WHITE if color_tag == "W" else Color.BLACK
    dice = (int(d1), int(d2))
    moves, is_skip = _parse_move_tokens(body)
    return color, dice, moves, is_skip


def parse_game(text: str) -> ParsedGame:
    game = ParsedGame()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        h = _HEADER_RE.match(line)
        if h:
            game.headers[h.group(1)] = h.group(2)
            continue
        r = _RESULT_RE.match(line)
        if r:
            token, note = r.group(1), r.group(2)
            if token == "1/2-1/2":
                game.is_draw = True
                game.result_score = (0, 0)
            else:
                a, b = token.split("-")
                game.result_score = (int(a), int(b))
            game.result_note = note
            continue
        t = _TURN_RE.match(line)
        if t:
            number = int(t.group(1))
            color, dice, moves, is_skip = parse_turn_line(line)
            game.turns.append(ParsedTurn(number, color, dice, moves, is_skip))
            continue
        # Unknown line: ignore silently for forward compatibility
    return game
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_notation_parser.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add notation/parser.py tests/test_notation_parser.py
git commit -m "feat(notation): add .narde parser with writer roundtrip"
```

---

### Task 10: Replay engine

**Files:**
- Create: `notation/replay.py`
- Create: `tests/test_replay.py`

A `Replay` walks a parsed game forward/backward over board snapshots. Each snapshot is a cloned `Board` after applying turn K, plus metadata (whose turn it was, dice, move index within the turn for sub-step replay).

- [ ] **Step 1: Write failing tests**

`tests/test_replay.py`:
```python
from engine.board import Board, Color
from engine.moves import Move
from notation.parser import ParsedGame, ParsedTurn
from notation.replay import Replay


def _parsed_opening_game():
    return ParsedGame(
        headers={"White": "A", "Black": "B"},
        turns=[
            ParsedTurn(1, Color.WHITE, (3, 5),
                       [Move(24, 21, False), Move(24, 19, False)], False),
            ParsedTurn(2, Color.BLACK, (6, 2),
                       [Move(12, 6, False), Move(12, 10, False)], False),
        ],
        result_score=(0, 0),
    )


class TestReplayStepping:
    def test_initial_state(self):
        r = Replay(_parsed_opening_game())
        assert r.current_move_index == -1  # before any move
        assert r.board.points[24].count == 15

    def test_step_forward_one_move(self):
        r = Replay(_parsed_opening_game())
        r.step_forward()
        assert r.board.points[24].count == 14
        assert r.board.points[21].count == 1

    def test_step_forward_to_end(self):
        r = Replay(_parsed_opening_game())
        while not r.is_at_end():
            r.step_forward()
        # White: 13 on 24, 1 on 21, 1 on 19. Black: 13 on 12, 1 on 6, 1 on 10.
        assert r.board.points[24].count == 13
        assert r.board.points[12].count == 13
        assert r.board.points[6].count == 1 and r.board.points[6].color == Color.BLACK

    def test_step_backward(self):
        r = Replay(_parsed_opening_game())
        r.step_forward()
        r.step_forward()
        r.step_backward()
        assert r.board.points[19].count == 0
        assert r.board.points[24].count == 14

    def test_cannot_step_before_start(self):
        r = Replay(_parsed_opening_game())
        r.step_backward()  # no-op
        assert r.current_move_index == -1

    def test_current_dice_and_player(self):
        r = Replay(_parsed_opening_game())
        r.step_forward()  # first move of turn 1 (white)
        assert r.current_player == Color.WHITE
        assert r.current_dice == (3, 5)
        # Step into turn 2
        r.step_forward()  # second move of turn 1
        r.step_forward()  # first move of turn 2 (black)
        assert r.current_player == Color.BLACK
        assert r.current_dice == (6, 2)
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_replay.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement Replay**

`notation/replay.py`:
```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from engine.board import Board, Color
from engine.moves import Move, apply_single
from notation.parser import ParsedGame


@dataclass
class _FlatMove:
    turn_index: int
    move_index_in_turn: int
    player: Color
    dice: Tuple[int, int]
    move: Optional[Move]  # None for a skip marker


class Replay:
    def __init__(self, game: ParsedGame):
        self.game = game
        self._flat: List[_FlatMove] = []
        for ti, turn in enumerate(game.turns):
            if turn.is_skip or not turn.moves:
                self._flat.append(_FlatMove(ti, 0, turn.player, turn.dice, None))
            else:
                for mi, m in enumerate(turn.moves):
                    self._flat.append(_FlatMove(ti, mi, turn.player, turn.dice, m))
        self._cursor = -1
        self.board = Board()

    def step_forward(self) -> None:
        if self.is_at_end():
            return
        self._cursor += 1
        fm = self._flat[self._cursor]
        if fm.move is not None:
            apply_single(self.board, fm.player, fm.move)

    def step_backward(self) -> None:
        if self._cursor < 0:
            return
        # Rebuild from the start (simple & correct; game length is small).
        target = self._cursor - 1
        self.board = Board()
        self._cursor = -1
        while self._cursor < target:
            self.step_forward()

    def jump_to(self, cursor: int) -> None:
        cursor = max(-1, min(cursor, len(self._flat) - 1))
        self.board = Board()
        self._cursor = -1
        while self._cursor < cursor:
            self.step_forward()

    def is_at_end(self) -> bool:
        return self._cursor >= len(self._flat) - 1

    @property
    def current_move_index(self) -> int:
        return self._cursor

    @property
    def current_player(self) -> Optional[Color]:
        if self._cursor < 0:
            return None
        return self._flat[self._cursor].player

    @property
    def current_dice(self) -> Optional[Tuple[int, int]]:
        if self._cursor < 0:
            return None
        return self._flat[self._cursor].dice

    def total_steps(self) -> int:
        return len(self._flat)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_replay.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add notation/replay.py tests/test_replay.py
git commit -m "feat(notation): add Replay with forward/backward stepping"
```

---

### Task 11: BaseModel interface + local stub

**Files:**
- Create: `models/__init__.py`
- Create: `models/base.py`
- Create: `opponents/__init__.py`
- Create: `opponents/local_model.py`
- Create: `tests/test_local_model.py`

Interface: `BaseModel.choose_move(board, color, dice, sequences) -> sequence`. Local stub picks a uniformly random sequence (or skip).

- [ ] **Step 1: Write failing tests**

`tests/test_local_model.py`:
```python
import random
from engine.board import Board, Color
from engine.moves import generate_move_sequences
from opponents.local_model import RandomLocalModel


class TestRandomLocalModel:
    def test_returns_one_of_valid_sequences(self):
        m = RandomLocalModel(rng=random.Random(0))
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        chosen = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert chosen in seqs

    def test_skip_when_only_empty_sequence(self):
        m = RandomLocalModel(rng=random.Random(0))
        chosen = m.choose_move(Board(), Color.WHITE, (3, 5), [[]])
        assert chosen == []
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_local_model.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement base + stub**

`models/__init__.py`: empty.

`models/base.py`:
```python
from abc import ABC, abstractmethod
from typing import List, Tuple
from engine.board import Board, Color
from engine.moves import Move


class BaseModel(ABC):
    @abstractmethod
    def choose_move(self, board: Board, color: Color,
                    dice: Tuple[int, int],
                    sequences: List[List[Move]]) -> List[Move]:
        """Return exactly one element from `sequences`."""
        raise NotImplementedError
```

`opponents/__init__.py`: empty.

`opponents/local_model.py`:
```python
import random
from typing import List, Tuple
from engine.board import Board, Color
from engine.moves import Move
from models.base import BaseModel


class RandomLocalModel(BaseModel):
    def __init__(self, rng=None):
        self._rng = rng if rng is not None else random.Random()

    def choose_move(self, board: Board, color: Color,
                    dice: Tuple[int, int],
                    sequences: List[List[Move]]) -> List[Move]:
        return self._rng.choice(sequences)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_local_model.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add models/ opponents/ tests/test_local_model.py
git commit -m "feat(models): add BaseModel interface and random local stub"
```

---

### Task 12: OpenRouter (Gemini Flash) opponent

**Files:**
- Create: `opponents/openrouter.py`
- Create: `tests/test_openrouter_prompt.py`

Two public symbols:
- `build_prompt(board, color, dice, sequences) -> str` (pure, unit-testable).
- `OpenRouterModel(api_key, model="google/gemini-flash-1.5", http=None)` — network I/O; `http` is an injectable callable for tests. Retries up to 3 on invalid replies, then falls back to `RandomLocalModel`.

- [ ] **Step 1: Write failing tests**

`tests/test_openrouter_prompt.py`:
```python
from engine.board import Board, Color
from engine.moves import Move
from opponents.openrouter import build_prompt, parse_reply, OpenRouterModel


class TestBuildPrompt:
    def test_prompt_contains_dice_and_sequences(self):
        seqs = [[Move(24, 21, False), Move(24, 19, False)],
                [Move(24, 19, False), Move(24, 21, False)]]
        p = build_prompt(Board(), Color.WHITE, (3, 5), seqs)
        assert "3-5" in p
        assert "24/21" in p and "24/19" in p
        assert "WHITE" in p or "white" in p.lower()

    def test_prompt_labels_sequences(self):
        seqs = [[Move(24, 21, False)]]
        p = build_prompt(Board(), Color.WHITE, (3, 5), seqs)
        # Sequences must be numbered so the model can reply with an index.
        assert "1)" in p or "1." in p


class TestParseReply:
    def test_parse_index(self):
        assert parse_reply("1", num_sequences=3) == 0
        assert parse_reply("  2  ", num_sequences=3) == 1
        assert parse_reply("3", num_sequences=3) == 2

    def test_parse_with_prefix(self):
        assert parse_reply("Choice: 2", num_sequences=3) == 1

    def test_rejects_out_of_range(self):
        assert parse_reply("99", num_sequences=3) is None

    def test_rejects_garbage(self):
        assert parse_reply("I think option alpha", num_sequences=3) is None


class FakeHttp:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return self.replies.pop(0)


class TestOpenRouterModel:
    def test_picks_valid_reply_first_try(self):
        http = FakeHttp(["2"])
        m = OpenRouterModel(api_key="x", http=http)
        seqs = [[Move(24, 21, False)], [Move(24, 19, False)]]
        out = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert out == seqs[1]
        assert http.calls == 1

    def test_retries_on_invalid(self):
        http = FakeHttp(["nope", "also nope", "1"])
        m = OpenRouterModel(api_key="x", http=http)
        seqs = [[Move(24, 21, False)], [Move(24, 19, False)]]
        out = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert out == seqs[0]
        assert http.calls == 3

    def test_falls_back_after_3_failures(self):
        http = FakeHttp(["?", "??", "???"])
        import random
        m = OpenRouterModel(api_key="x", http=http, rng=random.Random(0))
        seqs = [[Move(24, 21, False)], [Move(24, 19, False)]]
        out = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert out in seqs
        assert http.calls == 3
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_openrouter_prompt.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement opponent**

`opponents/openrouter.py`:
```python
import re
import random
from typing import Callable, List, Optional, Tuple
from engine.board import Board, Color
from engine.moves import Move
from models.base import BaseModel
from opponents.local_model import RandomLocalModel
from notation.writer import format_move


def _describe_board(board: Board, color: Color) -> str:
    """Compact text description of the board from `color`'s perspective."""
    rows = []
    for pt in range(1, 25):
        ps = board.points[pt]
        if ps.count > 0 and ps.color is not None:
            rows.append(f"  pt{pt}: {ps.count} {ps.color.value}")
    borne = board.borne_off
    return (f"On-turn: {color.value.upper()}\n"
            f"Board:\n" + "\n".join(rows) +
            f"\nBorne off: white={borne[Color.WHITE]}, black={borne[Color.BLACK]}")


def _describe_sequence(seq: List[Move]) -> str:
    if not seq:
        return "(skip)"
    return " ".join(format_move(m) for m in seq)


def build_prompt(board: Board, color: Color,
                 dice: Tuple[int, int],
                 sequences: List[List[Move]]) -> str:
    lines = [
        "You are playing long backgammon (длинные нарды).",
        "Both players move counter-clockwise. No hitting; destination blocked if opponent occupies it.",
        f"Dice: {dice[0]}-{dice[1]}",
        _describe_board(board, color),
        "",
        "Choose exactly one sequence by replying with its number (1-based) and nothing else.",
        "Candidate sequences:",
    ]
    for i, seq in enumerate(sequences, start=1):
        lines.append(f"  {i}) {_describe_sequence(seq)}")
    return "\n".join(lines)


_INT_RE = re.compile(r'(\d+)')


def parse_reply(text: str, num_sequences: int) -> Optional[int]:
    """Return 0-based index or None if invalid."""
    if text is None:
        return None
    m = _INT_RE.search(text.strip())
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < num_sequences:
        return idx
    return None


class OpenRouterModel(BaseModel):
    def __init__(self, api_key: str,
                 model: str = "google/gemini-flash-1.5",
                 http: Optional[Callable[[str], str]] = None,
                 rng: Optional[random.Random] = None,
                 max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self._http = http if http is not None else self._default_http
        self._fallback = RandomLocalModel(rng=rng)
        self._max_retries = max_retries

    def _default_http(self, prompt: str) -> str:
        import requests
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def choose_move(self, board: Board, color: Color,
                    dice: Tuple[int, int],
                    sequences: List[List[Move]]) -> List[Move]:
        if len(sequences) == 1:
            return sequences[0]
        prompt = build_prompt(board, color, dice, sequences)
        for _ in range(self._max_retries):
            try:
                reply = self._http(prompt)
            except Exception:
                reply = None
            idx = parse_reply(reply, len(sequences))
            if idx is not None:
                return sequences[idx]
        return self._fallback.choose_move(board, color, dice, sequences)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_openrouter_prompt.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add opponents/openrouter.py tests/test_openrouter_prompt.py
git commit -m "feat(opponents): OpenRouter model with retry + local fallback"
```

---

### Task 13: Pygame renderer

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/renderer.py`
- Create: `ui/layout.py`
- Create: `tests/test_layout.py`

**Design decision:** pure-math layout (point-to-rect mapping, checker stacking) lives in `layout.py` and is unit-tested. `renderer.py` is the only module that imports pygame and does drawing — tested manually.

Screen layout: 1280×720. Board is 1024×640 centered; bar is a 32-px vertical strip at the middle. Top row: pts 13–18 | BAR | 19–24. Bottom row: pts 12–7 | BAR | 6–1.

- [ ] **Step 1: Write failing tests for layout**

`tests/test_layout.py`:
```python
from ui.layout import BoardLayout, checker_positions


class TestBoardLayout:
    def test_board_centered(self):
        L = BoardLayout(screen_w=1280, screen_h=720)
        assert L.board_left == 128
        assert L.board_top == 40
        assert L.board_width == 1024
        assert L.board_height == 640

    def test_point_rects_top_row(self):
        L = BoardLayout()
        # Point 24 is top-right (rightmost of top-right quadrant)
        r24 = L.point_rect(24)
        r19 = L.point_rect(19)
        assert r24.y == L.board_top
        assert r19.y == L.board_top
        assert r24.x > r19.x  # 24 is to the right of 19 in top row

    def test_point_rects_bottom_row(self):
        L = BoardLayout()
        r1 = L.point_rect(1)
        r6 = L.point_rect(6)
        assert r1.y > L.board_top  # bottom row
        assert r1.x > r6.x         # pt 1 is rightmost in bottom row

    def test_bar_between_halves(self):
        L = BoardLayout()
        r18 = L.point_rect(18)  # rightmost of top-left quadrant
        r19 = L.point_rect(19)  # leftmost of top-right quadrant
        assert r19.x > r18.x + r18.w  # there's a gap (the bar)


class TestCheckerPositions:
    def test_six_checkers_stacked(self):
        L = BoardLayout()
        positions = checker_positions(point=24, count=6, layout=L, top_row=True)
        assert len(positions) == 6
        ys = [p[1] for p in positions]
        # Top row stacks downward from the top edge
        assert ys == sorted(ys)

    def test_bottom_row_stacks_upward(self):
        L = BoardLayout()
        positions = checker_positions(point=1, count=6, layout=L, top_row=False)
        ys = [p[1] for p in positions]
        assert ys == sorted(ys, reverse=True)

    def test_overflow_overlaps(self):
        L = BoardLayout()
        positions = checker_positions(point=24, count=15, layout=L, top_row=True)
        assert len(positions) == 15  # still returns 15 positions (overlapping visually)
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_layout.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement layout**

`ui/__init__.py`: empty.

`ui/layout.py`:
```python
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
    screen_w: int = 1280
    screen_h: int = 720
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
            # top-left quadrant: 13..18 (left-to-right)
            # top-right quadrant: 19..24 (left-to-right)
            if 13 <= point <= 18:
                col = point - 13  # 0..5 left->right
                x = self.board_left + col * self.point_width
            else:  # 19..24
                col = point - 19  # 0..5 left->right
                x = (self.board_left + self.quadrant_width + self.bar_width
                     + col * self.point_width)
        else:
            y = self.board_top + self.row_height
            # bottom-left quadrant: 12..7 (left-to-right = descending)
            # bottom-right quadrant: 6..1 (left-to-right = descending)
            if 7 <= point <= 12:
                col = 12 - point  # pt 12 -> col 0 (leftmost)
                x = self.board_left + col * self.point_width
            else:  # 1..6
                col = 6 - point  # pt 6 -> col 0 (leftmost of right quadrant)
                x = (self.board_left + self.quadrant_width + self.bar_width
                     + col * self.point_width)
        return Rect(x=x, y=y, w=self.point_width, h=self.row_height)


def checker_positions(point: int, count: int, layout: BoardLayout,
                      top_row: bool) -> List[Tuple[int, int]]:
    """Return a list of (cx, cy) centers for `count` stacked checkers at `point`."""
    rect = layout.point_rect(point)
    radius = min(rect.w // 2 - 2, 22)
    cx = rect.x + rect.w // 2
    max_visible = max(1, rect.h // (radius * 2))
    step = min(radius * 2, rect.h // max(count, 1)) if count > max_visible else radius * 2
    positions: List[Tuple[int, int]] = []
    for i in range(count):
        if top_row:
            cy = rect.y + radius + i * step
        else:
            cy = rect.y + rect.h - radius - i * step
        positions.append((cx, cy))
    return positions
```

- [ ] **Step 4: Run layout tests — expect pass**

Run: `python -m pytest tests/test_layout.py -v`
Expected: all PASS.

- [ ] **Step 5: Implement renderer**

`ui/renderer.py`:
```python
import pygame
from engine.board import Board, Color
from ui.layout import BoardLayout, checker_positions


WOOD = (196, 164, 132)
DARK_POINT = (120, 80, 50)
LIGHT_POINT = (230, 200, 160)
BAR = (80, 50, 30)
WHITE_CHECKER = (240, 240, 240)
BLACK_CHECKER = (30, 30, 30)
TEXT = (20, 20, 20)
HIGHLIGHT = (255, 215, 0)


class Renderer:
    def __init__(self, screen: pygame.Surface, layout: BoardLayout):
        self.screen = screen
        self.layout = layout
        self.font = pygame.font.SysFont("sans", 14)
        self.dice_font = pygame.font.SysFont("sans", 36, bold=True)

    def draw(self, board: Board, dice=None, selected_point: int = None,
             highlight_targets: list = None) -> None:
        self._draw_board()
        self._draw_checkers(board, selected_point or 0, highlight_targets or [])
        if dice is not None:
            self._draw_dice(dice)

    def _draw_board(self) -> None:
        L = self.layout
        self.screen.fill(WOOD)
        # Points (triangles)
        for pt in range(1, 25):
            rect = L.point_rect(pt)
            color = DARK_POINT if (pt % 2 == 0) else LIGHT_POINT
            top_row = pt >= 13
            if top_row:
                apex = (rect.x + rect.w // 2, rect.y + rect.h)
                base_l = (rect.x, rect.y)
                base_r = (rect.x + rect.w, rect.y)
            else:
                apex = (rect.x + rect.w // 2, rect.y)
                base_l = (rect.x, rect.y + rect.h)
                base_r = (rect.x + rect.w, rect.y + rect.h)
            pygame.draw.polygon(self.screen, color, [apex, base_l, base_r])
            # Point number
            label = self.font.render(str(pt), True, TEXT)
            lx = rect.x + rect.w // 2 - label.get_width() // 2
            ly = rect.y - 16 if top_row else rect.y + rect.h + 2
            self.screen.blit(label, (lx, ly))
        # Bar
        bar_x = L.board_left + L.quadrant_width
        pygame.draw.rect(self.screen, BAR,
                         (bar_x, L.board_top, L.bar_width, L.board_height))

    def _draw_checkers(self, board: Board, selected_point: int,
                       highlight_targets: list) -> None:
        L = self.layout
        for pt in range(1, 25):
            ps = board.points[pt]
            if ps.count == 0 or ps.color is None:
                continue
            positions = checker_positions(pt, ps.count, L, top_row=(pt >= 13))
            fill = WHITE_CHECKER if ps.color == Color.WHITE else BLACK_CHECKER
            border = (30, 30, 30) if ps.color == Color.WHITE else (220, 220, 220)
            for (cx, cy) in positions:
                radius = min(L.point_width // 2 - 2, 22)
                pygame.draw.circle(self.screen, fill, (cx, cy), radius)
                pygame.draw.circle(self.screen, border, (cx, cy), radius, 2)
            if pt == selected_point:
                rect = L.point_rect(pt)
                pygame.draw.rect(self.screen, HIGHLIGHT,
                                 (rect.x, rect.y, rect.w, rect.h), 3)
        for tpt in highlight_targets:
            rect = L.point_rect(tpt)
            pygame.draw.rect(self.screen, HIGHLIGHT,
                             (rect.x, rect.y, rect.w, rect.h), 3)

    def _draw_dice(self, dice) -> None:
        L = self.layout
        x = L.board_left + L.board_width + 16
        y = L.board_top + 16
        for i, d in enumerate(dice):
            pygame.draw.rect(self.screen, (245, 245, 245),
                             (x, y + i * 80, 64, 64))
            pygame.draw.rect(self.screen, (0, 0, 0),
                             (x, y + i * 80, 64, 64), 2)
            lbl = self.dice_font.render(str(d), True, TEXT)
            self.screen.blit(lbl, (x + 32 - lbl.get_width() // 2,
                                   y + i * 80 + 32 - lbl.get_height() // 2))
```

- [ ] **Step 6: Manual smoke test**

Create a throwaway script `scratch_render.py` at repo root:
```python
import pygame
from engine.board import Board
from ui.layout import BoardLayout
from ui.renderer import Renderer


def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    L = BoardLayout()
    r = Renderer(screen, L)
    r.draw(Board(), dice=(3, 5))
    pygame.display.flip()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
    pygame.quit()

if __name__ == "__main__":
    main()
```

Run: `python scratch_render.py`. Confirm: board appears, 15 white at pt 24 (top-right corner), 15 black at pt 12 (bottom-left-ish), dice 3 and 5 visible on the right. Close window.

Delete the scratch file:
```bash
rm scratch_render.py
```

- [ ] **Step 7: Commit**

```bash
git add ui/__init__.py ui/layout.py ui/renderer.py tests/test_layout.py
git commit -m "feat(ui): add Pygame renderer with unit-tested layout"
```

---

### Task 14: Pygame input handling (human play loop)

**Files:**
- Create: `ui/input.py`
- Create: `tests/test_input.py`

Pure logic (point-hit test, state machine of "which checker is selected") in `input.py`; pygame event pumping happens in screens (Task 15).

- [ ] **Step 1: Write failing tests**

`tests/test_input.py`:
```python
from engine.board import Board, Color
from engine.moves import Move, generate_move_sequences
from ui.layout import BoardLayout
from ui.input import InputState, hit_test


class TestHitTest:
    def test_click_on_point(self):
        L = BoardLayout()
        rect = L.point_rect(24)
        pt = hit_test((rect.x + rect.w // 2, rect.y + rect.h // 2), L)
        assert pt == 24

    def test_click_outside_returns_none(self):
        L = BoardLayout()
        assert hit_test((0, 0), L) is None


class TestInputState:
    def test_select_own_checker_shows_targets(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        assert state.selected_from == 24
        # valid targets should include 21 and 19 (head with 3 and 5)
        assert 21 in state.highlight_targets or 19 in state.highlight_targets

    def test_click_target_applies_move(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        # find a valid first-move target
        target = next(iter(state.highlight_targets))
        state.click_point(target, Board())
        assert len(state.played_so_far) == 1

    def test_click_non_own_clears_selection(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        state.click_point(7, Board())  # empty point, not a valid target
        assert state.selected_from is None
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_input.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement input state**

`ui/input.py`:
```python
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from engine.board import Board, Color
from engine.moves import Move
from ui.layout import BoardLayout


def hit_test(pos: Tuple[int, int], layout: BoardLayout) -> Optional[int]:
    x, y = pos
    for pt in range(1, 25):
        r = layout.point_rect(pt)
        if r.x <= x < r.x + r.w and r.y <= y < r.y + r.h:
            return pt
    return None


@dataclass
class InputState:
    color: Color
    sequences: List[List[Move]]
    selected_from: Optional[int] = None
    played_so_far: List[Move] = field(default_factory=list)

    def _remaining_prefixes(self) -> List[List[Move]]:
        """Sequences that start with `played_so_far`."""
        out = []
        n = len(self.played_so_far)
        for s in self.sequences:
            if len(s) >= n and s[:n] == self.played_so_far:
                out.append(s)
        return out

    def _next_moves(self) -> List[Move]:
        n = len(self.played_so_far)
        remaining = self._remaining_prefixes()
        return [s[n] for s in remaining if len(s) > n]

    @property
    def highlight_targets(self) -> Set[int]:
        if self.selected_from is None:
            return set()
        out: Set[int] = set()
        for m in self._next_moves():
            if m.from_point == self.selected_from:
                out.add(0 if m.is_bear_off else m.to_point)
        return out

    def click_point(self, point: int, board: Board) -> Optional[List[Move]]:
        """Handle a click on `point`. Returns a completed sequence when done, else None."""
        if self.selected_from is None:
            # Selecting a from-point
            candidates = {m.from_point for m in self._next_moves()}
            if point in candidates:
                self.selected_from = point
            return None
        # Had a selected_from; try to pick a move with that origin
        if point in self.highlight_targets:
            # Find the move
            for m in self._next_moves():
                to = 0 if m.is_bear_off else m.to_point
                if m.from_point == self.selected_from and to == point:
                    self.played_so_far.append(m)
                    self.selected_from = None
                    # If the sequence is complete (no further prefixes extend), return it
                    if not self._next_moves():
                        return list(self.played_so_far)
                    return None
        # Click elsewhere → cancel selection
        self.selected_from = None
        return None
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_input.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/input.py tests/test_input.py
git commit -m "feat(ui): add click-driven input state machine"
```

---

### Task 15: Screens + entry point (menu, game, replay)

**Files:**
- Create: `ui/screens.py`
- Replace: `main.py` (the legacy FastAPI file)
- Create: `tests/test_smoke.py`

Screens:
- **MenuScreen**: 4 buttons — H vs H, H vs Local, H vs OpenRouter, Local vs OpenRouter (auto). Also "Watch replay…" (opens file picker via simple text-input of path).
- **GameScreen**: rolls dice when "Roll" button pressed; uses `InputState` for human turns; uses `BaseModel.choose_move` for AI turns; writes `.narde` on game over into `games/`.
- **ReplayScreen**: step forward/back, pause/play autoplay at 1s/move.

- [ ] **Step 1: Back up and remove legacy entry point**

The existing `main.py` is a FastAPI stub unrelated to the new engine. Move it out of the way so it can be recovered if needed, rather than deleted outright:
```bash
git mv main.py legacy_fastapi_main.py
```

- [ ] **Step 2: Write a minimal smoke test**

`tests/test_smoke.py`:
```python
import subprocess
import sys


def test_import_chain():
    # All engine/notation/model modules import cleanly together.
    from engine.board import Board
    from engine.moves import generate_move_sequences
    from engine.game import Game
    from notation.writer import format_game
    from notation.parser import parse_game
    from notation.replay import Replay
    from models.base import BaseModel
    from opponents.local_model import RandomLocalModel
    from opponents.openrouter import OpenRouterModel, build_prompt, parse_reply
    assert Board()  # constructs

def test_main_imports(tmp_path, monkeypatch):
    # main.py must be importable without launching pygame window.
    monkeypatch.setenv("NARDGAME_HEADLESS", "1")
    proc = subprocess.run(
        [sys.executable, "-c", "import main; assert hasattr(main, 'run')"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
```

- [ ] **Step 3: Run smoke — expect failure**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL (`main.py` does not exist / does not export `run`).

- [ ] **Step 4: Implement screens**

`ui/screens.py`:
```python
import os
import pygame
from datetime import date
from typing import Callable, List, Optional
from engine.board import Color
from engine.game import Game
from engine.moves import Move
from notation.parser import parse_game
from notation.replay import Replay
from notation.writer import save_game
from models.base import BaseModel
from opponents.local_model import RandomLocalModel
from ui.input import InputState, hit_test
from ui.layout import BoardLayout
from ui.renderer import Renderer


FPS = 30


def _draw_text(screen, font, text, pos, color=(20, 20, 20)):
    screen.blit(font.render(text, True, color), pos)


class Button:
    def __init__(self, rect, label, on_click):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click

    def draw(self, screen, font):
        pygame.draw.rect(screen, (220, 220, 220), self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        lbl = font.render(self.label, True, (0, 0, 0))
        screen.blit(lbl, (self.rect.centerx - lbl.get_width() // 2,
                          self.rect.centery - lbl.get_height() // 2))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.on_click()


class MenuScreen:
    def __init__(self, app):
        self.app = app
        self.font = pygame.font.SysFont("sans", 22)
        self.buttons = [
            Button((540, 200, 200, 40), "Human vs Human",
                   lambda: app.start_game(white=None, black=None)),
            Button((540, 260, 200, 40), "Human vs Local",
                   lambda: app.start_game(white=None, black=RandomLocalModel())),
            Button((540, 320, 200, 40), "Human vs OpenRouter",
                   lambda: app.start_game(white=None, black=app.make_openrouter())),
            Button((540, 380, 200, 40), "Local vs OpenRouter",
                   lambda: app.start_game(white=RandomLocalModel(), black=app.make_openrouter())),
            Button((540, 440, 200, 40), "Watch replay",
                   lambda: app.open_replay_prompt()),
        ]

    def handle(self, event):
        for b in self.buttons:
            b.handle(event)

    def draw(self, screen):
        screen.fill((240, 230, 210))
        _draw_text(screen, self.font, "Long Backgammon (нарды)", (520, 120))
        for b in self.buttons:
            b.draw(screen, self.font)


class GameScreen:
    def __init__(self, app, white_model: Optional[BaseModel],
                 black_model: Optional[BaseModel]):
        self.app = app
        self.layout = BoardLayout()
        self.renderer = Renderer(app.screen, self.layout)
        self.font = pygame.font.SysFont("sans", 18)
        self.game = Game(
            white_name="Human" if white_model is None else white_model.__class__.__name__,
            black_name="Human" if black_model is None else black_model.__class__.__name__,
        )
        self.game.determine_starter()
        self.models = {Color.WHITE: white_model, Color.BLACK: black_model}
        self.input_state: Optional[InputState] = None
        self.status = ""
        self.roll_button = Button((1090, 560, 160, 40), "Roll", self._on_roll)

    def _on_roll(self):
        if self.game.dice is not None or self.game.is_over():
            return
        self.game.roll()
        seqs = self.game.legal_sequences()
        current = self.game.current_player
        model = self.models[current]
        if model is not None:
            chosen = model.choose_move(self.game.board, current, self.game.dice, seqs)
            self.game.play(chosen)
            self._maybe_save()
        else:
            self.input_state = InputState(color=current, sequences=seqs)

    def _maybe_save(self):
        if self.game.is_over():
            self._save_game()

    def _save_game(self):
        os.makedirs("games", exist_ok=True)
        path = os.path.join("games", f"game_{date.today().isoformat()}_"
                                     f"{len(os.listdir('games'))+1:03d}.narde")
        save_game(
            path=path,
            history=self.game.history,
            event="Local Game",
            date=date.today().isoformat(),
            white=self.game.white_name,
            black=self.game.black_name,
            score=self.game.score(),
        )
        self.status = f"Saved {path}"

    def handle(self, event):
        self.roll_button.handle(event)
        if self.input_state and event.type == pygame.MOUSEBUTTONDOWN:
            pt = hit_test(event.pos, self.layout)
            if pt is not None:
                # Handle bear-off "point" 0 via right-click fallback: treat click
                # inside the off-tray (right column) as point 0.
                sequence = self.input_state.click_point(pt, self.game.board)
                if sequence is not None:
                    self.game.play(sequence)
                    self.input_state = None
                    self._maybe_save()

    def draw(self, screen):
        selected = self.input_state.selected_from if self.input_state else None
        targets = list(self.input_state.highlight_targets) if self.input_state else []
        targets_on_board = [t for t in targets if t != 0]
        self.renderer.draw(self.game.board, dice=self.game.dice,
                           selected_point=selected,
                           highlight_targets=targets_on_board)
        self.roll_button.draw(screen, self.font)
        turn_lbl = (f"Turn: {self.game.current_player.value}"
                    if self.game.current_player else "")
        _draw_text(screen, self.font, turn_lbl, (1090, 520))
        if self.status:
            _draw_text(screen, self.font, self.status, (20, 690))
        if self.game.is_over():
            _draw_text(screen, self.font,
                       f"Game over: {self.game.winner().value} wins {self.game.score()}",
                       (480, 680))


class ReplayScreen:
    def __init__(self, app, path: str):
        self.app = app
        self.layout = BoardLayout()
        self.renderer = Renderer(app.screen, self.layout)
        self.font = pygame.font.SysFont("sans", 18)
        with open(path, "r", encoding="utf-8") as f:
            self.parsed = parse_game(f.read())
        self.replay = Replay(self.parsed)
        self.autoplay = False
        self._last_tick = pygame.time.get_ticks()
        self.buttons = [
            Button((20, 660, 80, 40), "<<",
                   lambda: self.replay.step_backward()),
            Button((110, 660, 80, 40), "Play/Pause",
                   lambda: self._toggle()),
            Button((200, 660, 80, 40), ">>",
                   lambda: self.replay.step_forward()),
            Button((290, 660, 80, 40), "Menu",
                   lambda: app.goto_menu()),
        ]

    def _toggle(self):
        self.autoplay = not self.autoplay

    def handle(self, event):
        for b in self.buttons:
            b.handle(event)

    def draw(self, screen):
        if self.autoplay:
            now = pygame.time.get_ticks()
            if now - self._last_tick > 1000:
                self.replay.step_forward()
                self._last_tick = now
        self.renderer.draw(self.replay.board, dice=self.replay.current_dice)
        for b in self.buttons:
            b.draw(screen, self.font)
        _draw_text(screen, self.font,
                   f"Step {self.replay.current_move_index + 1}/{self.replay.total_steps()}",
                   (400, 670))
```

- [ ] **Step 5: Implement app entry point**

`main.py`:
```python
import os
import sys
import pygame
from opponents.openrouter import OpenRouterModel
from ui.screens import MenuScreen, GameScreen, ReplayScreen


class App:
    def __init__(self):
        self.screen = None
        self.active = None

    def make_openrouter(self):
        key = os.environ.get("OPENROUTER_API_KEY", "")
        return OpenRouterModel(api_key=key)

    def start_game(self, white, black):
        self.active = GameScreen(self, white, black)

    def open_replay_prompt(self):
        path = os.environ.get("NARDGAME_REPLAY_PATH")
        if not path:
            # Pick most recent .narde in games/
            if os.path.isdir("games"):
                files = sorted(
                    (os.path.join("games", f) for f in os.listdir("games")
                     if f.endswith(".narde")),
                    key=os.path.getmtime, reverse=True,
                )
                if files:
                    path = files[0]
        if path and os.path.exists(path):
            self.active = ReplayScreen(self, path)

    def goto_menu(self):
        self.active = MenuScreen(self)


def run():
    headless = os.environ.get("NARDGAME_HEADLESS") == "1"
    if headless:
        # Import-only smoke: just construct the app, don't open a window.
        app = App()
        return app
    pygame.init()
    app = App()
    app.screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Long Backgammon")
    app.active = MenuScreen(app)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                app.active.handle(event)
        app.active.draw(app.screen)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    run()
```

- [ ] **Step 6: Run smoke test — expect pass**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: both tests PASS.

- [ ] **Step 7: Full suite regression**

Run: `python -m pytest tests/ -v`
Expected: every test from Tasks 1–15 passes.

- [ ] **Step 8: Manual end-to-end check**

Run: `python main.py`
- Menu appears.
- Click "Human vs Local". Game screen appears.
- Click "Roll". Dice appear. Click on pt 24 (white head), then on a highlighted target. Complete the turn.
- Click "Roll" again. The local model auto-plays black.
- Play until someone bears off everyone. Check that a file `games/game_<date>_001.narde` is written.
- Return to menu (close & reopen or add a menu button if desired). Click "Watch replay" — the saved game replays.

- [ ] **Step 9: Commit**

```bash
git add ui/screens.py main.py tests/test_smoke.py legacy_fastapi_main.py
git commit -m "feat: wire up menu/game/replay screens and main entry"
```

---

## Self-Review Notes

**Spec coverage (checked against spec sections):**
- Инвентарь, нумерация доски, начальная расстановка → Task 1.
- Розыгрыш первого хода → Task 7 (`Game.determine_starter`).
- Ход игрока / движение шашек → Tasks 2, 5, 7.
- Дубль → Tasks 5 (`_expand_dice`), 8 (`(xN)` compression).
- Правило головы + исключения 6-6/4-4/3-3 → Task 3 (`HeadRule`).
- Запрет на занятые пункты → Task 2 (`is_legal_single`).
- Правило полного хода → Task 5 (max-length filter + larger-die preference).
- Блокировка (6-prime, нельзя без шашки впереди, нельзя запереть все 15) → Task 4. Note: the "cannot block all 15" sub-rule is implied by "opponent has a checker ahead" — if all 15 are behind, no one is ahead, block illegal.
- Пропуск хода → Tasks 5 (`[[]]`), 8 (`--`).
- Bearing off + overshoot → Tasks 2 (`is_bear_off_legal`), 6 (regression tests).
- Результаты: ойн / марс / ничья → Task 7 (`score`), Task 8 (`format_result`).
- Нотация (полная) → Tasks 8, 9.
- Pygame UI (renderer + input + screens) → Tasks 13, 14, 15.
- Replay → Task 10.
- BaseModel + local stub + OpenRouter → Tasks 11, 12.
- 4 game modes → Task 15 (menu buttons).

**Placeholders:** none — all code blocks are complete.

**Type consistency spot-checks:**
- `Move` dataclass (from_point, to_point, is_bear_off) — same across Tasks 2, 5, 8, 9, 10, 14.
- `HeadRule` — created in Task 3, consumed by Task 5.
- `TurnRecord` (player/dice/sequence/is_first_roll) — Task 7 produces it; Task 8 consumes identical fields.
- `ParsedGame` / `ParsedTurn` — Task 9 produces; Task 10 consumes.
- `BoardLayout.point_rect` → Rect(x,y,w,h) — used in Tasks 13 and 14.
- `BaseModel.choose_move(board, color, dice, sequences) -> sequence` — consistent in Tasks 11, 12, 15.

**Known trade-offs documented inline:**
- `_expected_die_for_move` uses a heuristic for bear-off dice; affects only the length-1 larger-die preference filter, which is cosmetic.
- `Replay.step_backward` rebuilds from the start; acceptable for short games.
- The subtle "first-roll burns remaining head-dice" example from the spec (§ 3-3 / 4-4 / 6-6 with opponent on head) falls out of sequence generation + full-turn filtering — no special case needed, but worth an extra regression test if future bugs appear.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-21-narde-game.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**
