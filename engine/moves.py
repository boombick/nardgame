from dataclasses import dataclass, field
from typing import Tuple
from engine.board import Board, Color, point_to_step, step_to_point


@dataclass(frozen=True)
class Move:
    """A single-die move. `to_point` is 0 if and only if `is_bear_off` is True.
    Construct via `make_single_move` to ensure this invariant holds."""
    from_point: int          # 1..24
    to_point: int            # 1..24 for regular moves, 0 for bear-off
    is_bear_off: bool = False


def destination_step(from_point: int, die: int, color: Color) -> int:
    """Target player-step after moving `die` pips from `from_point`. May exceed 23."""
    return point_to_step(from_point, color) + die


def is_bear_off_legal(board: Board, color: Color, from_point: int, die: int) -> bool:
    """True if bearing off from `from_point` with `die` is legal.
    Pre: a checker of `color` sits on from_point.
    Requires: all checkers in home, and either exact die (lands on step 24) or
    overshoot (target > 24) with no own checker at a smaller step (farther from
    the exit) within the home zone."""
    if not board.all_in_home(color):
        return False
    src_step = point_to_step(from_point, color)
    if src_step < 18:
        return False
    target_step = src_step + die
    if target_step == 24:
        return True
    if target_step > 24:
        for s in range(18, src_step):
            pt = step_to_point(s, color)
            if board.count_at(pt, color) > 0:
                return False
        return True
    return False  # undershoot is not a bear-off


def is_legal_single(board: Board, color: Color, from_point: int, die: int) -> bool:
    """Check one-die legality ignoring head-rule and 6-block rule.
    Head/block rules are layered in later tasks via sequence generation."""
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
    if move.is_bear_off:
        board.bear_off_one(move.from_point, color)
    else:
        board.remove_one(move.from_point, color)
        board.place_one(move.to_point, color)


def make_single_move(board: Board, color: Color, from_point: int, die: int) -> Move:
    """Build the canonical Move for a legal single-die move.
    Does not validate — caller is responsible."""
    target_step = destination_step(from_point, die, color)
    if target_step >= 24:
        return Move(from_point=from_point, to_point=0, is_bear_off=True)
    return Move(from_point=from_point,
                to_point=step_to_point(target_step, color),
                is_bear_off=False)


_FIRST_ROLL_DOUBLE_EXCEPTIONS = {(6, 6), (4, 4), (3, 3)}


@dataclass
class HeadRule:
    """Tracks head-use count per turn; exposes whether further head-moves are allowed.
    On the first roll of the game, doubles 6-6 / 4-4 / 3-3 allow 2 head-uses instead of 1."""
    color: Color
    is_first_roll: bool
    dice: Tuple[int, int]
    _head_uses: int = field(default=0, init=False)

    @property
    def max_head_uses(self) -> int:
        """Maximum number of checkers that may leave the head this turn."""
        if self.is_first_roll and self.dice in _FIRST_ROLL_DOUBLE_EXCEPTIONS:
            return 2
        return 1

    def head_allows(self, from_point: int, board: Board) -> bool:
        """Return True if a move leaving `from_point` is allowed by the head rule.
        Non-head origins are always allowed."""
        if not board.is_head(from_point, self.color):
            return True
        return self._head_uses < self.max_head_uses

    def register_head_use(self) -> None:
        """Record that one checker left the head this turn."""
        self._head_uses += 1

    def clone(self) -> "HeadRule":
        """Return an independent copy that shares no state with the original."""
        hr = HeadRule(self.color, self.is_first_roll, self.dice)
        hr._head_uses = self._head_uses
        return hr
