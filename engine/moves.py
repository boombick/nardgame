from dataclasses import dataclass
from engine.board import Board, Color, point_to_step, step_to_point


@dataclass(frozen=True)
class Move:
    """A single-die move. `to_point` is 0 when `is_bear_off` is True."""
    from_point: int          # 1..24
    to_point: int            # 1..24, or 0 for bear-off
    is_bear_off: bool = False


def destination_step(from_point: int, die: int, color: Color) -> int:
    """Target player-step after moving `die` pips from `from_point`. May exceed 23."""
    return point_to_step(from_point, color) + die


def is_bear_off_legal(board: Board, color: Color, from_point: int, die: int) -> bool:
    """True if bearing off from `from_point` with `die` is legal.
    Pre: a checker of `color` sits on from_point.
    Requires: all 15 in home, and either exact die or overshoot with no higher-priority
    (lower-step) home point occupied by `color`."""
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
    board.remove_one(move.from_point, color)
    if move.is_bear_off:
        board.borne_off[color] += 1
    else:
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
