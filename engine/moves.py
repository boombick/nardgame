from dataclasses import dataclass, field
from typing import List, Tuple
from engine.board import Board, Color, point_to_step, step_to_point, opposite


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


def _consecutive_run_through(board: Board, color: Color, pt: int) -> list:
    """Return the sorted list of consecutive points (absolute 1..24, no wrap)
    owned by `color` that includes `pt`. Returns [] if `pt` is not owned by color."""
    if board.count_at(pt, color) == 0:
        return []
    run = [pt]
    p = pt - 1
    while p >= 1 and board.count_at(p, color) > 0:
        run.append(p)
        p -= 1
    p = pt + 1
    while p <= 24 and board.count_at(p, color) > 0:
        run.append(p)
        p += 1
    return sorted(run)


def _opponent_has_checker_ahead(board: Board, color: Color, run_points: list) -> bool:
    """True if the opponent of `color` has any non-head checker ahead of the block —
    i.e. past it in the block owner's direction of travel. A checker still sitting
    on the opponent's head point has not yet entered play and does not count."""
    opp = opposite(color)
    block_max_owner_step = max(point_to_step(pt, color) for pt in run_points)
    for p in range(1, 25):
        if board.count_at(p, opp) > 0 and not board.is_head(p, opp):
            if point_to_step(p, color) > block_max_owner_step:
                return True
    return False


def forms_illegal_block(board: Board, color: Color, move: Move) -> bool:
    """Return True if applying `move` would create an illegal 6-prime (six
    consecutive own points with no opponent checker ahead). Bear-off moves
    can never form a block."""
    if move.is_bear_off:
        return False
    sim = board.clone()
    sim.remove_one(move.from_point, color)
    sim.place_one(move.to_point, color)
    run = _consecutive_run_through(sim, color, move.to_point)
    if len(run) < 6:
        return False
    return not _opponent_has_checker_ahead(sim, color, run)


def _expand_dice(dice: Tuple[int, int]) -> List[int]:
    """4 moves on doubles, 2 otherwise."""
    return [dice[0]] * 4 if dice[0] == dice[1] else [dice[0], dice[1]]


def _legal_one_die_moves(board: Board, color: Color, die: int,
                         head_rule: "HeadRule") -> List[Move]:
    """All single-die moves legal on `board` for `color` with this `die`,
    respecting head rule and six-block rule."""
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
    """DFS over orderings of `remaining_dice`, collecting every maximal
    prefix path. Leaf = no dice left. Dead-end = no legal move for any
    remaining die (the partial path is still recorded)."""
    if not remaining_dice:
        collected.append(list(path))
        return

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


def _expected_die_for_move(move: Move, color: Color) -> int:
    """Back-compute which die was used for a single move. For bear-off we
    return the pip count that reached/overshot step 24 (sufficient to
    distinguish which die was used when the two dice differ)."""
    from_step = point_to_step(move.from_point, color)
    if move.is_bear_off:
        return 24 - from_step
    to_step = point_to_step(move.to_point, color)
    return to_step - from_step


def generate_move_sequences(board: Board, color: Color,
                            dice: Tuple[int, int],
                            is_first_roll: bool) -> List[List[Move]]:
    """Return all legal maximum-length move sequences for this turn.

    Enforces:
      - per-die legality (is_legal_single)
      - head rule (HeadRule, with first-roll double exceptions)
      - six-block rule (forms_illegal_block)
      - full-turn rule: only sequences of maximum achievable length are kept
      - larger-die preference: if only one die can be played and the dice
        differ, sequences using the larger die are preferred when available.

    If no legal move exists, returns `[[]]` (a single empty sequence)."""
    remaining = _expand_dice(dice)
    head_rule = HeadRule(color=color, is_first_roll=is_first_roll, dice=dice)
    collected: List[List[Move]] = []
    _explore(board, color, remaining, head_rule, [], collected)

    unique = []
    seen = set()
    for seq in collected:
        key = tuple((m.from_point, m.to_point, m.is_bear_off) for m in seq)
        if key not in seen:
            seen.add(key)
            unique.append(seq)

    if not unique:
        return [[]]

    max_len = max(len(s) for s in unique)
    filtered = [s for s in unique if len(s) == max_len]

    if max_len == 1 and dice[0] != dice[1]:
        larger = max(dice)
        has_larger = any(
            _expected_die_for_move(s[0], color) == larger for s in filtered
        )
        if has_larger:
            filtered = [s for s in filtered
                        if _expected_die_for_move(s[0], color) == larger]

    if not filtered:
        return [[]]
    return filtered
