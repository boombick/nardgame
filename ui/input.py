from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from engine.board import Board, Color
from engine.moves import Move
from ui.layout import BoardLayout


def hit_test(pos: Tuple[int, int], layout: BoardLayout) -> Optional[int]:
    """Return the absolute point (1..24) whose rect contains `pos`, or None."""
    x, y = pos
    for pt in range(1, 25):
        r = layout.point_rect(pt)
        if r.x <= x < r.x + r.w and r.y <= y < r.y + r.h:
            return pt
    return None


@dataclass
class InputState:
    """Click-driven state machine that filters the legal-sequence list down
    to one concrete sequence. The caller feeds click_point() each click; when
    a full sequence is resolved, click_point() returns it.

    Clicking a chained destination (e.g. `src - d1 - d2` reachable by the same
    checker) applies the whole chain at once so the user can pick any
    reachable end point, not only the immediate next stop."""

    color: Color
    sequences: List[List[Move]]
    selected_from: Optional[int] = None
    played_so_far: List[Move] = field(default_factory=list)

    def _remaining_prefixes(self) -> List[List[Move]]:
        out = []
        n = len(self.played_so_far)
        for s in self.sequences:
            if len(s) >= n and s[:n] == self.played_so_far:
                out.append(s)
        return out

    def _next_moves(self) -> List[Move]:
        n = len(self.played_so_far)
        return [s[n] for s in self._remaining_prefixes() if len(s) > n]

    @property
    def legal_from_points(self) -> Set[int]:
        """Points whose checkers can legally start the next move."""
        return {m.from_point for m in self._next_moves()}

    @property
    def highlight_targets(self) -> Set[int]:
        """Every destination reachable from `selected_from` by chaining
        consecutive moves of the same checker within some remaining legal
        sequence. Includes both single-die destinations and multi-die
        compound destinations. Bear-off is reported as 0."""
        if self.selected_from is None:
            return set()
        out: Set[int] = set()
        n = len(self.played_so_far)
        for seq in self._remaining_prefixes():
            pos = self.selected_from
            for m in seq[n:]:
                if m.from_point != pos:
                    break
                dst = 0 if m.is_bear_off else m.to_point
                out.add(dst)
                if m.is_bear_off:
                    break
                pos = dst
        return out

    def reset(self) -> None:
        """Discard any partially-played moves and the current selection.

        The user might half-commit a move and then want to reconsider —
        the GameScreen exposes this via the "Отменить ход" button. The
        InputState itself just forgets what it had; snapshot restore of
        the display board is GameScreen's job."""
        self.played_so_far = []
        self.selected_from = None

    def click_point(self, point: int, board: Board) -> Optional[List[Move]]:
        """Handle a click on `point`. Returns the fully-chosen sequence when
        the last move has been resolved; otherwise None."""
        if self.selected_from is None:
            if point in self.legal_from_points:
                self.selected_from = point
            return None
        chain = self._find_chain(point)
        if chain:
            self.played_so_far.extend(chain)
            self.selected_from = None
            if not self._next_moves():
                return list(self.played_so_far)
            return None
        # Click missed — clear selection so user can retry.
        self.selected_from = None
        return None

    def _find_chain(self, target: int) -> Optional[List[Move]]:
        """Find the shortest chain of moves starting from `selected_from`
        that reaches `target`, where the chain is a prefix of some remaining
        legal sequence."""
        n = len(self.played_so_far)
        best: Optional[List[Move]] = None
        for seq in self._remaining_prefixes():
            pos = self.selected_from
            chain: List[Move] = []
            for m in seq[n:]:
                if m.from_point != pos:
                    break
                chain.append(m)
                dst = 0 if m.is_bear_off else m.to_point
                if dst == target:
                    if best is None or len(chain) < len(best):
                        best = list(chain)
                    break
                if m.is_bear_off:
                    break
                pos = dst
        return best
