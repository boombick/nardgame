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
    a full sequence is resolved, click_point() returns it."""

    color: Color
    sequences: List[List[Move]]
    selected_from: Optional[int] = None
    played_so_far: List[Move] = field(default_factory=list)

    def _remaining_prefixes(self) -> List[List[Move]]:
        """Sequences whose first `len(played_so_far)` moves equal played_so_far."""
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
    def highlight_targets(self) -> Set[int]:
        """Set of destination points (0 for bear-off) reachable from the
        currently selected from-point."""
        if self.selected_from is None:
            return set()
        out: Set[int] = set()
        for m in self._next_moves():
            if m.from_point == self.selected_from:
                out.add(0 if m.is_bear_off else m.to_point)
        return out

    def click_point(self, point: int, board: Board) -> Optional[List[Move]]:
        """Handle a click on `point`. Returns the fully-chosen sequence when
        the last move in it has been resolved; otherwise None."""
        if self.selected_from is None:
            candidates = {m.from_point for m in self._next_moves()}
            if point in candidates:
                self.selected_from = point
            return None
        if point in self.highlight_targets:
            for m in self._next_moves():
                to = 0 if m.is_bear_off else m.to_point
                if m.from_point == self.selected_from and to == point:
                    self.played_so_far.append(m)
                    self.selected_from = None
                    if not self._next_moves():
                        return list(self.played_so_far)
                    return None
        self.selected_from = None
        return None
