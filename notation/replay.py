from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine.board import Board, Color
from engine.moves import Move, apply_single
from notation.parser import ParsedGame


@dataclass
class _FlatMove:
    """A single replay step. `move` is None for a skip-turn marker."""
    turn_index: int
    move_index_in_turn: int
    player: Color
    dice: Tuple[int, int]
    move: Optional[Move]


class Replay:
    """Walks a parsed game forward/backward over board snapshots, one half-move
    at a time. Backward stepping rebuilds from the start — cheap given a
    backgammon game's bounded length."""

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
