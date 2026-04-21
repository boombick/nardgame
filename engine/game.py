import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from engine.board import Board, Color, opposite
from engine.moves import Move, apply_single, generate_move_sequences


@dataclass
class TurnRecord:
    """One turn played, stored for notation/replay."""
    player: Color
    dice: Tuple[int, int]
    sequence: List[Move]
    is_first_roll: bool


class Game:
    """Turn manager for a full long-backgammon match.

    Responsibilities: starter roll, per-turn dice, legal-sequence exposure,
    applying a chosen sequence, and reporting termination + scoring
    (ойн = 1 pt, марс = 2 pts)."""

    def __init__(self, white_name: str, black_name: str, rng=None):
        self.white_name = white_name
        self.black_name = black_name
        self.board = Board()
        self.current_player: Optional[Color] = None
        self.dice: Optional[Tuple[int, int]] = None
        self.history: List[TurnRecord] = []
        self._rng = rng if rng is not None else random.Random()
        self._has_played = {Color.WHITE: False, Color.BLACK: False}

    # Start ---------------------------------------------------------------

    def determine_starter(self) -> Color:
        """Roll one die per side until unequal; higher wins and starts."""
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
        """True if the current player has not yet played a turn in this game."""
        assert self.current_player is not None
        return not self._has_played[self.current_player]

    # Turn ----------------------------------------------------------------

    def roll(self) -> Tuple[int, int]:
        """Roll two dice for the current player and remember them."""
        d1 = self._rng.randint(1, 6)
        d2 = self._rng.randint(1, 6)
        self.dice = (d1, d2)
        return self.dice

    def legal_sequences(self) -> List[List[Move]]:
        """Return all maximum-length legal sequences for the current dice."""
        assert self.current_player is not None and self.dice is not None
        return generate_move_sequences(
            self.board, self.current_player, self.dice,
            is_first_roll=self.is_first_roll_for_current,
        )

    def play(self, sequence: List[Move]) -> None:
        """Apply `sequence` for the current player, record it, pass the turn."""
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
        """Return (white_points, black_points): 2 for марс, 1 for ойн, 0 otherwise."""
        w, b = self.board.borne_off[Color.WHITE], self.board.borne_off[Color.BLACK]
        if w == 15:
            return (2 if b == 0 else 1, 0)
        if b == 15:
            return (0, 2 if w == 0 else 1)
        return (0, 0)
