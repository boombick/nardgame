import random
from typing import List, Tuple

from engine.board import Board, Color
from engine.moves import Move
from models.base import BaseModel


class RandomLocalModel(BaseModel):
    """Picks a uniformly random legal sequence. Useful as a sparring partner
    and as a deterministic baseline in tests when given a seeded rng."""

    def __init__(self, rng=None):
        self._rng = rng if rng is not None else random.Random()
        # Populated on every choose_move so the UI has a uniform
        # way to surface model commentary — including `last_evaluation`
        # so that screens.py can render both fields without type-checking
        # for the model class.
        self.last_reason: str = ""
        self.last_evaluation: str = ""

    def choose_move(self, board: Board, color: Color,
                    dice: Tuple[int, int],
                    sequences: List[List[Move]]) -> List[Move]:
        self.last_reason = "случайный выбор"
        self.last_evaluation = ""
        return self._rng.choice(sequences)
