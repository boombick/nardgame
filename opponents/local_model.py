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

    def choose_move(self, board: Board, color: Color,
                    dice: Tuple[int, int],
                    sequences: List[List[Move]]) -> List[Move]:
        return self._rng.choice(sequences)
