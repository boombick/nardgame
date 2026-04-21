from abc import ABC, abstractmethod
from typing import List, Tuple

from engine.board import Board, Color
from engine.moves import Move


class BaseModel(ABC):
    """Abstract opponent. Implementations choose one legal sequence per turn."""

    @abstractmethod
    def choose_move(self, board: Board, color: Color,
                    dice: Tuple[int, int],
                    sequences: List[List[Move]]) -> List[Move]:
        """Return exactly one element from `sequences`."""
        raise NotImplementedError
