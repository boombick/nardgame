import random

from engine.board import Board, Color
from engine.moves import generate_move_sequences
from opponents.local_model import RandomLocalModel


class TestRandomLocalModel:
    def test_returns_one_of_valid_sequences(self):
        m = RandomLocalModel(rng=random.Random(0))
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        chosen = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert chosen in seqs

    def test_skip_when_only_empty_sequence(self):
        m = RandomLocalModel(rng=random.Random(0))
        chosen = m.choose_move(Board(), Color.WHITE, (3, 5), [[]])
        assert chosen == []
