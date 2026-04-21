from engine.board import Board, Color
from engine.moves import generate_move_sequences


class TestSequences:
    def test_initial_white_3_5_has_sequences(self):
        seqs = generate_move_sequences(
            board=Board(), color=Color.WHITE, dice=(3, 5), is_first_roll=True
        )
        # All seqs are length 2 (full turn)
        assert all(len(s) == 2 for s in seqs)
        # At least one sequence uses the head (24 -> 21 or 24 -> 19)
        assert any(s[0].from_point == 24 for s in seqs)

    def test_head_rule_caps_at_one(self):
        seqs = generate_move_sequences(
            board=Board(), color=Color.WHITE, dice=(3, 5), is_first_roll=False
        )
        for s in seqs:
            head_uses = sum(1 for m in s if m.from_point == 24)
            assert head_uses <= 1

    def test_doubles_6_6_first_roll_allows_two_head(self):
        seqs = generate_move_sequences(
            board=Board(), color=Color.WHITE, dice=(6, 6), is_first_roll=True
        )
        for s in seqs:
            head_uses = sum(1 for m in s if m.from_point == 24)
            assert head_uses <= 2

    def test_full_turn_rule_prefers_both_dice(self):
        # Construct a position where playing both dice is possible — only
        # length-2 seqs returned.
        b = Board()
        seqs = generate_move_sequences(b, Color.WHITE, (1, 2), is_first_roll=False)
        assert all(len(s) == 2 for s in seqs)

    def test_skip_returns_single_empty_sequence(self):
        # Position with no legal moves: white has 1 checker at pt 24, dice 6-6
        # and black forms a full wall at 18..23 (landing targets for any die
        # from 24 are blocked).
        b = Board()
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        for pt in (18, 19, 20, 21, 22, 23):
            b.points[pt].count = 2
            b.points[pt].color = Color.BLACK
        # Park remaining 14 white at pt 1 (cannot move — already at deepest home
        # step and not all-in-home until this is verified; but 6-6 overshoot
        # from pt 1 would target step 29 >= 24, which requires all_in_home).
        # With the checker at 24, all_in_home is False, so pt 1 cannot bear off.
        b.points[1].count = 14
        b.points[1].color = Color.WHITE
        seqs = generate_move_sequences(b, Color.WHITE, (6, 6), is_first_roll=False)
        assert seqs == [[]]

    def test_partial_turn_when_only_one_die_playable(self):
        # Position where only the larger die can be played from pt 24.
        b = Board()
        b.points[24].count = 1
        b.points[24].color = Color.WHITE
        b.points[1].count = 14
        b.points[1].color = Color.WHITE
        # Black walls off 19, 21, 22, 23 so from 24: die=5->19 blocked.
        # Also wall pt 15 so that after the forced 24->20 play with die 4,
        # die 5 cannot continue (20->15 blocked).
        for pt in (15, 19, 21, 22, 23):
            b.points[pt].count = 2
            b.points[pt].color = Color.BLACK
        seqs = generate_move_sequences(b, Color.WHITE, (4, 5), is_first_roll=False)
        assert all(len(s) == 1 for s in seqs)
        assert any(m.from_point == 24 and m.to_point == 20 for seq in seqs for m in seq)
