from engine.board import Board, Color
from engine.moves import generate_move_sequences
from ui.input import InputState, hit_test
from ui.layout import BoardLayout


class TestHitTest:
    def test_click_on_point(self):
        L = BoardLayout()
        rect = L.point_rect(24)
        pt = hit_test((rect.x + rect.w // 2, rect.y + rect.h // 2), L)
        assert pt == 24

    def test_click_outside_returns_none(self):
        L = BoardLayout()
        assert hit_test((0, 0), L) is None


class TestInputState:
    def test_select_own_checker_shows_targets(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        assert state.selected_from == 24
        assert 21 in state.highlight_targets or 19 in state.highlight_targets

    def test_click_target_applies_move(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        # Chained destinations (compound moves with both dice) may apply more
        # than one move at once; check at least one was played.
        target = next(iter(state.highlight_targets))
        state.click_point(target, Board())
        assert len(state.played_so_far) >= 1

    def test_compound_target_highlighted(self):
        # Dice (3,5): from head (pt 24), one checker plays both dice -> 24-8=16.
        # Expect intermediate (21 or 19) AND compound destination 16 to show.
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        assert 16 in state.highlight_targets
        assert 21 in state.highlight_targets or 19 in state.highlight_targets

    def test_click_compound_target_applies_chain(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        state.click_point(16, Board())
        # Full sequence resolved by a single compound click.
        assert len(state.played_so_far) == 2
        assert state.played_so_far[0].from_point == 24
        assert state.played_so_far[-1].to_point == 16

    def test_click_non_own_clears_selection(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        state.click_point(7, Board())  # empty, not a valid target
        assert state.selected_from is None


class TestReset:
    def test_reset_clears_played_and_selection(self):
        # After a partial click the state holds a move in played_so_far.
        # reset() should put us back to "nothing chosen yet" for the same
        # sequence list, so the user can reconsider.
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        state.click_point(21, Board())
        assert state.played_so_far
        state.reset()
        assert state.played_so_far == []
        assert state.selected_from is None
        # After reset the full set of legal starting points is back.
        assert state.legal_from_points == {24}

    def test_reset_on_empty_state_is_noop(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.reset()
        assert state.played_so_far == []
        assert state.selected_from is None
