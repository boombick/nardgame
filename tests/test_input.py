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
        target = next(iter(state.highlight_targets))
        state.click_point(target, Board())
        assert len(state.played_so_far) == 1

    def test_click_non_own_clears_selection(self):
        seqs = generate_move_sequences(Board(), Color.WHITE, (3, 5),
                                       is_first_roll=True)
        state = InputState(color=Color.WHITE, sequences=seqs)
        state.click_point(24, Board())
        state.click_point(7, Board())  # empty, not a valid target
        assert state.selected_from is None
