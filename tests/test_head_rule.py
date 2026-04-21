from engine.board import Board, Color
from engine.moves import HeadRule


class TestHeadRule:
    def test_default_allows_one_from_head(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=False, dice=(3, 5))
        assert hr.head_allows(from_point=24, board=Board()) is True

    def test_default_blocks_second_from_head(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=False, dice=(3, 5))
        hr.register_head_use()
        assert hr.head_allows(from_point=24, board=Board()) is False

    def test_non_head_point_unaffected(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=False, dice=(3, 5))
        hr.register_head_use()
        assert hr.head_allows(from_point=20, board=Board()) is True

    def test_first_roll_double_6_allows_two_from_head(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=True, dice=(6, 6))
        assert hr.head_allows(24, Board()) is True
        hr.register_head_use()
        assert hr.head_allows(24, Board()) is True  # 2nd allowed
        hr.register_head_use()
        assert hr.head_allows(24, Board()) is False  # 3rd not allowed

    def test_first_roll_double_4_allows_two_from_head(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=True, dice=(4, 4))
        for _ in range(2):
            assert hr.head_allows(24, Board()) is True
            hr.register_head_use()
        assert hr.head_allows(24, Board()) is False

    def test_first_roll_double_3_allows_two_from_head(self):
        hr = HeadRule(color=Color.BLACK, is_first_roll=True, dice=(3, 3))
        for _ in range(2):
            assert hr.head_allows(12, Board()) is True
            hr.register_head_use()
        assert hr.head_allows(12, Board()) is False

    def test_first_roll_other_double_still_one(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=True, dice=(5, 5))
        assert hr.head_allows(24, Board()) is True
        hr.register_head_use()
        assert hr.head_allows(24, Board()) is False

    def test_clone_preserves_state(self):
        hr = HeadRule(color=Color.WHITE, is_first_roll=False, dice=(3, 5))
        hr.register_head_use()
        clone = hr.clone()
        assert clone.head_allows(24, Board()) is False
        # Mutating clone doesn't affect original
        # (Here we'd register on clone, but it's already at max — use a fresh pair)
        hr2 = HeadRule(color=Color.WHITE, is_first_roll=False, dice=(3, 5))
        c2 = hr2.clone()
        c2.register_head_use()
        assert hr2.head_allows(24, Board()) is True  # original unaffected
        assert c2.head_allows(24, Board()) is False
