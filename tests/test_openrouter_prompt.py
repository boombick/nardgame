import random

from engine.board import Board, Color
from engine.moves import Move
from opponents.openrouter import (
    OpenRouterModel, build_prompt, extract_evaluation, extract_reason,
    parse_reply,
)


class TestBuildPrompt:
    def test_prompt_contains_dice_and_sequences(self):
        seqs = [[Move(24, 21, False), Move(24, 19, False)],
                [Move(24, 19, False), Move(24, 21, False)]]
        p = build_prompt(Board(), Color.WHITE, (3, 5), seqs)
        assert "3-5" in p
        assert "24/21" in p and "24/19" in p
        assert "WHITE" in p or "white" in p.lower()

    def test_prompt_labels_sequences(self):
        seqs = [[Move(24, 21, False)]]
        p = build_prompt(Board(), Color.WHITE, (3, 5), seqs)
        assert "1)" in p or "1." in p

    def test_prompt_includes_rules(self):
        # The prompt must carry enough rules so the model does not need any
        # outside context: head rule, six-block rule, bear-off rule.
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        assert "голов" in p.lower()       # голова / головы
        assert "блок" in p.lower() or "заграждени" in p.lower()
        assert "выбрас" in p.lower() or "bearing" in p.lower()

    def test_prompt_includes_board_layout(self):
        # The LLM must see the 2D geometry (not just the path) so it can
        # reason about "home", "head", and quadrants from the ptN listing.
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        assert "разметка доски" in low
        # Numeric path for both sides, written in the canonical "A → B → …"
        # shape so the model can see direction at a glance.
        assert "24 → 19" in p and "→ 1" in p        # white path
        assert "12 → 7" in p and "→ 13" in p        # black path
        # Layout rows: top row has 13..18 and 19..24, bottom has 7..12 and 1..6.
        assert "13 14 15 16 17 18" in p
        assert "19 20 21 22 23 24" in p
        assert "12 11 10  9  8  7" in p
        assert " 6  5  4  3  2  1" in p

    def test_prompt_does_not_duplicate_path_in_rules(self):
        # The path is owned by the board-layout section; RULES must not
        # carry its own "Путь: 24→23→…" line anymore.
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        assert "путь:" not in p.lower()
        assert "24→23" not in p and "12→11" not in p

    def test_prompt_includes_notation_guide(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        # Describes the ptN/from/to/off notation used inline.
        assert "pt" in p.lower()
        assert "off" in p.lower()
        assert "from/to" in p.lower() or "from/off" in p.lower()

    def test_prompt_requests_three_sections(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        assert "оценка" in low
        assert "ход" in low
        assert "объяснени" in low


class TestParseReply:
    def test_parse_index(self):
        assert parse_reply("1", num_sequences=3) == 0
        assert parse_reply("  2  ", num_sequences=3) == 1
        assert parse_reply("3", num_sequences=3) == 2

    def test_parse_with_prefix(self):
        assert parse_reply("Choice: 2", num_sequences=3) == 1

    def test_rejects_out_of_range(self):
        assert parse_reply("99", num_sequences=3) is None

    def test_rejects_garbage(self):
        assert parse_reply("I think option alpha", num_sequences=3) is None

    def test_parses_two_line_reply(self):
        assert parse_reply("2\nBecause it blocks the head", 3) == 1

    def test_reason_on_later_line_does_not_hijack_choice(self):
        # If the reasoning mentions other numbers, the choice on line 1 wins.
        assert parse_reply("2\nI also considered 1 and 3", 3) == 1

    def test_parses_labeled_three_line_reply(self):
        reply = ("Оценка: белые держат голову\n"
                 "Ход: 2\n"
                 "Объяснение: блокирую выход чёрных")
        assert parse_reply(reply, 3) == 1

    def test_labeled_move_line_wins_over_numbers_in_evaluation(self):
        # "Оценка" has a '1' inside; the labelled "Ход: 3" must still win.
        reply = ("Оценка: белые на 1 темп впереди\n"
                 "Ход: 3\n"
                 "Объяснение: крепкий дом")
        assert parse_reply(reply, 4) == 2


class TestExtractReason:
    def test_extracts_line_after_first(self):
        assert extract_reason("2\nBlocks the head") == "Blocks the head"

    def test_joins_multiple_reason_lines(self):
        assert extract_reason("1\nReason one\nReason two") == \
            "Reason one Reason two"

    def test_empty_when_single_line(self):
        assert extract_reason("2") == ""

    def test_labeled_reason_wins_over_fallback(self):
        reply = ("Оценка: равная позиция\n"
                 "Ход: 1\n"
                 "Объяснение: держу голову")
        assert extract_reason(reply) == "держу голову"


class TestExtractEvaluation:
    def test_extracts_labeled_evaluation(self):
        reply = ("Оценка: чёрные заперли голову\n"
                 "Ход: 2\n"
                 "Объяснение: разбегаюсь")
        assert extract_evaluation(reply) == "чёрные заперли голову"

    def test_empty_when_no_label(self):
        assert extract_evaluation("2\nBlocks the head") == ""

    def test_empty_on_none(self):
        assert extract_evaluation(None) == ""


class FakeHttp:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return self.replies.pop(0)


class TestOpenRouterModel:
    def test_picks_valid_reply_first_try(self):
        http = FakeHttp(["2"])
        m = OpenRouterModel(api_key="x", http=http)
        seqs = [[Move(24, 21, False)], [Move(24, 19, False)]]
        out = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert out == seqs[1]
        assert http.calls == 1

    def test_retries_on_invalid(self):
        http = FakeHttp(["nope", "also nope", "1"])
        m = OpenRouterModel(api_key="x", http=http)
        seqs = [[Move(24, 21, False)], [Move(24, 19, False)]]
        out = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert out == seqs[0]
        assert http.calls == 3

    def test_falls_back_after_3_failures(self):
        http = FakeHttp(["?", "??", "???"])
        m = OpenRouterModel(api_key="x", http=http, rng=random.Random(0))
        seqs = [[Move(24, 21, False)], [Move(24, 19, False)]]
        out = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert out in seqs
        assert http.calls == 3
        assert m.last_evaluation == ""

    def test_populates_evaluation_and_reason_from_labeled_reply(self):
        reply = ("Оценка: позиция ровная\n"
                 "Ход: 1\n"
                 "Объяснение: прикрываю голову")
        http = FakeHttp([reply])
        m = OpenRouterModel(api_key="x", http=http)
        seqs = [[Move(24, 21, False)], [Move(24, 19, False)]]
        out = m.choose_move(Board(), Color.WHITE, (3, 5), seqs)
        assert out == seqs[0]
        assert m.last_evaluation == "позиция ровная"
        assert m.last_reason == "прикрываю голову"
