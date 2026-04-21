import random

from engine.board import Board, Color
from engine.moves import Move
from opponents.openrouter import OpenRouterModel, build_prompt, parse_reply


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
