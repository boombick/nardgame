import random
import re
from typing import Callable, List, Optional, Tuple

from engine.board import Board, Color
from engine.moves import Move
from models.base import BaseModel
from notation.writer import format_move
from opponents.local_model import RandomLocalModel


def _describe_board(board: Board, color: Color) -> str:
    """Compact text description of the board from `color`'s perspective."""
    rows = []
    for pt in range(1, 25):
        ps = board.points[pt]
        if ps.count > 0 and ps.color is not None:
            rows.append(f"  pt{pt}: {ps.count} {ps.color.value}")
    borne = board.borne_off
    return (f"On-turn: {color.value.upper()}\n"
            f"Board:\n" + "\n".join(rows) +
            f"\nBorne off: white={borne[Color.WHITE]}, black={borne[Color.BLACK]}")


def _describe_sequence(seq: List[Move]) -> str:
    if not seq:
        return "(skip)"
    return " ".join(format_move(m) for m in seq)


def build_prompt(board: Board, color: Color,
                 dice: Tuple[int, int],
                 sequences: List[List[Move]]) -> str:
    """Render the chat prompt sent to the OpenRouter model. Pure; test-friendly."""
    lines = [
        "You are playing long backgammon (длинные нарды).",
        "Both players move counter-clockwise. No hitting; destination blocked "
        "if opponent occupies it.",
        f"Dice: {dice[0]}-{dice[1]}",
        _describe_board(board, color),
        "",
        "Choose exactly one sequence by replying with its number (1-based) "
        "and nothing else.",
        "Candidate sequences:",
    ]
    for i, seq in enumerate(sequences, start=1):
        lines.append(f"  {i}) {_describe_sequence(seq)}")
    return "\n".join(lines)


_INT_RE = re.compile(r'(\d+)')


def parse_reply(text: Optional[str], num_sequences: int) -> Optional[int]:
    """Extract a 1-based integer choice from `text` and return it as a 0-based
    index. Returns None when the reply has no integer, multiple digits that
    encode an out-of-range choice, or is otherwise unparsable."""
    if text is None:
        return None
    m = _INT_RE.search(text.strip())
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < num_sequences:
        return idx
    return None


class OpenRouterModel(BaseModel):
    """BaseModel implementation backed by the OpenRouter chat API.

    `http` is a callable prompt->reply_text; injectable for tests. On three
    consecutive unparsable replies the model delegates to a RandomLocalModel."""

    def __init__(self, api_key: str,
                 model: str = "google/gemini-flash-1.5",
                 http: Optional[Callable[[str], str]] = None,
                 rng: Optional[random.Random] = None,
                 max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self._http = http if http is not None else self._default_http
        self._fallback = RandomLocalModel(rng=rng)
        self._max_retries = max_retries

    def _default_http(self, prompt: str) -> str:
        import requests
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def choose_move(self, board: Board, color: Color,
                    dice: Tuple[int, int],
                    sequences: List[List[Move]]) -> List[Move]:
        if len(sequences) == 1:
            return sequences[0]
        prompt = build_prompt(board, color, dice, sequences)
        for _ in range(self._max_retries):
            try:
                reply = self._http(prompt)
            except Exception:
                reply = None
            idx = parse_reply(reply, len(sequences))
            if idx is not None:
                return sequences[idx]
        return self._fallback.choose_move(board, color, dice, sequences)
