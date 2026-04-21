from typing import List, Tuple

from engine.board import Color
from engine.game import TurnRecord
from engine.moves import Move


def format_move(move: Move) -> str:
    """Render a single move as `from/to` or `from/off` for bear-offs."""
    if move.is_bear_off:
        return f"{move.from_point}/off"
    return f"{move.from_point}/{move.to_point}"


def _compress_sequence(moves: List[Move]) -> List[str]:
    """Collapse runs of identical consecutive moves into `from/to(xN)` tokens.
    Primarily used for doubles where the same move repeats."""
    if not moves:
        return []
    out = []
    i = 0
    while i < len(moves):
        j = i
        while (j + 1 < len(moves)
               and moves[j + 1].from_point == moves[i].from_point
               and moves[j + 1].to_point == moves[i].to_point
               and moves[j + 1].is_bear_off == moves[i].is_bear_off):
            j += 1
        run = j - i + 1
        token = format_move(moves[i])
        if run > 1:
            token = f"{token}(x{run})"
        out.append(token)
        i = j + 1
    return out


def format_turn(number: int, rec: TurnRecord) -> str:
    """Render one turn as `N. <W|B> d1-d2: moves` (or `--` when skipped)."""
    color_tag = "W" if rec.player == Color.WHITE else "B"
    dice = f"{rec.dice[0]}-{rec.dice[1]}"
    body = "--" if not rec.sequence else " ".join(_compress_sequence(rec.sequence))
    return f"{number}. {color_tag} {dice}: {body}"


def format_result(score: Tuple[int, int], draw: bool = False) -> str:
    """Render the final result line. Марс = 2 pts, Ойн = 1 pt, draw = 1/2-1/2."""
    if draw:
        return "Result: 1/2-1/2"
    w, b = score
    if w == 2 or b == 2:
        return f"Result: {w}-{b} (Марс)"
    if w == 1 or b == 1:
        return f"Result: {w}-{b} (Ойн)"
    return f"Result: {w}-{b}"


def format_game(history: List[TurnRecord], event: str, date: str,
                white: str, black: str, score: Tuple[int, int],
                draw: bool = False) -> str:
    """Render a full game as the .narde text format."""
    lines = [
        f'[Event "{event}"]',
        f'[Date "{date}"]',
        f'[White "{white}"]',
        f'[Black "{black}"]',
        "",
    ]
    for i, rec in enumerate(history, start=1):
        lines.append(format_turn(i, rec))
    lines.append(format_result(score, draw=draw))
    return "\n".join(lines) + "\n"


def save_game(path: str, **kwargs) -> None:
    """Write a formatted game to `path` as UTF-8 text."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_game(**kwargs))
