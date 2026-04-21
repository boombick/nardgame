import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.board import Color
from engine.moves import Move


_HEADER_RE = re.compile(r'^\[(\w+)\s+"([^"]*)"\]$')
_TURN_RE = re.compile(r'^\s*(\d+)\.\s+([WB])\s+(\d)-(\d):\s+(.+?)\s*$')
_MOVE_TOKEN_RE = re.compile(r'^(\d+)/(off|\d+)(?:\(x(\d+)\))?$')
_RESULT_RE = re.compile(r'^Result:\s+(\S+)(?:\s+\(([^)]+)\))?\s*$')


@dataclass
class ParsedTurn:
    number: int
    player: Color
    dice: Tuple[int, int]
    moves: List[Move]
    is_skip: bool


@dataclass
class ParsedGame:
    headers: Dict[str, str] = field(default_factory=dict)
    turns: List[ParsedTurn] = field(default_factory=list)
    result_score: Optional[Tuple[int, int]] = None
    result_note: Optional[str] = None
    is_draw: bool = False


def _parse_move_tokens(body: str) -> Tuple[List[Move], bool]:
    """Parse the move-list portion of a turn line. Returns (moves, is_skip)."""
    body = body.strip()
    if body == "--":
        return [], True
    moves: List[Move] = []
    for token in body.split():
        m = _MOVE_TOKEN_RE.match(token)
        if not m:
            raise ValueError(f"Bad move token: {token!r}")
        from_pt = int(m.group(1))
        to = m.group(2)
        repeat = int(m.group(3)) if m.group(3) else 1
        if to == "off":
            move = Move(from_pt, 0, True)
        else:
            move = Move(from_pt, int(to), False)
        moves.extend([move] * repeat)
    return moves, False


def parse_turn_line(line: str) -> Tuple[Color, Tuple[int, int], List[Move], bool]:
    """Parse a single turn line, returning (color, dice, moves, is_skip)."""
    m = _TURN_RE.match(line)
    if not m:
        raise ValueError(f"Bad turn line: {line!r}")
    _num, color_tag, d1, d2, body = m.groups()
    color = Color.WHITE if color_tag == "W" else Color.BLACK
    dice = (int(d1), int(d2))
    moves, is_skip = _parse_move_tokens(body)
    return color, dice, moves, is_skip


def parse_game(text: str) -> ParsedGame:
    """Parse a full .narde text into a ParsedGame. Unknown lines are ignored."""
    game = ParsedGame()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        h = _HEADER_RE.match(line)
        if h:
            game.headers[h.group(1)] = h.group(2)
            continue
        r = _RESULT_RE.match(line)
        if r:
            token, note = r.group(1), r.group(2)
            if token == "1/2-1/2":
                game.is_draw = True
                game.result_score = (0, 0)
            else:
                a, b = token.split("-")
                game.result_score = (int(a), int(b))
            game.result_note = note
            continue
        t = _TURN_RE.match(line)
        if t:
            number = int(t.group(1))
            color, dice, moves, is_skip = parse_turn_line(line)
            game.turns.append(ParsedTurn(number, color, dice, moves, is_skip))
            continue
    return game
