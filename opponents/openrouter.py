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


# Compact rulebook baked into every prompt. Keep this in sync with the
# project spec at docs/superpowers/specs/2026-04-21-narde-game-design.md —
# the LLM sees only what's listed here, so anything missing is "unknown".
#
# Board geometry and both players' paths live in BOARD_LAYOUT below; RULES
# deliberately avoids duplicating that content so there's one source of truth.
RULES = """Правила длинных нард (long backgammon), сжато:
* Доска из 24 пунктов (1..24). Геометрия и направление движения — см. раздел «Разметка доски» ниже.
* Пункт считается занятым соперником, если на нём стоит хотя бы одна его шашка. На занятый пункт вставать нельзя. Сбития шашек нет.
* За ход бросаются 2 кости. При дубле (одинаковые значения) кости играются 4 раза.
* Правило головы: за один ход со своей головы можно снять только одну шашку. Исключение — первый ход партии при 6-6, 4-4, 3-3: можно снять 2 шашки с головы, если иначе ход не строится из-за шашек соперника на голове.
* Правило полного хода: нужно использовать оба значения костей, если это возможно. Если играется ровно одно значение и кости разные — обязан играть большее.
* Блок из 6 своих подряд разрешён только если хотя бы одна шашка соперника уже находится в его доме; нельзя запереть все 15 шашек соперника.
* Выбрасывание (bearing off): доступно только когда все 15 своих шашек в своём доме. Выпало k — снимается шашка с пункта, находящегося в k шагах от выхода; если там пусто, можно снять с меньшего поля, но только если на всех бо́льших шашек тоже нет.
* Партия: марс (победитель выбросил все 15, соперник — 0) = 2 очка; ойн = 1 очко."""


# 2D layout of the board as rendered on screen, plus per-player paths.
# We teach the model this explicitly because the "ptN: M color" listing by
# itself gives no hint about quadrants, adjacency, or direction of travel.
BOARD_LAYOUT = """Разметка доски (нумерация 1..24, вид как на экране):

  верх-лево (дом чёрных)    │   верх-право
  ptN идут слева направо    │   ptN идут слева направо
    13 14 15 16 17 18       │    19 20 21 22 23 24 ← голова белых (pt24)
  ──────────────────────────┼──────────────────────────
    12 11 10  9  8  7       │     6  5  4  3  2  1
  ptN идут справа налево    │   ptN идут справа налево
  голова чёрных (pt12)      │   дом белых (pt1..6)
  низ-лево                  │   низ-право

Направление движения (оба игрока — против часовой стрелки):
* Белые: 24 → 19 → 18 → 13 → 12 → 7 → 6 → 1 → off (выход справа внизу).
* Чёрные: 12 → 7 → 6 → 1 → 24 → 19 → 18 → 13 → off (выход слева сверху).
* Голова — стартовый пункт своей стороны (белые: 24, чёрные: 12).
* Дом — последние 6 пунктов перед выбрасыванием (белые: 1..6, чёрные: 13..18)."""


NOTATION = """Нотация в этом промпте:
* Состояние доски задано строками вида «  ptN: M color», где N — номер пункта (1..24), M — сколько шашек на пункте, color — white/black.
* «Borne off» — сколько шашек каждая сторона уже выбросила (из 15). После 15 партия окончена.
* Ходы пишутся как «from/to» (например 24/21 — шашка с пункта 24 на 21) или «from/off» — выбрасывание. Последовательность ходов в рамках одного броска записывается через пробел."""


def build_prompt(board: Board, color: Color,
                 dice: Tuple[int, int],
                 sequences: List[List[Move]]) -> str:
    """Render the chat prompt sent to the OpenRouter model. Pure; test-friendly.

    The prompt is structured into five sections — rules, board layout,
    notation, the current position, and the enumerated legal sequences —
    followed by a strict three-line response schema (Оценка/Ход/Объяснение).
    Keeping rules, layout, and notation inside the prompt rather than tuning
    via the system role makes the interaction self-contained and replayable
    from a log."""
    lines = [
        "Ты играешь в длинные нарды (long backgammon) и сейчас выбираешь ход.",
        "",
        RULES,
        "",
        BOARD_LAYOUT,
        "",
        NOTATION,
        "",
        f"Сейчас ход: {color.value.upper()}.",
        f"Кости: {dice[0]}-{dice[1]}",
        _describe_board(board, color),
        "",
        "Варианты ходов (все — законные и максимальной длины):",
    ]
    for i, seq in enumerate(sequences, start=1):
        lines.append(f"  {i}) {_describe_sequence(seq)}")
    lines += [
        "",
        "Ответ строго тремя строками в этом порядке, каждая с меткой и "
        "двоеточием, по-русски:",
        "  Оценка: одна фраза (≤20 слов) — кто сейчас сильнее и почему "
        "(структура дома, блоки, позиция головы, запертые шашки и т.п.).",
        "  Ход: только целое число 1..N — номер выбранной последовательности.",
        "  Объяснение: одна фраза (≤20 слов) — что даёт именно этот ход.",
    ]
    return "\n".join(lines)


_INT_RE = re.compile(r'(\d+)')

# Labelled-line markers — matched case-insensitively; both Russian and a
# short English alias are accepted so the model can't miss by translating.
_LABEL_MOVE_RE = re.compile(
    r'^\s*(?:ход|move|choice|выбор)\s*[:\-]\s*(.*)$', re.IGNORECASE,
)
_LABEL_EVAL_RE = re.compile(
    r'^\s*(?:оценка|evaluation|position)\s*[:\-]\s*(.*)$', re.IGNORECASE,
)
_LABEL_REASON_RE = re.compile(
    r'^\s*(?:объяснение|пояснение|почему|reason|rationale|explanation)'
    r'\s*[:\-]\s*(.*)$',
    re.IGNORECASE,
)


def _find_labeled_line(text: str, pattern: "re.Pattern") -> Optional[str]:
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return None


def parse_reply(text: Optional[str], num_sequences: int) -> Optional[int]:
    """Extract a 1-based sequence index from `text`, returned 0-based.

    Prefers a line starting with `Ход:`/`Move:`/`Choice:`; falls back to
    the first non-empty line so legacy single-number replies keep working.
    Returns None when no integer is found or it's out of range."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    labeled = _find_labeled_line(text, _LABEL_MOVE_RE)
    target = labeled if labeled is not None else stripped.splitlines()[0]
    m = _INT_RE.search(target)
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < num_sequences:
        return idx
    return None


def extract_evaluation(text: Optional[str]) -> str:
    """Return the content of the `Оценка:` / `Evaluation:` line if any."""
    if text is None:
        return ""
    labeled = _find_labeled_line(text, _LABEL_EVAL_RE)
    return labeled if labeled is not None else ""


def extract_reason(text: Optional[str]) -> str:
    """Return the content of the `Объяснение:` / `Reason:` line if any;
    otherwise fall back to everything after the first line (legacy format)."""
    if text is None:
        return ""
    labeled = _find_labeled_line(text, _LABEL_REASON_RE)
    if labeled is not None:
        return labeled
    lines = text.strip().splitlines()
    return " ".join(ln.strip() for ln in lines[1:] if ln.strip()).strip()


class OpenRouterModel(BaseModel):
    """BaseModel implementation backed by the OpenRouter chat API.

    `http` is a callable prompt->reply_text; injectable for tests. On three
    consecutive unparsable replies the model delegates to a RandomLocalModel."""

    def __init__(self, api_key: str,
                 model: str = "openai/gpt-oss-120b:free",
                 http: Optional[Callable[[str], str]] = None,
                 rng: Optional[random.Random] = None,
                 max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self._http = http if http is not None else self._default_http
        self._fallback = RandomLocalModel(rng=rng)
        self._max_retries = max_retries
        # Populated by the last choose_move call so the UI can surface the
        # model's position assessment and explanation to the player.
        self.last_reason: str = ""
        self.last_evaluation: str = ""

    def _default_http(self, prompt: str) -> str:
        import requests
        url = "https://openrouter.ai/api/v1/chat/completions"
        print(f"[OPENROUTER] POST {url}  model={self.model}")
        print("[OPENROUTER] --- REQUEST PROMPT ---")
        print(prompt)
        print("[OPENROUTER] --- END REQUEST ---")
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if not r.ok:
            # Surface the server-side reason (e.g. "No endpoints found for
            # model X") so users can fix misconfiguration instead of staring
            # at a bare "404 Not Found".
            print(f"[OPENROUTER] HTTP {r.status_code} body: {r.text[:500]}")
            r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        print("[OPENROUTER] --- RESPONSE ---")
        print(reply)
        print("[OPENROUTER] --- END RESPONSE ---")
        return reply

    def choose_move(self, board: Board, color: Color,
                    dice: Tuple[int, int],
                    sequences: List[List[Move]]) -> List[Move]:
        self.last_reason = ""
        self.last_evaluation = ""
        if len(sequences) == 1:
            return sequences[0]
        prompt = build_prompt(board, color, dice, sequences)
        for attempt in range(1, self._max_retries + 1):
            try:
                reply = self._http(prompt)
            except Exception as e:
                print(f"[OPENROUTER] attempt {attempt} HTTP error: {e}")
                reply = None
            idx = parse_reply(reply, len(sequences))
            if idx is not None:
                self.last_evaluation = extract_evaluation(reply)
                self.last_reason = extract_reason(reply)
                msg = (f"[OPENROUTER] chose sequence "
                       f"{idx + 1}/{len(sequences)} on attempt {attempt}")
                if self.last_evaluation:
                    msg += f" — eval: {self.last_evaluation}"
                if self.last_reason:
                    msg += f" — reason: {self.last_reason}"
                print(msg)
                return sequences[idx]
            print(f"[OPENROUTER] attempt {attempt} returned unparsable reply")
        print(f"[OPENROUTER] {self._max_retries} failures — "
              f"falling back to RandomLocalModel")
        self.last_reason = "(fallback: случайный ход)"
        self.last_evaluation = ""
        return self._fallback.choose_move(board, color, dice, sequences)
