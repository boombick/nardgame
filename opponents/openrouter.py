import random
import re
from typing import Callable, List, Optional, Tuple

from engine.board import Board, Color, point_to_step, step_to_point
from engine.moves import Move, apply_single
from models.base import BaseModel
from opponents.local_model import RandomLocalModel


def _classify_point(pt: int, owner: Color) -> str:
    """Return a short role tag for `pt` from `owner`'s perspective.

    Produced for every checker's point in the listing so the model cannot
    misread a black checker on pt5 as "in its home" — it sees the literal
    string `[транзит через дом белых]`. The tags only encode board roles,
    not positional judgement, so they're safe to bake into the prompt.
    """
    # Heads
    if owner == Color.WHITE and pt == 24:
        return "голова белых"
    if owner == Color.BLACK and pt == 12:
        return "голова чёрных"
    # Own home
    if owner == Color.WHITE and 1 <= pt <= 6:
        return "дом белых"
    if owner == Color.BLACK and 13 <= pt <= 18:
        return "дом чёрных"
    # Opponent's home — transit only; the model keeps mistaking these for
    # bearing-off territory, hence the explicit "транзит" word.
    if owner == Color.WHITE and 13 <= pt <= 18:
        return "транзит через дом чёрных"
    if owner == Color.BLACK and 1 <= pt <= 6:
        return "транзит через дом белых"
    return ""


def _pip_count(board: Board, color: Color) -> int:
    """Thin wrapper around Board.pip_count kept for the module's internal
    callers and the existing prompt tests. Lower is better. At the starting
    position each side has 15 checkers × 24 pips = 360."""
    return board.pip_count(color)


def _stuck_in_opp_home(board: Board, color: Color) -> int:
    """How many of `color`'s checkers are currently sitting in the
    opponent's home zone (i.e. still in the long transit through enemy
    territory).

    These checkers have high pip cost and are the first thing a race-phase
    policy should unload. Observed failure mode: the LLM kept building
    "blocks" in the opponent's home instead of extracting the several
    checkers it had parked there, burning ~20 pips apiece to haul them
    back around. Surfacing this as a single integer in the prompt lets
    the model prioritise extraction concretely."""
    opp_home_range = range(13, 19) if color == Color.WHITE else range(1, 7)
    total = 0
    for pt in opp_home_range:
        ps = board.points[pt]
        if ps.color == color and ps.count > 0:
            total += ps.count
    return total


def _longest_opp_prime_ahead_of_stuck(board: Board, color: Color) -> int:
    """Length of the longest consecutive run of opponent-held (≥2 checkers)
    points on the escape path of `color`'s stuck-in-opp-home checkers.

    Walks MY step-space starting one step ahead of the rearmost stuck
    checker, scanning through the end of opponent's outer quadrant (step
    17 = one before opponent's head-quadrant exit in this layout). A
    run of 6 is a full prime that locks the stuck checkers until a 6-6
    jumps over it; anything ≥4 is already a strong trap worth reacting to.

    Returns 0 when there are no stuck checkers — in that case the metric
    is meaningless and we don't want to mislead the model into chasing
    phantom primes."""
    my_stuck = _stuck_in_opp_home(board, color)
    if my_stuck == 0:
        return 0
    opp = Color.BLACK if color == Color.WHITE else Color.WHITE
    opp_home_range = range(13, 19) if color == Color.WHITE else range(1, 7)
    rear_step: Optional[int] = None
    for pt in opp_home_range:
        ps = board.points[pt]
        if ps.color == color and ps.count > 0:
            s = point_to_step(pt, color)
            if rear_step is None or s < rear_step:
                rear_step = s
    assert rear_step is not None  # my_stuck > 0 guarantees at least one
    longest = 0
    cur = 0
    # Range 6..11 is opp's home in my step-space (transit); 12..17 is
    # opp's outer quadrant where primes typically extend from the home.
    # Past step 17 lies opp's own head/home-quadrant from my POV —
    # irrelevant to my stuck's escape.
    for step in range(rear_step + 1, 18):
        pt = step_to_point(step, color)
        ps = board.points[pt]
        if ps.color == opp and ps.count >= 2:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return longest


def _opponent_before_home(board: Board, color: Color) -> int:
    """How many of the opponent's checkers still sit outside their own home.

    "Before home" = step 0..17 in the opponent's step-space. When this is 0,
    the opponent has already collected everything into its home zone and
    blocking is pointless — any block in a transit zone would catch air.
    We expose this number raw in the prompt so the model can key its phase
    judgement on a single objective integer instead of guessing from the
    position listing."""
    opp = Color.BLACK if color == Color.WHITE else Color.WHITE
    total = 0
    for pt in range(1, 25):
        ps = board.points[pt]
        if ps.color == opp and ps.count > 0:
            if point_to_step(pt, opp) < 18:
                total += ps.count
    return total


def _phase(board: Board, color: Color) -> str:
    """Classify the board into a strategic phase from `color`'s POV.

    * "endgame" — all 15 of my checkers are in my home; bearing-off dominates.
    * "race"    — opponent has nothing left to block (all in their home); no
                  point building walls, maximise Δpip.
    * "contact" — default: paths still cross, blocks matter.

    The classification is deterministic so the prompt can print one label
    instead of asking the LLM to derive it. This removes the recurring
    failure where the model yells "блок!" in positions where no block can
    possibly catch anything."""
    if board.all_in_home(color):
        return "endgame"
    if _opponent_before_home(board, color) == 0:
        return "race"
    return "contact"


def _delta_pip(board: Board, color: Color, seq: List[Move]) -> int:
    """Pip-reduction achieved by applying `seq` to a clone of `board`.

    Positive = my pip count decreased by that many. `apply_single` mutates
    in place, so we clone first. This is the raw scoring signal the model
    needs to compare race candidates head-to-head without doing the sum
    itself (and getting it wrong — observed in logs)."""
    before = _pip_count(board, color)
    sim = board.clone()
    for m in seq:
        apply_single(sim, color, m)
    after = _pip_count(sim, color)
    return before - after


def _is_dead_zone_move(board: Board, color: Color, seq: List[Move]) -> bool:
    """True if `seq` drops a checker into the opponent's home *and* the
    opponent is already fully collected there.

    "Dead zone" because once the opponent's checkers are all in their home,
    a block on that home catches zero checkers — it's just pip wasted on
    the long way back. We only flag moves in `race` phase (i.e. blocking
    impossible). In `contact` the same landing might set up a future block,
    so we leave it alone."""
    if _phase(board, color) != "race":
        return False
    # Opponent's home = my "mirror" home by point index.
    opp_home_range = range(13, 19) if color == Color.WHITE else range(1, 7)
    sim = board.clone()
    before = sum(1 for pt in opp_home_range
                 if sim.points[pt].color == color
                 and sim.points[pt].count > 0)
    for m in seq:
        apply_single(sim, color, m)
    after = sum(1 for pt in opp_home_range
                if sim.points[pt].color == color
                and sim.points[pt].count > 0)
    # Landing more own checkers in the opponent's home after the move than
    # before = this sequence parks something in the dead zone.
    return after > before


def _head_left(board: Board, color: Color) -> int:
    """How many of color's 15 checkers still sit on their head point."""
    head_pt = 24 if color == Color.WHITE else 12
    ps = board.points[head_pt]
    return ps.count if ps.color == color else 0


def _describe_board(board: Board, color: Color) -> str:
    """Compact text description of the board from `color`'s perspective.

    Each occupied point carries a `[role]` tag for its owner (`голова
    чёрных`, `дом белых`, `транзит через дом чёрных`, …) so the model
    cannot confuse its own home with the opponent's. A summary line adds
    pip counts and how many checkers are still stuck on each head — both
    objective numbers the model would otherwise have to derive itself."""
    rows = []
    for pt in range(1, 25):
        ps = board.points[pt]
        if ps.count > 0 and ps.color is not None:
            tag = _classify_point(pt, ps.color)
            suffix = f"  [{tag}]" if tag else ""
            rows.append(f"  pt{pt}: {ps.count} {ps.color.value}{suffix}")
    borne = board.borne_off
    head_w = _head_left(board, Color.WHITE)
    head_b = _head_left(board, Color.BLACK)
    pip_w = _pip_count(board, Color.WHITE)
    pip_b = _pip_count(board, Color.BLACK)
    phase = _phase(board, color)
    opp_before = _opponent_before_home(board, color)
    my_stuck = _stuck_in_opp_home(board, color)
    opp_prime = _longest_opp_prime_ahead_of_stuck(board, color)
    # Derived guidance per phase — written out so the model has a single
    # canonical instruction for each regime instead of re-deriving it.
    phase_ru = {
        "contact": "контакт (пути пересекаются, блоки осмысленны)",
        "race":    "гонка (сопернику уже нечем грозить — максимизируй Δpip)",
        "endgame": "эндшпиль (все свои в доме — равномерно заполняй без дыр)",
    }[phase]
    # "Contact" with zero stuck is a soft race: any block you'd build in
    # opp's home now catches no one of yours because none of yours has to
    # cross it anymore. Flag it so the model stops chanting "мини-блок".
    if phase == "contact" and my_stuck == 0:
        phase_ru += (" — но твоих стуков 0: блоки в доме соперника ценности "
                     "уже не имеют, твои шашки туда не вернутся")
    opp_color_ru = "белых" if color == Color.BLACK else "чёрных"
    # Prime-ahead-of-stuck is only informative when stuck > 0. Render it
    # with an actionable warning at ≥4; below that it's just context.
    if my_stuck > 0:
        if opp_prime >= 6:
            prime_tail = (f": {opp_prime} — полный блок, выход только через "
                          f"дубль 6-6; приоритет #1 вытащить стуки сейчас")
        elif opp_prime >= 4:
            prime_tail = (f": {opp_prime} — сильный заслон, пока ещё можно "
                          f"проскочить; приоритет #1 вытащить стуки сейчас")
        else:
            prime_tail = f": {opp_prime}"
        prime_line = ("\nСамый длинный заслон соперника (≥2 шашки подряд) "
                      f"перед твоими застрявшими{prime_tail}.")
    else:
        prime_line = ""
    return (
        f"On-turn: {color.value.upper()}\n"
        f"Board:\n" + "\n".join(rows) +
        f"\nШашек на голове (ещё не тронулись): белые {head_w}/15, "
        f"чёрные {head_b}/15."
        f"\nШашек ушло с головы: белые {15 - head_w}/15, "
        f"чёрные {15 - head_b}/15."
        f"\nPip count (сумма шагов до выхода у всех шашек; меньше = ближе "
        f"к победе):"
        f"\n  белые: {pip_w}"
        f"\n  чёрные: {pip_b}"
        f"\nBorne off: white={borne[Color.WHITE]}, "
        f"black={borne[Color.BLACK]}"
        f"\nШашек соперника ещё не в своём доме: {opp_before} {opp_color_ru} "
        f"из 15 (когда 0 — блокировать уже нечего, идёт гонка)."
        f"\nТвоих шашек застряло в доме соперника (транзит, длинный путь "
        f"до выхода): {my_stuck}." + prime_line +
        f"\nФаза: {phase_ru}."
    )


def _describe_sequence(seq: List[Move]) -> str:
    """Render a move sequence so it's unambiguous *which* checker moves.

    Chained moves (where one move's destination is the next move's source)
    are merged into a single arrow chain like `12→6→4`, signalling that the
    same checker passes through the intermediate point without stopping.
    Different checkers within the same turn are separated by `, `. Bear-offs
    render as `→off`. Example:
        [12/6, 6/4, 24/21]          → "12→6→4, 24→21"
        [9/4, 4/off]                → "9→4→off"

    We avoid `notation.writer.format_move` here because the .narde save
    format intentionally uses `/` and must stay stable across versions; the
    prompt rendering is free to use a clearer in-place notation."""
    if not seq:
        return "(skip)"
    groups: List[List[Move]] = []
    for m in seq:
        if (groups
                and not groups[-1][-1].is_bear_off
                and groups[-1][-1].to_point == m.from_point):
            groups[-1].append(m)
        else:
            groups.append([m])
    rendered: List[str] = []
    for g in groups:
        parts = [str(g[0].from_point)]
        for m in g:
            parts.append("off" if m.is_bear_off else str(m.to_point))
        rendered.append("→".join(parts))
    return ", ".join(rendered)


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
* Дом — последние 6 пунктов перед выбрасыванием (белые: 1..6, чёрные: 13..18).

Дома не пересекаются:
* Чёрные не выбрасываются из pt1..6 — это дом белых. Чёрная шашка на pt1..6 проходит транзитом по пути к перевалу через pt24 и продолжит 24→19→…→13.
* Белые не выбрасываются из pt13..18 — это дом чёрных. Белая шашка на pt13..18 проходит транзитом по пути к pt12→7→6→1.

Стартовая расстановка: 15 белых на pt24, 15 чёрных на pt12. Большое число шашек на своей голове — это нетронутый старт, т.е. *отставание*, а не достижение. Продвижение считается по шашкам, уже ушедшим с головы."""


NOTATION = """Нотация в этом промпте:
* Состояние доски задано строками вида «  ptN: M color», где N — номер пункта (1..24), M — сколько шашек на пункте, color — white/black.
* «Borne off» — сколько шашек каждая сторона уже выбросила (из 15). После 15 партия окончена.
* Отдельный ход пишется как «from→to» (например 24→21 — шашка с пункта 24 на 21) или «from→off» — выбрасывание.
* Если в варианте стрелки идут подряд без запятой (например «12→6→4»), это означает, что *одна и та же шашка* прошла 12→6, затем 6→4. На промежуточных пунктах (здесь pt6) она не остаётся — ставить блок или занять пункт она не может.
* Разные шашки в одном броске разделяются запятой: «24→21, 12→10» — две разные шашки, одна с pt24 на pt21, вторая с pt12 на pt10."""


# Strategic guidance. Without this the model reduces every decision to
# "minimise my pip count this turn" — which loses to blocking and to
# premature home-collection. Laid out as three phases plus named concepts
# (арьергард, темп, марс-рескью) so the model can refer back to them in
# its Оценка/Объяснение lines.
STRATEGY = """Стратегия (обязательно учитывай при оценке и выборе хода):

Три фазы партии:
1. Гонка (race) — траектории сторон не пересекаются или почти не пересекаются, блокировать соперника уже невозможно. Задача: минимизировать pip count, выбрасывать максимально эффективно.
2. Блокирование (blocking) — шашки соперника ещё должны пройти через твою зону. Задача: выстраивать заслоны из 2+ подряд занятых пунктов перед бегущими шашками соперника. Лучший заслон — 6 подряд (полный блок, если это разрешено правилом 6-блока).
3. Эндшпиль — когда почти все шашки в доме. Задача: ровно заполнять дом, не плодить «дыры», держать шашки далеко от выхода пока не выпали нужные кости.

Ключевые понятия:
* Арьергард — твои последние шашки (ближайшие к голове). Их не надо гнать вслепую: один-два арьергардных бойца можно использовать, чтобы поздно пройти мимо соперника и успеть поставить неожиданный блок. Арьергард ценен именно *угрозой* блока.
* Темп — кто опережает по pip count. Ведущий играет на гонку (безопасный разгон вперёд), отстающий играет на блок (пытается задержать соперника своими пунктами в его транзитной зоне).
* Марс/ойн — проигрыш с 0 выброшенных = марс (-2 очка), с ≥1 выброшенным = ойн (-1). Если проигрываешь гонку, имеет смысл жертвовать темпом ради блокировки соперника, чтобы успеть выбросить хотя бы одну шашку (рескью от марса).

Типичные ошибки, которых надо избегать:
* Не собирай весь дом рано. Если у соперника ещё идут шашки через твою зону, ранний перевод всех своих в дом = потеря возможности блокировать. Держи 2-4 шашки на подходе к дому как заслон, пока соперник не пройдёт.
* Не ставь одинокие шашки на пункты, где соперник может построить свой блок, запирая их. В длинных нардах сбития нет, но запертые шашки = мёртвый груз по pip.
* Не гони всё подряд на голову-2/голову-3 в первые ходы: это короткий темп без структуры. Ищи ходы, которые создают связку из 2+ своих подряд или идут в дом по плану.

Как выбирать ход:
1. Определи текущую фазу (гонка / блок / эндшпиль).
2. Сравни pip count — ты ведёшь или отстаёшь.
3. Из двух кандидатов предпочти тот, что либо создаёт/удерживает заслон против бегущих шашек соперника, либо (в гонке) продвигает самую заднюю собственную шашку, либо (в эндшпиле) равномерно заполняет дом без дыр.
4. Не выбирай ход просто по минимальному pip: иногда правильный ход оставляет pip чуть выше, но задерживает соперника на несколько ходов."""


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
        STRATEGY,
        "",
        f"Сейчас ход: {color.value.upper()}.",
        f"Кости: {dice[0]}-{dice[1]}",
        _describe_board(board, color),
        "",
        "Варианты ходов (все — законные и максимальной длины):",
    ]
    for i, seq in enumerate(sequences, start=1):
        # Δpip and dead-zone flags let the model compare candidates without
        # mentally simulating each one — pure pip-race decisions can be
        # made by index lookup, and parking-into-a-dead-home is visible at
        # a glance.
        tags: List[str] = []
        dpip = _delta_pip(board, color, seq)
        # Use the Unicode minus (U+2212) so the sign reads as a math minus
        # in logs, not a hyphen-run-together.
        tags.append(f"Δpip=\u2212{dpip}" if dpip >= 0
                    else f"Δpip=+{-dpip}")
        if _is_dead_zone_move(board, color, seq):
            tags.append("мёртвая зона: шашка идёт в дом соперника, "
                        "где блок уже ничего не даёт")
        suffix = "  (" + "; ".join(tags) + ")" if tags else ""
        lines.append(f"  {i}) {_describe_sequence(seq)}{suffix}")
    lines += [
        "",
        "Ответ строго тремя строками в этом порядке, каждая с меткой и "
        "двоеточием, по-русски:",
        "  Оценка: одна фраза (≤25 слов) — укажи фазу партии "
        "(гонка / блок / эндшпиль), кто ведёт по pip, и свой план на 2-3 "
        "хода вперёд (какой заслон строишь, какую шашку гонишь, что "
        "удерживаешь).",
        "  Ход: только целое число 1..N — номер выбранной последовательности.",
        "  Объяснение: одна фраза (≤20 слов) — как выбранный ход воплощает "
        "план из Оценки.",
        "",
        # Guardrails distilled from observed model mistakes: LLMs trained
        # on short/classical backgammon tend to drag in hit/blot concepts
        # and misread the starting head stack as "advanced". State both
        # traps explicitly so the model can't silently apply them.
        "ВАЖНО при оценке:",
        "* В длинных нардах НЕТ сбития шашек (no hitting). Не используй "
        "понятия «давление на блот», «попадание», «снятие», «удар» — их "
        "здесь нет. Одиночная шашка соперника неуязвима.",
        "* Шашки на своей голове — отстающие, а не продвинутые. Если на "
        "pt24 стоит 14 белых — это значит ушла ровно 1 белая, а не 14.",
        "* Траектория «a→b→c» — одна шашка, проходящая через b. На b она "
        "не остаётся; ни блока, ни занятия pt=b этот ход не создаёт.",
        "* Дом белых = pt1..6, дом чёрных = pt13..18 — они разные и не "
        "пересекаются. Свой цвет на pt1..6 — это дом только для белых; "
        "чёрная шашка там — транзит, а не готовность к выбрасыванию. И "
        "наоборот для pt13..18.",
        "* Для сравнения сторон опирайся на pip count и число шашек, "
        "ушедших с головы, из описания позиции выше — это объективные "
        "числа. Кто ближе к победе — у кого pip меньше.",
        "* Блок — это ≥2 твоих шашек на каждом из ≥2 соседних пунктов "
        "подряд (например 2+ на pt14 И 2+ на pt15). Одиночная шашка "
        "(ровно 1) на пункте — это НЕ блок: соперник проходит её любым "
        "броском. «Блок 13-14» с одиночками в обоих пунктах — фикция.",
        "* Блок (настоящий) на пути бегущих шашек соперника ценнее, "
        "чем -1 pip на этом ходу. Ищи возможности построить или удержать "
        "заслон, особенно если отстаёшь по pip.",
        "* В фазе «гонка» правило жёсткое: выбирай кандидата с "
        "максимальным Δpip. «Укрепить блок», «удержать заслон», «создать "
        "мини-блок» в гонке = отказ от победы. Единственное исключение — "
        "если у соперника осталась ровно одна шашка вне его дома и твой "
        "ход ставит на её пути заслон из ≥2 шашек на соседних пунктах; "
        "тогда заслонный ход допустим даже с меньшим Δpip.",
        "* Кандидаты помечены метрикой «(Δpip=−N)» — насколько ход снижает "
        "твой pip. В гонке и эндшпиле ориентируйся на это число.",
        "* Строка «Твоих шашек застряло в доме соперника: N» — это твои "
        "шашки, которые ещё в транзите. У них самый длинный путь до выхода "
        "(18+ шагов) и они уязвимы для блока соперника. Если N>0 и ты в "
        "гонке, каждый ход должен либо вытаскивать одну из них, либо "
        "(если не получается) иметь максимальный Δpip среди альтернатив. "
        "Парковать ещё одну шашку в доме соперника в гонке — запрещено.",
        "* Метка «мёртвая зона» у кандидата означает, что шашка заходит в "
        "дом соперника, где все соперниковы шашки уже собрались — такой "
        "ход не блокирует ничего и уводит шашку на длинный круг обратно. "
        "Не выбирай такой ход, если есть альтернатива без этой метки.",
        "* Строка «Самый длинный заслон соперника перед твоими "
        "застрявшими: N» считает, сколько пунктов подряд в твоей "
        "транзитной зоне соперник держит минимум двумя шашками. N=6 = "
        "полный заслон (6-прайм): твои стуки в капкане, пока не выпадет "
        "6-6. N=5 = почти капкан. При N≥4 и stuck>0 — игнорируй любые "
        "«усиление дома» и «мини-блок», единственная разумная цель "
        "хода — вытащить застрявшую шашку из дома соперника (лучше всего "
        "компаундом, прыгая сразу за арьергард соперника).",
        "* Соответственно, пока соперник ещё *строит* заслон (N=3..5), у "
        "тебя короткое окно, чтобы вывести стуков до того, как появится "
        "шестой пункт. Каждый «потерянный» на укрепление дома ход =  "
        "подарок сопернику на закрытие капкана.",
        "* Если у тебя stuck = 0, то даже в фазе «контакт» любые «блоки» "
        "и «заслоны» внутри дома соперника — самообман: твои шашки туда "
        "больше не вернутся, ловить там нечего. Осмысленный блок в этой "
        "ситуации — только на пути шашек соперника к его дому, то есть "
        "где-то в твоей транзитной зоне (pt13..18 для чёрных / pt1..6 для "
        "белых) по его маршруту.",
        "* Пока у тебя stuck > 0, не торопись снимать последнюю свою шашку "
        "с пунктов сразу за домом соперника — pt24 (для чёрных) / pt12 "
        "(для белых). Пока на этом пункте стоит твоя шашка, соперник "
        "туда встать не может (правило «занятого пункта»), и твой стук "
        "всегда имеет эту точку как безопасную посадку. Снимешь — "
        "сопернику достаточно одной шашки, чтобы её перекрыть, и у твоих "
        "стуков сразу схлопывается выход из его дома.",
        "* Не собирай весь дом рано: пока соперник ещё идёт через твою "
        "транзитную зону, держи 2-4 шашки снаружи дома как потенциальный "
        "заслон. Ранний сбор всех в дом = отказ от блокирования.",
        "* Арьергардные шашки (последние, ближе к голове) — ресурс угрозы "
        "блоком, а не отстающий балласт. Не обязательно гнать их первыми.",
        "* Если проигрываешь по pip — играй от блокирования, задержи "
        "соперника, чтобы успеть выбросить хотя бы одну шашку (иначе марс, "
        "-2 очка вместо -1).",
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
        # The full prompt is deterministic given board+color+dice (all
        # three are logged in choose_move), so printing it here every turn
        # is just noise. We keep the one-line URL marker and the response
        # block because the response is what actually differs per call.
        import requests
        url = "https://openrouter.ai/api/v1/chat/completions"
        print(f"[OPENROUTER] POST {url}  model={self.model}")
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
        # Log just what varies turn to turn: the dice and the compact
        # board description. Full prompt (rules/layout/notation) is
        # identical every turn and lives in the source.
        print(f"[OPENROUTER] {color.value} to move, dice={dice[0]}-{dice[1]}")
        print(_describe_board(board, color))
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
