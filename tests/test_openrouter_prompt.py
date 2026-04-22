import random

from engine.board import Board, Color
from engine.moves import Move
from opponents.openrouter import (
    OpenRouterModel, _delta_pip, _describe_board, _describe_sequence,
    _is_dead_zone_move, _longest_opp_prime_ahead_of_stuck,
    _opponent_before_home, _phase, _pip_count, _stuck_in_opp_home,
    build_prompt, extract_evaluation, extract_reason, parse_reply,
)


class TestBuildPrompt:
    def test_prompt_contains_dice_and_sequences(self):
        # Two moves off the head (head-rule is orthogonal — we're only
        # exercising arrow notation and the physical applicability of the
        # sequence to the start board).
        seqs = [[Move(24, 21, False), Move(24, 19, False)],
                [Move(24, 19, False), Move(24, 18, False)]]
        p = build_prompt(Board(), Color.WHITE, (3, 5), seqs)
        assert "3-5" in p
        assert "24→21" in p and "24→19" in p
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
        # Describes the ptN/from→to/off notation used inline.
        assert "pt" in p.lower()
        assert "off" in p.lower()
        assert "from→to" in p.lower() or "from→off" in p.lower()

    def test_prompt_warns_about_starting_position(self):
        # Without this the model reads "pt24: 14 white" as "14 advanced"
        # instead of "14 still on the head, only one moved".
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        assert "старт" in low or "нетронут" in low
        assert "отстав" in low or "не достижение" in low

    def test_prompt_warns_about_opposite_home(self):
        # The model keeps treating pt1..6 as black's home and bearing-off
        # zone. The prompt must explicitly say which side bears off where.
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        assert "чёрные не выбрасываются из pt1..6" in low
        assert "белые не выбрасываются из pt13..18" in low

    def test_prompt_position_tags_head_and_home(self):
        # With checkers on every role-bearing point we should see each
        # labelled so the LLM cannot confuse black's head for black's home.
        b = Board()  # 15 white on pt24, 15 black on pt12 — both heads
        b.points[5].count = 1           # black transiting through white's home
        b.points[5].color = Color.BLACK
        b.points[15].count = 1          # white transiting through black's home
        b.points[15].color = Color.WHITE
        p = build_prompt(b, Color.BLACK, (3, 5),
                         [[Move(12, 9, False)]])
        # Heads labelled
        assert "pt12: 15 black  [голова чёрных]" in p
        assert "pt24: 15 white  [голова белых]" in p
        # Transit points labelled — critical: black on pt5 is NOT "in home"
        assert "pt5: 1 black  [транзит через дом белых]" in p
        assert "pt15: 1 white  [транзит через дом чёрных]" in p

    def test_prompt_position_tags_own_home(self):
        b = Board()
        b.points[3].count = 2           # white in its own home
        b.points[3].color = Color.WHITE
        b.points[16].count = 2          # black in its own home
        b.points[16].color = Color.BLACK
        p = build_prompt(b, Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        assert "pt3: 2 white  [дом белых]" in p
        assert "pt16: 2 black  [дом чёрных]" in p

    def test_prompt_includes_pip_count_and_head_counters(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        # At the starting position both sides have 15 checkers × 24 pips = 360.
        assert "белые: 360" in p
        assert "чёрные: 360" in p
        assert "ушло с головы" in p
        # Start: nobody has left their head.
        assert "белые 0/15" in p
        assert "чёрные 0/15" in p

    def test_prompt_includes_strategy_section(self):
        # Without strategic guidance the model treats every turn as pure
        # pip-reduction. We bake in race/block/endgame framing plus named
        # concepts the model can refer back to.
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        assert "стратеги" in low
        assert "гонк" in low           # race phase
        assert "блокиров" in low       # blocking phase
        assert "эндшпил" in low        # endgame
        assert "арьергард" in low      # rear checkers — a named concept
        assert "не собирай весь дом рано" in low

    def test_prompt_response_schema_asks_for_phase_and_plan(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        # Evaluation line must ask for the strategic phase and a multi-turn
        # plan, not just "who's stronger".
        assert "фаз" in low and "план" in low

    def test_prompt_warns_no_hitting_classical_jargon(self):
        # The previous reply used classical-backgammon concepts ("давление
        # на одиночку", "блокируя пункт") that don't apply here.
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        assert "сбити" in low or "no hitting" in low
        assert "давлени" in low or "блот" in low or "попадани" in low

    def test_prompt_requests_three_sections(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        assert "оценка" in low
        assert "ход" in low
        assert "объяснени" in low


class TestPhase:
    def test_start_position_is_contact(self):
        # Opening: both sides have 15 checkers in front of their homes,
        # paths still overlap — classic blocking game.
        assert _phase(Board(), Color.WHITE) == "contact"
        assert _phase(Board(), Color.BLACK) == "contact"

    def test_no_opponent_before_home_is_race(self):
        # Black has run everything into its home (pt13..18). White's own
        # checkers still need travel, but blocks don't matter — black is
        # uncatchable. This is pure race from white's POV.
        b = Board()
        # Clear black's head.
        b.points[12].count = 0
        b.points[12].color = None
        # All 15 black checkers inside their home (pt18).
        b.points[18].count = 15
        b.points[18].color = Color.BLACK
        # White still on the head — not yet in endgame.
        assert _phase(b, Color.WHITE) == "race"

    def test_all_own_in_home_is_endgame(self):
        # White has all 15 in its home → bearing-off phase. Black's state
        # irrelevant for this judgement (it's already "my endgame").
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[6].count = 15
        b.points[6].color = Color.WHITE
        assert _phase(b, Color.WHITE) == "endgame"


class TestOpponentBeforeHome:
    def test_start_all_15_each_side(self):
        # Nobody has moved — 15 of each side are outside their home zone.
        b = Board()
        assert _opponent_before_home(b, Color.WHITE) == 15   # 15 black
        assert _opponent_before_home(b, Color.BLACK) == 15   # 15 white

    def test_zero_when_opponent_all_in_home(self):
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[1].count = 15
        b.points[1].color = Color.WHITE
        # From black's POV the opponent (white) is fully in pt1..6.
        assert _opponent_before_home(b, Color.BLACK) == 0


class TestStuckInOppHome:
    def test_start_position_zero_for_both(self):
        # On the starting position nobody has transited into the other
        # side's home yet.
        b = Board()
        assert _stuck_in_opp_home(b, Color.WHITE) == 0
        assert _stuck_in_opp_home(b, Color.BLACK) == 0

    def test_black_checker_in_white_home_is_stuck(self):
        # Move a black checker into pt3 (inside white's home = black's
        # transit). That's one stuck for black, zero for white.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(3, Color.BLACK)
        assert _stuck_in_opp_home(b, Color.BLACK) == 1
        assert _stuck_in_opp_home(b, Color.WHITE) == 0

    def test_white_checker_in_black_home_is_stuck(self):
        # Symmetric: a white checker on pt15 (inside black's home = white
        # transit) counts as stuck for white.
        b = Board()
        b.remove_one(24, Color.WHITE)
        b.place_one(15, Color.WHITE)
        assert _stuck_in_opp_home(b, Color.WHITE) == 1
        assert _stuck_in_opp_home(b, Color.BLACK) == 0


class TestPromptShowsStuckCounter:
    def test_prompt_mentions_stuck_line(self):
        # The literal prefix of the new line must appear in the prompt so
        # the model can key off it in the body of its reasoning.
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        assert "застряло в доме соперника" in p.lower()


class TestOppPrimeAheadOfStuck:
    def test_zero_when_no_stuck(self):
        # Start position: nobody stuck, so the metric is 0 and doesn't
        # mislead the model into reading phantom opponent primes.
        assert _longest_opp_prime_ahead_of_stuck(Board(), Color.WHITE) == 0
        assert _longest_opp_prime_ahead_of_stuck(Board(), Color.BLACK) == 0

    def test_single_opp_point_with_count_one_is_ignored(self):
        # A lone opp checker (count=1) is not a blocker for landing so
        # it never counts toward the prime length. Set up black stuck on
        # pt5 and a single white on pt3; also clear pt24 stack so it
        # doesn't register as an independent 1-long run.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(5, Color.BLACK)
        # Clear pt24 stack directly — it's initial-position scaffolding
        # that would otherwise register as an opp block of length 1.
        b.points[24].count = 0
        b.points[24].color = None
        b.points[3].count = 1
        b.points[3].color = Color.WHITE
        assert _longest_opp_prime_ahead_of_stuck(b, Color.BLACK) == 0

    def test_prime_of_two_counted(self):
        # Black stuck on pt5 (step 7). Build a mini-prime at pt3..pt4 with
        # 2 white each. Clear pt24 so only pt3 and pt4 register as ≥2
        # white in the scan; longest should be 2.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(5, Color.BLACK)
        b.points[24].count = 0
        b.points[24].color = None
        b.points[3].count = 2
        b.points[3].color = Color.WHITE
        b.points[4].count = 2
        b.points[4].color = Color.WHITE
        assert _longest_opp_prime_ahead_of_stuck(b, Color.BLACK) == 2

    def test_full_six_prime_trap(self):
        # The user's scenario: opponent has 6 consecutive points held with
        # ≥2 checkers each directly in front of black's stuck. Black stuck
        # on pt6 (step 6), white with ≥2 on pt1..pt5 and pt24 — in black's
        # step space those are steps 7, 8, 9, 10, 11, 12 consecutive.
        # Expected: 6 (locked-in prime).
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(6, Color.BLACK)  # stuck on pt6
        # Need 2 whites each on pt1, pt2, pt3, pt4, pt5. That's 10 whites.
        # Start has 15 on pt24. Also keep some on pt24 (so there's a
        # "white 2+" at step 12 in black's space extending the prime).
        targets = [1, 2, 3, 4, 5]
        for pt in targets:
            for _ in range(2):
                b.remove_one(24, Color.WHITE)
                b.place_one(pt, Color.WHITE)
        # pt24 still has 5 whites (2+), extending the run by one more step.
        assert _longest_opp_prime_ahead_of_stuck(b, Color.BLACK) == 6

    def test_gap_breaks_the_run(self):
        # Black stuck on pt6. White ≥2 on pt5, pt4, pt2, pt1 (missing pt3
        # = gap). Clear pt24 so it doesn't extend the pt1 run. Two runs
        # of length 2 each (pt5-pt4 and pt2-pt1) — longest is 2, not 4.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(6, Color.BLACK)
        b.points[24].count = 0
        b.points[24].color = None
        for pt in (1, 2, 4, 5):
            b.points[pt].count = 2
            b.points[pt].color = Color.WHITE
        assert _longest_opp_prime_ahead_of_stuck(b, Color.BLACK) == 2

    def test_own_presence_breaks_the_run(self):
        # Black stuck on pt6 (rearmost) AND another black on pt3 acting
        # as an anchor. White ≥2 on pt1, pt2, pt4, pt5. Clear pt24 first.
        # The pt3 with black breaks what would otherwise be a 5-run
        # (pt1..pt5) into 2 and 2.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(6, Color.BLACK)  # rearmost stuck
        b.remove_one(12, Color.BLACK)
        b.place_one(3, Color.BLACK)  # own anchor inside opp home
        b.points[24].count = 0
        b.points[24].color = None
        for pt in (1, 2, 4, 5):
            b.points[pt].count = 2
            b.points[pt].color = Color.WHITE
        assert _longest_opp_prime_ahead_of_stuck(b, Color.BLACK) == 2

    def test_scans_from_rearmost_not_frontmost_stuck(self):
        # Two stuck: one on pt6 (rearmost, step 6) and one on pt2
        # (frontmost, step 10). A prime between them should count from
        # pt6's perspective — so pt5/pt4 with ≥2 white counts as 2.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(6, Color.BLACK)
        b.remove_one(12, Color.BLACK)
        b.place_one(2, Color.BLACK)
        b.points[24].count = 0
        b.points[24].color = None
        for pt in (4, 5):
            b.points[pt].count = 2
            b.points[pt].color = Color.WHITE
        assert _longest_opp_prime_ahead_of_stuck(b, Color.BLACK) == 2


class TestPromptShowsPrimeThreat:
    def test_prime_line_appears_when_stuck_and_prime_exist(self):
        # Black stuck on pt6 with a 2-prime on pt4-pt5: prompt must surface
        # the metric so the model can react. Clear pt24 to isolate.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(6, Color.BLACK)
        b.points[24].count = 0
        b.points[24].color = None
        for pt in (4, 5):
            b.points[pt].count = 2
            b.points[pt].color = Color.WHITE
        p = build_prompt(b, Color.BLACK, (3, 5),
                         [[Move(12, 9, False)]])
        low = p.lower()
        assert "самый длинный заслон соперника" in low
        # With this layout the prime run is 2 and should appear as a number.
        assert "перед твоими застрявшими: 2" in low

    def test_prime_line_absent_when_no_stuck(self):
        # Without stuck the metric is undefined/0 and should NOT be
        # rendered in the describe-board block (the prompt rules section
        # still mentions the metric as a rule, so we match the specific
        # describe-line signature with the "(≥2 шашки подряд)" parenthetical).
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        assert "(≥2 шашки подряд) перед твоими застрявшими" not in p.lower()

    def test_prime_line_warns_at_six(self):
        # 6-prime = full lockout; the warning text must say so explicitly.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(6, Color.BLACK)
        for pt in (1, 2, 3, 4, 5):
            for _ in range(2):
                b.remove_one(24, Color.WHITE)
                b.place_one(pt, Color.WHITE)
        # pt24 still ≥2 white — extends to 6.
        p = build_prompt(b, Color.BLACK, (3, 5),
                         [[Move(12, 9, False)]])
        low = p.lower()
        assert "перед твоими застрявшими: 6" in low
        assert "полный блок" in low or "6-6" in low


class TestPhaseAnnotationWhenStuckZero:
    def test_contact_phase_gets_soft_race_note_when_my_stuck_is_zero(self):
        # Start position: contact phase, stuck = 0. The phase line must
        # carry a note that blocks inside opp's home no longer pay off.
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        assert "фаза: контакт" in low
        assert "но твоих стуков 0" in low

    def test_contact_phase_no_note_when_stuck_exists(self):
        # Black with one stuck on pt5: the "stuck = 0" soft-race note must
        # NOT appear — blocks in opp home are still meaningful for
        # opponent's own traffic.
        b = Board()
        b.remove_one(12, Color.BLACK)
        b.place_one(5, Color.BLACK)
        p = build_prompt(b, Color.BLACK, (3, 5),
                         [[Move(12, 9, False)]])
        low = p.lower()
        assert "но твоих стуков 0" not in low


class TestPromptPrimeAndAnchorRules:
    def test_prompt_explains_prime_threshold_rule(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        # Explicit rule: at N≥4 with stuck>0, escape is priority #1.
        assert "n≥4" in low or "n=6" in low
        assert "вытащить застрявш" in low

    def test_prompt_explains_stuck_zero_block_futility(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        # Rule about blocks inside opp home being futile when stuck=0.
        assert "stuck = 0" in low or "stuck=0" in low
        assert "самообман" in low or "ловить там нечего" in low

    def test_prompt_explains_anchor_preservation(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        # Rule about not vacating pt24/pt12 while stuck > 0.
        assert "pt24" in low and "pt12" in low
        assert "якор" in low or "безопасн" in low


class TestDeltaPip:
    def test_single_move_delta_equals_die(self):
        # 24→21 = step 0 → step 3, so 3 pips less.
        assert _delta_pip(Board(), Color.WHITE, [Move(24, 21, False)]) == 3

    def test_chained_move_sums_pips(self):
        # 12→6 is 6 pips for black (step 0 → 6), 6→4 is 2 more (step 6 → 8).
        seq = [Move(12, 6, False), Move(6, 4, False)]
        assert _delta_pip(Board(), Color.BLACK, seq) == 8

    def test_bear_off_delta_is_pips_to_exit(self):
        # Put a white checker at pt6 (step 18, 6 pips to exit). Bearing off
        # it erases those 6 pips.
        b = Board()
        b.points[24].count = 14
        b.points[6].count = 1
        b.points[6].color = Color.WHITE
        # Need all-in-home for bear-off to be legal in the engine, but
        # _delta_pip just computes Δpip — it doesn't validate legality.
        # Use a simpler board: everyone in home.
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[6].count = 15
        b.points[6].color = Color.WHITE
        assert _delta_pip(b, Color.WHITE, [Move(6, 0, True)]) == 6


class TestIsDeadZoneMove:
    def test_move_into_opponents_home_in_race_is_dead(self):
        # Black about to enter pt1..6 while white is fully collected in
        # pt1..6 — the block there blocks nothing.
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[1].count = 15
        b.points[1].color = Color.WHITE
        # Black still has checkers on pt12 — give them a reachable move.
        assert _is_dead_zone_move(
            b, Color.BLACK, [Move(12, 7, False), Move(7, 5, False)]
        )

    def test_same_move_in_contact_is_not_dead(self):
        # Starting position — white hasn't run yet, blocks are meaningful.
        assert not _is_dead_zone_move(
            Board(), Color.BLACK, [Move(12, 7, False), Move(7, 5, False)]
        )

    def test_normal_advance_not_flagged(self):
        # 24→21 doesn't land in pt13..18 — never dead.
        assert not _is_dead_zone_move(
            Board(), Color.WHITE, [Move(24, 21, False)]
        )


class TestPromptShowsNewMetrics:
    def test_prompt_shows_phase_label(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        # Phase emitted in-line with the position block.
        assert "фаза: контакт" in low

    def test_prompt_shows_opponent_before_home_counter(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        # "Шашек соперника ещё не в своём доме: 15" at the start.
        assert "шашек соперника" in p.lower()
        assert "15" in p

    def test_prompt_shows_delta_pip_per_candidate(self):
        seqs = [[Move(24, 21, False)],
                [Move(24, 19, False), Move(24, 23, False)]]
        p = build_prompt(Board(), Color.WHITE, (3, 5), seqs)
        # First candidate: -3 pip; second: -5 + -1 = -6 (6+2=8 dice, 5+1=6).
        # We render Δpip as "(Δpip=−N)" next to each candidate.
        assert "Δpip=−3" in p or "Δpip=-3" in p

    def test_prompt_marks_dead_zone_candidate(self):
        # Build a race-phase board for black where white is fully
        # collected in pt1..6.
        b = Board()
        b.points[24].count = 0
        b.points[24].color = None
        b.points[1].count = 15
        b.points[1].color = Color.WHITE
        # One normal candidate (advance in own side), one dead-zone one
        # (black entering pt1..6).
        live = [Move(12, 7, False)]
        dead = [Move(12, 7, False), Move(7, 5, False)]
        p = build_prompt(b, Color.BLACK, (5, 2), [live, dead])
        assert "мёртвая зона" in p.lower()

    def test_prompt_refines_block_definition(self):
        p = build_prompt(Board(), Color.WHITE, (3, 5),
                         [[Move(24, 21, False)]])
        low = p.lower()
        # Must clarify: a "block" needs ≥2 checkers on each of ≥2
        # consecutive points — a lone checker on a point is not a block.
        assert "одиночка" in low or "одиночная шашка" in low
        assert "2 шашек" in low or "≥2" in low or ">=2" in low


class TestPipCount:
    def test_start_position_is_360_each(self):
        # 15 checkers × 24 pips-to-off at the head = 360 for each side.
        b = Board()
        assert _pip_count(b, Color.WHITE) == 360
        assert _pip_count(b, Color.BLACK) == 360

    def test_white_pip_decreases_when_white_advances(self):
        b = Board()
        # Move one white checker from pt24 (step 0) to pt21 (step 3).
        # Pip cost drops from 24 to 21, so total drops by 3.
        b.points[24].count = 14
        b.points[21].count = 1
        b.points[21].color = Color.WHITE
        assert _pip_count(b, Color.WHITE) == 360 - 3

    def test_borne_off_checkers_do_not_contribute(self):
        # A checker 1 pip away from bearing off costs 1 pip. Once off, 0.
        b = Board()
        b.points[24].count = 14
        # Move a checker all the way to pt1 (step 23, 1 pip from off).
        b.points[1].count = 1
        b.points[1].color = Color.WHITE
        assert _pip_count(b, Color.WHITE) == 14 * 24 + 1


class TestDescribeSequence:
    def test_empty_sequence_is_skip(self):
        assert _describe_sequence([]) == "(skip)"

    def test_single_move_renders_with_arrow(self):
        assert _describe_sequence([Move(24, 21, False)]) == "24→21"

    def test_single_checker_chained_moves_render_as_one_arrow_chain(self):
        # The previous `12/6 6/4` rendering made the LLM think these were
        # two different checkers — one landing on pt6, a second coming
        # from pt6. Chaining with arrows makes it obvious it's one
        # checker passing through pt6 without stopping.
        seq = [Move(12, 6, False), Move(6, 4, False)]
        assert _describe_sequence(seq) == "12→6→4"

    def test_different_checkers_separated_by_comma(self):
        seq = [Move(24, 23, False), Move(12, 10, False)]
        assert _describe_sequence(seq) == "24→23, 12→10"

    def test_mixed_chain_and_separate_checker(self):
        # First two moves chain (one checker 12→6→4), third move is a
        # different checker (24→21).
        seq = [Move(12, 6, False), Move(6, 4, False),
               Move(24, 21, False)]
        assert _describe_sequence(seq) == "12→6→4, 24→21"

    def test_bear_off_renders_as_off(self):
        assert _describe_sequence([Move(6, 0, True)]) == "6→off"

    def test_chain_into_bear_off(self):
        seq = [Move(9, 4, False), Move(4, 0, True)]
        assert _describe_sequence(seq) == "9→4→off"


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
