"""End-to-end undo: partial click -> undo -> board restored.

These tests touch the full GameScreen so we can catch bugs in the
snapshot/restore wiring — pure-InputState tests in test_input.py cover
the reset() primitive, but not the display_board snapshot flow."""

import pygame
import pytest

from engine.board import Color
from engine.moves import Move


@pytest.fixture(scope="module", autouse=True)
def _pygame_display():
    # GameScreen constructs SysFonts at import, so we need a live pygame
    # display. One module-scoped init avoids paying it per test.
    pygame.init()
    pygame.display.set_mode((1280, 800))
    yield
    pygame.quit()


def _make_screen():
    # Defer import until pygame is up; otherwise font construction at
    # import-time fails before the fixture ran.
    import main
    from ui.screens import GameScreen
    app = main.App()
    app.screen = pygame.display.get_surface()
    screen = GameScreen(app, white_model=None, black_model=None)
    # Force white to be on move and already past the "Старт" banner.
    screen.game.current_player = Color.WHITE
    screen._on_start()
    return screen


def _prepare_human_turn(screen, dice):
    # Directly wire up a human turn with a fixed dice roll — bypasses the
    # roll animation.
    from ui.input import InputState
    screen.game.dice = dice
    seqs = screen.game.legal_sequences()
    screen.input_state = InputState(color=screen.game.current_player,
                                    sequences=seqs)
    screen._turn_snapshot = screen.display_board.clone()
    # State name imported lazily via module lookup to avoid a re-export.
    from ui import screens as screens_mod
    screen.state = screens_mod._HUMAN


def _point_stacks(board):
    return {pt: (board.points[pt].count, board.points[pt].color)
            for pt in range(1, 25)}


def test_undo_restores_board_after_partial_click():
    screen = _make_screen()
    _prepare_human_turn(screen, (3, 5))
    before = _point_stacks(screen.display_board)
    # Apply a single partial move by directly extending the state — the
    # exact click-through path is exercised in test_input.py.
    partial = Move(24, 21, False)
    from engine.moves import apply_single
    apply_single(screen.display_board, Color.WHITE, partial)
    screen.input_state.played_so_far.append(partial)
    assert screen._can_undo()
    assert _point_stacks(screen.display_board) != before
    screen._on_undo()
    assert not screen.input_state.played_so_far
    assert screen.input_state.selected_from is None
    assert _point_stacks(screen.display_board) == before


def test_undo_disabled_before_any_move():
    screen = _make_screen()
    _prepare_human_turn(screen, (3, 5))
    # Nothing played — undo button must not be offered.
    assert not screen._can_undo()


def test_undo_disabled_when_not_humans_turn():
    screen = _make_screen()
    # Bot state: no input_state, definitely no undo.
    screen.input_state = None
    screen._turn_snapshot = None
    assert not screen._can_undo()
