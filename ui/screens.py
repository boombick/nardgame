import os
import random
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import pygame

from engine.board import Board, Color
from engine.game import Game
from engine.moves import Move, apply_single
from notation.parser import parse_game
from notation.replay import Replay
from notation.writer import save_game
from models.base import BaseModel
from opponents.local_model import RandomLocalModel
from ui.input import InputState, hit_test
from ui.layout import (
    BoardLayout, DICE_SIZE, checker_positions, dice_slot_count,
    dice_slot_offsets, dice_slot_values,
)
from ui.renderer import (
    BLACK_CHECKER, Renderer, TARGET_DOT, TEXT, WHITE_CHECKER,
)


FPS = 30
DICE_ANIM_MS = 800       # total dice-spin duration
DICE_FLIP_MS = 60        # how often the spinning dice flash a new face
MOVE_ANIM_MS = 350       # time for one checker slide
MOVE_GAP_MS = 80         # small pause between chained moves
BOT_PAUSE_MS = 150       # render-a-frame delay between dice settle and bot call

# Internal state machine
_START = "start"   # initial banner: announce starter, wait for "Старт" click
_IDLE = "idle"
_ROLLING = "rolling"
_BOT_PAUSE = "bot_pause"  # dice settled; waiting a frame before bot plays
_HUMAN = "human"
_MOVING = "moving"
_OVER = "over"

_COLOR_RU = {Color.WHITE: "Белые", Color.BLACK: "Чёрные"}


def _fmt_move(m: Move) -> str:
    dst = "off" if m.is_bear_off else str(m.to_point)
    return f"{m.from_point}/{dst}"


def _fmt_seq(seq: List[Move]) -> str:
    return " ".join(_fmt_move(m) for m in seq) if seq else "--"


def _lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _ease_out(t: float) -> float:
    """Standard ease-out-cubic so the checker decelerates as it lands."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _draw_text(screen, font, text, pos, color=(20, 20, 20)):
    screen.blit(font.render(text, True, color), pos)


class Button:
    def __init__(self, rect, label, on_click):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click

    def draw(self, screen, font):
        pygame.draw.rect(screen, (220, 220, 220), self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        lbl = font.render(self.label, True, (0, 0, 0))
        screen.blit(lbl, (self.rect.centerx - lbl.get_width() // 2,
                          self.rect.centery - lbl.get_height() // 2))

    def handle(self, event):
        if (event.type == pygame.MOUSEBUTTONDOWN
                and self.rect.collidepoint(event.pos)):
            self.on_click()


class MenuScreen:
    def __init__(self, app):
        self.app = app
        self.font = pygame.font.SysFont("sans", 22)
        self.buttons = [
            Button((540, 200, 200, 40), "Human vs Human",
                   lambda: app.start_game(white=None, black=None)),
            Button((540, 260, 200, 40), "Human vs Local",
                   lambda: app.start_game(white=None, black=RandomLocalModel())),
            Button((540, 320, 200, 40), "Human vs OpenRouter",
                   lambda: app.start_game(white=None, black=app.make_openrouter())),
            Button((540, 380, 200, 40), "Local vs OpenRouter",
                   lambda: app.start_game(white=RandomLocalModel(),
                                          black=app.make_openrouter())),
            Button((540, 440, 200, 40), "Watch replay",
                   lambda: app.open_replay_prompt()),
        ]

    def tick(self, now_ms):
        pass

    def handle(self, event):
        for b in self.buttons:
            b.handle(event)

    def draw(self, screen):
        screen.fill((240, 230, 210))
        _draw_text(screen, self.font, "Long Backgammon (nardy)", (520, 120))
        for b in self.buttons:
            b.draw(screen, self.font)


@dataclass
class _MoveAnim:
    """One in-flight checker slide."""
    player: Color
    move: Move
    start_xy: Tuple[int, int]
    end_xy: Tuple[int, int]
    progress: float = 0.0  # 0..1


class GameScreen:
    def __init__(self, app, white_model: Optional[BaseModel],
                 black_model: Optional[BaseModel]):
        self.app = app
        self.layout = BoardLayout()
        self.renderer = Renderer(app.screen, self.layout)
        self.font = pygame.font.SysFont("sans", 18)
        self.dice_font = pygame.font.SysFont("sans", 36, bold=True)
        self.banner_font = pygame.font.SysFont("sans", 48, bold=True)
        self.game = Game(
            white_name="Human" if white_model is None
            else white_model.__class__.__name__,
            black_name="Human" if black_model is None
            else black_model.__class__.__name__,
        )
        self.game.determine_starter()
        print(f"[GAME] New game. Starter: {self.game.current_player.value} "
              f"(W={self.game.white_name}, B={self.game.black_name})")
        self.models = {Color.WHITE: white_model, Color.BLACK: black_model}
        # Keep the original factories so "Новая игра" can restart with the
        # same matchup. Re-using the instances is fine: the models' only
        # mutable state (last_reason, RNG) is irrelevant across games.
        self._white_model = white_model
        self._black_model = black_model
        self.input_state: Optional[InputState] = None
        self.status = ""
        # Right margin layout: black dice top, white dice bottom, roll+label
        # in the middle. Start button sits over the board (shown only during
        # the _START banner). The game-over buttons sit in the same central
        # band as the start button, side by side.
        self.roll_button = Button((1168, 280, 96, 40), "Roll", self._on_roll)
        self.start_button = Button((580, 400, 120, 50), "Старт",
                                   self._on_start)
        self.again_button = Button((500, 460, 130, 50), "Новая игра",
                                   self._on_again)
        self.menu_button = Button((650, 460, 130, 50), "В меню",
                                  self._on_menu)

        # Board used exclusively for rendering. Lags the real `game.board`
        # during move animations; they converge when all anims complete.
        self.display_board: Board = self.game.board.clone()

        # Dice state — one slot per player, persistent across turns so the
        # opponent's last roll stays visible while you're playing.
        self._flash_rng = random.SystemRandom()
        self._dice_display: Dict[Color, Optional[Tuple[int, int]]] = {
            Color.WHITE: None, Color.BLACK: None,
        }
        self._dice_offsets: Dict[Color, List[Tuple[int, int]]] = {
            Color.WHITE: [(0, 0), (0, 0)], Color.BLACK: [(0, 0), (0, 0)],
        }
        self._rolling_for: Optional[Color] = None
        self._dice_anim_start = 0
        self._dice_last_flip = 0

        # Move animations
        self.anim_player: Optional[Color] = None
        self.anim_queue: List[Move] = []
        self.anim_current: Optional[_MoveAnim] = None
        self._anim_last_tick = 0
        self._anim_gap_until = 0
        # Full sequence to commit via game.play() once the current anim
        # finishes. None means this anim is just playing back a partial
        # human click — no turn advance, just visual catch-up.
        self._pending_commit: Optional[List[Move]] = None

        # Bot-turn deferral: (color, sequences, no_moves). Stored between
        # _finish_rolling and _run_bot_turn so a frame renders with the
        # freshly-settled dice before any HTTP-blocking choose_move call.
        self._pending_bot: Optional[
            Tuple[Color, List[List[Move]], bool]] = None
        self._bot_pause_until: int = 0

        # Last bot message surfaced below the board: "Чёрные: 24/20 24/22 —
        # заблокировать голову противника".
        self.last_bot_msg: str = ""

        # Wait for the user to acknowledge who starts before any roll.
        self.state = _START

    # ------------------------------------------------------------ roll logic

    def _on_roll(self):
        if self.state != _IDLE:
            return
        current = self.game.current_player
        if self.models[current] is not None:
            return  # Roll button is for humans only
        self._begin_roll()

    def _on_start(self):
        if self.state != _START:
            return
        print(f"[GAME] Start acknowledged. "
              f"{self.game.current_player.value} to move.")
        self.state = _IDLE
        self._maybe_begin_auto_roll()

    def _on_again(self):
        if self.state != _OVER:
            return
        print("[GAME] Restart requested — new match with same opponents")
        self.app.start_game(self._white_model, self._black_model)

    def _on_menu(self):
        if self.state != _OVER:
            return
        print("[GAME] Return-to-menu requested")
        self.app.goto_menu()

    def _maybe_begin_auto_roll(self):
        if self.state != _IDLE or self.game.is_over():
            return
        current = self.game.current_player
        if current and self.models[current] is not None:
            self._begin_roll()

    def _begin_roll(self):
        self.game.roll()
        current = self.game.current_player
        print(f"[{current.value}] Roll {self.game.dice}")
        self.state = _ROLLING
        self._rolling_for = current
        now = pygame.time.get_ticks()
        self._dice_anim_start = now
        self._dice_last_flip = now
        # Start with random faces; settle on real dice when anim expires.
        self._dice_display[current] = (self._flash_rng.randint(1, 6),
                                       self._flash_rng.randint(1, 6))
        self._dice_offsets[current] = [self._rand_offset(),
                                       self._rand_offset()]

    def _rand_offset(self):
        return (self._flash_rng.randint(-3, 3),
                self._flash_rng.randint(-3, 3))

    def _finish_rolling(self):
        current = self.game.current_player
        self._dice_display[current] = self.game.dice
        self._dice_offsets[current] = [(0, 0), (0, 0)]
        self._rolling_for = None
        seqs = self.game.legal_sequences()
        model = self.models[current]
        no_moves = (not seqs) or seqs == [[]]
        if model is None:
            # Human: hand control over immediately.
            if no_moves:
                print(f"[{current.value}] No legal moves, auto-passing")
                self._commit_sequence(current, [])
                return
            self.input_state = InputState(color=current, sequences=seqs)
            self.state = _HUMAN
            print(f"[{current.value}] Waiting for click. "
                  f"Movable points: "
                  f"{sorted(self.input_state.legal_from_points)}")
            return
        # Bot: defer the choose_move call by one frame. model.choose_move may
        # block on a network round-trip; without the pause the user would see
        # the last random dice flip frozen on-screen and then the move arrive,
        # making it look like the dice and the move don't match.
        self._pending_bot = (current, seqs, no_moves)
        self._bot_pause_until = pygame.time.get_ticks() + BOT_PAUSE_MS
        self.state = _BOT_PAUSE

    def _run_bot_turn(self):
        assert self._pending_bot is not None
        current, seqs, no_moves = self._pending_bot
        self._pending_bot = None
        model = self.models[current]
        if no_moves:
            print(f"[{current.value} BOT] No legal moves, passing")
            self.last_bot_msg = f"{_COLOR_RU[current]}: пропуск хода"
            self._commit_sequence(current, [])
            return
        chosen = model.choose_move(
            self.game.board, current, self.game.dice, seqs,
        )
        evaluation = getattr(model, "last_evaluation", "")
        reason = getattr(model, "last_reason", "")
        print(f"[{current.value} BOT] Plays {_fmt_seq(chosen)}")
        # Compose the panel message: "<player>: <moves> — оценка: X — почему: Y".
        # Either field may be empty (local model has no evaluation; the OpenRouter
        # fallback reports only a reason), and the panel word-wraps automatically.
        parts = [f"{_COLOR_RU[current]}: {_fmt_seq(chosen)}"]
        if evaluation:
            parts.append(f"оценка: {evaluation}")
        if reason:
            parts.append(f"почему: {reason}")
        self.last_bot_msg = " — ".join(parts)
        self._commit_sequence(current, chosen)

    # ------------------------------------------------------- sequence commit

    def _commit_sequence(self, player: Color, seq: List[Move]):
        """Commit a complete turn: animate it and advance the game."""
        self._begin_anim(player, seq, commit_full=seq)

    def _begin_anim(self, player: Color, moves: List[Move],
                    commit_full: Optional[List[Move]]):
        """Queue `moves` for animation on `display_board`. When `commit_full`
        is provided it's the full turn sequence applied to `game` now; when
        None, this is a partial human click — the input_state stays alive
        and the UI returns to _HUMAN once the anim finishes."""
        self.anim_player = player
        self.anim_queue = list(moves)
        self.anim_current = None
        self._anim_last_tick = pygame.time.get_ticks()
        self._anim_gap_until = 0
        self._pending_commit = commit_full
        if commit_full is not None:
            # Real game state jumps to post-move immediately; display_board
            # catches up one move at a time via apply_single.
            self.input_state = None
            self.game.play(list(commit_full))
        self.state = _MOVING
        if not self.anim_queue:
            # Nothing to animate (skip turn, or partial click with zero-
            # length delta — shouldn't happen but handle it).
            self._finish_moving()

    def _start_next_anim(self, now_ms: int):
        if not self.anim_queue:
            return
        m = self.anim_queue.pop(0)
        start_xy = self._top_checker_xy(self.display_board, m.from_point)
        if m.is_bear_off:
            end_xy = self._bear_off_xy(self.anim_player)
        else:
            sim = self.display_board.clone()
            apply_single(sim, self.anim_player, m)
            end_xy = self._top_checker_xy(sim, m.to_point)
        self.anim_current = _MoveAnim(
            player=self.anim_player, move=m,
            start_xy=start_xy, end_xy=end_xy, progress=0.0,
        )
        self._anim_last_tick = now_ms

    def _finish_current_anim(self):
        # Commit the finished animation to the display board.
        apply_single(self.display_board, self.anim_current.player,
                     self.anim_current.move)
        self.anim_current = None
        self._anim_gap_until = pygame.time.get_ticks() + MOVE_GAP_MS

    def _finish_moving(self):
        self.anim_player = None
        commit = self._pending_commit
        self._pending_commit = None
        if commit is None:
            # Partial human move: animation caught display_board up, but
            # the turn isn't over. Return control to the player.
            self.state = _HUMAN
            return
        # Dice stay displayed on the player's side until they roll again —
        # opponent's last roll remains visible during your turn.
        if self.game.is_over():
            self._on_game_over()
            self.state = _OVER
            return
        self.state = _IDLE
        # If next player is a bot, auto-advance.
        self._maybe_begin_auto_roll()

    def _on_game_over(self):
        winner = self.game.winner()
        print(f"[GAME OVER] {winner.value} wins, "
              f"score (W,B) = {self.game.score()}")
        self._save_game()

    def _save_game(self):
        os.makedirs("games", exist_ok=True)
        path = os.path.join(
            "games",
            f"game_{date.today().isoformat()}_"
            f"{len(os.listdir('games')) + 1:03d}.narde",
        )
        save_game(
            path=path,
            history=self.game.history,
            event="Local Game",
            date=date.today().isoformat(),
            white=self.game.white_name,
            black=self.game.black_name,
            score=self.game.score(),
        )
        self.status = f"Saved {path}"
        print(f"[GAME] {self.status}")

    # -------------------------------------------------------- layout helpers

    def _top_checker_xy(self, board: Board,
                        point: int) -> Tuple[int, int]:
        """Centre of the topmost checker at `point` on `board`. If the stack
        is empty, return the triangle centre so animations land on the
        visually correct spot."""
        ps = board.points[point]
        count = ps.count if ps.color is not None else 0
        if count == 0:
            rect = self.layout.point_rect(point)
            return (rect.x + rect.w // 2,
                    rect.y + rect.h // 2)
        positions = checker_positions(
            point, count, self.layout, top_row=(point >= 13),
        )
        return positions[-1]

    def _bear_off_rect(self, color: Color) -> pygame.Rect:
        """Clickable tray where borne-off checkers pile up. Placed adjacent
        to each player's exit edge: white exits past pt 1 (bottom-right)
        so the tray sits in the right margin above white's dice; black
        exits past pt 13 (top-left) so the tray sits in the left margin
        above the top row, away from black's bottom-left dice."""
        L = self.layout
        w, h = 100, 140
        if color == Color.WHITE:
            x = L.board_left + L.board_width + 14
            y = L.board_top + L.row_height + 20
        else:
            x = 14
            y = L.board_top + 20
        return pygame.Rect(x, y, w, h)

    def _bear_off_xy(self, color: Color) -> Tuple[int, int]:
        """Centre of the bear-off tray — used as the animation landing
        point so borne-off checkers fly into the same place they pile up."""
        rect = self._bear_off_rect(color)
        return (rect.x + rect.w // 2, rect.y + rect.h // 2)

    def _bear_off_hit(self, pos: Tuple[int, int]) -> bool:
        """True when `pos` falls inside the current human player's tray —
        the only tray a click can act on."""
        if not self.input_state:
            return False
        rect = self._bear_off_rect(self.input_state.color)
        return rect.collidepoint(pos)

    def _checker_radius(self) -> int:
        return min(self.layout.point_width // 2 - 2, 22)

    # ---------------------------------------------------------- per-frame

    def tick(self, now_ms: int):
        if self.state == _ROLLING:
            c = self._rolling_for
            if c is None:
                return
            if now_ms - self._dice_last_flip >= DICE_FLIP_MS:
                self._dice_display[c] = (self._flash_rng.randint(1, 6),
                                         self._flash_rng.randint(1, 6))
                self._dice_offsets[c] = [self._rand_offset(),
                                         self._rand_offset()]
                self._dice_last_flip = now_ms
            if now_ms - self._dice_anim_start >= DICE_ANIM_MS:
                self._finish_rolling()
        elif self.state == _BOT_PAUSE:
            if now_ms >= self._bot_pause_until:
                self._run_bot_turn()
        elif self.state == _MOVING:
            if self.anim_current is None:
                if now_ms < self._anim_gap_until:
                    return
                if not self.anim_queue:
                    self._finish_moving()
                    return
                self._start_next_anim(now_ms)
            else:
                dt = now_ms - self._anim_last_tick
                self._anim_last_tick = now_ms
                self.anim_current.progress += dt / MOVE_ANIM_MS
                if self.anim_current.progress >= 1.0:
                    self._finish_current_anim()

    # ----------------------------------------------------------- input

    def handle(self, event):
        if self.state == _START:
            self.start_button.handle(event)
            return
        if self.state == _OVER:
            self.again_button.handle(event)
            self.menu_button.handle(event)
            return
        self.roll_button.handle(event)
        if self.state != _HUMAN:
            return
        if not self.input_state:
            return
        if event.type != pygame.MOUSEBUTTONDOWN:
            return
        pt = hit_test(event.pos, self.layout)
        if pt is None:
            # Fall back to the bear-off tray: clicking the current player's
            # tray acts like clicking destination 0 (bear-off). Clicking any
            # other empty space is ignored.
            if self._bear_off_hit(event.pos):
                pt = 0
            else:
                return
        before_selected = self.input_state.selected_from
        before_len = len(self.input_state.played_so_far)
        sequence = self.input_state.click_point(pt, self.game.board)
        after_len = len(self.input_state.played_so_far)
        new_moves = list(self.input_state.played_so_far[before_len:after_len])
        if sequence is not None:
            print(f"[{self.game.current_player.value}] "
                  f"Plays {_fmt_seq(sequence)}")
            # Animate only the slice that wasn't already shown on
            # display_board (earlier partial clicks advanced it already).
            self._begin_anim(self.game.current_player, new_moves,
                             commit_full=sequence)
            return
        if after_len > before_len:
            movable = sorted(self.input_state.legal_from_points)
            print(f"  Partial: played {_fmt_seq(new_moves)}. "
                  f"Movable next: {movable}")
            self._begin_anim(self.game.current_player, new_moves,
                             commit_full=None)
            return
        new_selected = self.input_state.selected_from
        if before_selected is None and new_selected == pt:
            targets = sorted(self.input_state.highlight_targets)
            print(f"  Selected pt {pt}. Targets: {targets}")
        elif before_selected is not None and new_selected is None:
            print(f"  Click on pt {pt} not a target of pt "
                  f"{before_selected}; selection cleared")
        elif before_selected is None and new_selected is None:
            print(f"  Pt {pt} has no legal move")

    # ----------------------------------------------------------- rendering

    def draw(self, screen):
        if self.state == _MOVING and self.anim_current is not None:
            self._draw_moving(screen)
        else:
            self._draw_static(screen)
        # Roll button is hidden on the start and end banners — both have
        # their own action buttons so leaving Roll around would clutter.
        if self.state not in (_START, _OVER):
            self.roll_button.draw(screen, self.font)
        # "Ход: …" is meaningful only while the game is running. After the
        # last bear-off `game.current_player` points at the *next* side,
        # which would confusingly still say "Ход: чёрные" on the winning
        # frame. Suppress it while over or on the start banner.
        if (self.game.current_player and self.state not in (_START, _OVER)):
            turn_lbl = f"Ход: {_COLOR_RU[self.game.current_player]}"
            if self.state == _MOVING and self.anim_player is not None:
                turn_lbl = f"Ход: {_COLOR_RU[self.anim_player]} (анимация…)"
            _draw_text(screen, self.font, turn_lbl, (1168, 350))
        self._draw_bot_msg(screen)
        if self.status and self.state != _OVER:
            _draw_text(screen, self.font, self.status, (20, 10))
        if self.state == _START:
            self._draw_start_banner(screen)
        elif self.state == _OVER:
            self._draw_over_banner(screen)

    def _wrap_text(self, text: str, font: pygame.font.Font,
                   max_w: int) -> List[str]:
        """Greedy word-wrap that keeps each line under `max_w` pixels. Long
        single tokens are kept on their own line rather than hard-split —
        backgammon notation like '24/21' is meant to stay intact."""
        lines: List[str] = []
        cur = ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if font.size(trial)[0] <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def _draw_bot_msg(self, screen):
        # A clearly-bordered panel below the board so the bot's latest move
        # and reason have their own visual space and aren't mistaken for
        # board chrome. Two lines of word-wrapped text fit inside.
        L = self.layout
        panel_x = 12
        panel_y = L.board_top + L.board_height + 16
        panel_w = L.screen_w - 24
        panel_h = 56
        pygame.draw.rect(screen, (250, 245, 225),
                         (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(screen, (80, 60, 40),
                         (panel_x, panel_y, panel_w, panel_h), 2)
        if not self.last_bot_msg:
            return
        lines = self._wrap_text(self.last_bot_msg, self.font, panel_w - 20)
        for i, line in enumerate(lines[:2]):
            _draw_text(screen, self.font, line,
                       (panel_x + 10, panel_y + 6 + i * 22))

    def _draw_bear_off_trays(self, screen, highlight_zero: bool):
        """Draw both bear-off trays with the current pile of borne-off
        checkers. When `highlight_zero` is set and the current human has
        0 among their targets, ring the current player's tray so the
        player knows where to click to bear off."""
        L = self.layout
        for color in (Color.WHITE, Color.BLACK):
            rect = self._bear_off_rect(color)
            pygame.draw.rect(screen, (210, 190, 140), rect)
            pygame.draw.rect(screen, (60, 40, 20), rect, 2)
            borne = self.display_board.borne_off[color]
            fill = WHITE_CHECKER if color == Color.WHITE else BLACK_CHECKER
            border = ((30, 30, 30) if color == Color.WHITE
                      else (220, 220, 220))
            r = 9
            # Pile into a 5×3 grid at the top of the tray.
            for i in range(min(borne, 15)):
                cx = rect.x + 14 + (i % 5) * 18
                cy = rect.y + 14 + (i // 5) * 20
                pygame.draw.circle(screen, fill, (cx, cy), r)
                pygame.draw.circle(screen, border, (cx, cy), r, 1)
            lbl = self.font.render(f"{borne}/15 off", True, (30, 20, 10))
            screen.blit(lbl, (rect.x + rect.w // 2 - lbl.get_width() // 2,
                              rect.y + rect.h - 22))
        # L is intentionally unused below, kept for future symmetry.
        _ = L
        if (highlight_zero and self.state == _HUMAN and self.input_state
                and 0 in self.input_state.highlight_targets):
            rect = self._bear_off_rect(self.input_state.color)
            pygame.draw.rect(screen, TARGET_DOT, rect, 4)

    def _draw_start_banner(self, screen):
        L = self.layout
        overlay = pygame.Surface((L.screen_w, L.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        title = f"{_COLOR_RU[self.game.current_player]} начинают!"
        surf = self.banner_font.render(title, True, (245, 240, 210))
        screen.blit(surf, (L.screen_w // 2 - surf.get_width() // 2, 280))
        self.start_button.draw(screen, self.font)

    def _draw_over_banner(self, screen):
        """Semi-transparent overlay with the winner, score type (Марс/Ойн),
        the path to the saved .narde, and two buttons — «Новая игра» keeps
        the same matchup, «В меню» returns to the menu screen."""
        L = self.layout
        overlay = pygame.Surface((L.screen_w, L.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        winner = self.game.winner()
        w, b = self.game.score()
        points = w if winner == Color.WHITE else b
        # 2 = марс (opponent borne off nothing), 1 = ойн.
        kind = "Марс" if points == 2 else "Ойн"
        title = f"{_COLOR_RU[winner]} выиграли!"
        subtitle = f"{kind} · счёт {w}:{b}"
        t_surf = self.banner_font.render(title, True, (245, 240, 210))
        s_surf = self.font.render(subtitle, True, (230, 220, 190))
        screen.blit(t_surf,
                    (L.screen_w // 2 - t_surf.get_width() // 2, 260))
        screen.blit(s_surf,
                    (L.screen_w // 2 - s_surf.get_width() // 2, 330))
        if self.status:
            p_surf = self.font.render(self.status, True, (200, 200, 185))
            screen.blit(p_surf,
                        (L.screen_w // 2 - p_surf.get_width() // 2, 370))
        self.again_button.draw(screen, self.font)
        self.menu_button.draw(screen, self.font)

    def _draw_static(self, screen):
        selected = (self.input_state.selected_from
                    if self.input_state else None)
        targets = (list(self.input_state.highlight_targets)
                   if self.input_state else [])
        targets_on_board = [t for t in targets if t != 0]
        self.renderer.draw(self.display_board, dice=None,
                           selected_point=selected,
                           highlight_targets=targets_on_board)
        self._draw_dice(screen)
        self._draw_bear_off_trays(screen, highlight_zero=(0 in targets))

    def _draw_moving(self, screen):
        # Show the display board with one fewer checker at the moving-from
        # point, then draw a lone checker at the interpolated position.
        clone = self.display_board.clone()
        m = self.anim_current.move
        clone.remove_one(m.from_point, self.anim_current.player)
        self.renderer.draw(clone, dice=None)
        self._draw_dice(screen)
        self._draw_bear_off_trays(screen, highlight_zero=False)
        t = _ease_out(self.anim_current.progress)
        sx, sy = self.anim_current.start_xy
        ex, ey = self.anim_current.end_xy
        x, y = _lerp(sx, ex, t), _lerp(sy, ey, t)
        fill = (WHITE_CHECKER if self.anim_current.player == Color.WHITE
                else BLACK_CHECKER)
        border = ((30, 30, 30) if self.anim_current.player == Color.WHITE
                  else (220, 220, 220))
        r = self._checker_radius()
        pygame.draw.circle(screen, fill, (x, y), r)
        pygame.draw.circle(screen, border, (x, y), r, 2)

    def _used_slots_for(self, color: Color) -> int:
        """How many dice slots have already been spent this turn by `color`.

        Each move consumes one slot (doubles get four slots total, others
        two). While a human is mid-turn we read `input_state.played_so_far`
        so the dice panel dims slots click-by-click. Bots apply the whole
        sequence atomically, so between rolls this is always 0 or full."""
        if color != self.game.current_player:
            return 0
        if self.input_state is not None:
            return len(self.input_state.played_so_far)
        return 0

    def _draw_dice(self, screen):
        """Draw each player's last roll next to their head. Black's head is
        pt 12 (bottom-left of the board), so black's dice live in the left
        margin near the bottom. White's head is pt 24 (top-right), but we
        keep white's dice on the right margin near the bottom to line them
        up horizontally — side + colour make the owner obvious. The side
        belonging to whoever's NOT on turn is dimmed so players never
        confuse the opponent's previous roll with their own current dice.

        For doubles we lay four slots out in a 2×2 grid; already-played
        slots get a darker overlay and a diagonal strike-through so the
        player can see at a glance which dice still have moves in them."""
        L = self.layout
        # Slot geometry: keep 64px squares; for doubles we now need two
        # columns, so pull black's anchor further left to keep the grid in
        # the 128px-wide margin.
        base_x_for = {Color.BLACK: 16,
                      Color.WHITE: L.board_left + L.board_width + 16}
        base_y = L.board_top + L.board_height - 160
        current = self.game.current_player
        for color in (Color.BLACK, Color.WHITE):
            dice = self._dice_display[color]
            if dice is None:
                continue
            base_x = base_x_for[color]
            rolling = (self.state == _ROLLING and self._rolling_for == color)
            # "On-turn" means the dice belong to the side whose move is
            # actively being resolved (rolling, thinking, clicking, or
            # animating). Otherwise we dim them to visually demote the
            # opponent's previous-turn dice.
            on_turn = (current == color) and not self.game.is_over()
            if color == Color.BLACK:
                bg_active = (60, 60, 60)
                bg_dim = (110, 110, 110)
                bg_rolling = (120, 90, 50)
                bg_used = (35, 35, 35)
                text_color = (240, 240, 240)
                text_used = (120, 120, 120)
                strike = (200, 80, 80)
            else:
                bg_active = (250, 250, 250)
                bg_dim = (210, 205, 190)
                bg_rolling = (255, 240, 200)
                bg_used = (170, 170, 160)
                text_color = (20, 20, 20)
                text_used = (140, 140, 140)
                strike = (200, 80, 80)
            if rolling:
                bg = bg_rolling
            elif on_turn:
                bg = bg_active
            else:
                bg = bg_dim
            # During a spin the engine hasn't published the real dice yet,
            # so we render whatever pair the flash RNG picked — always 2
            # slots, never 4, even if the spin lands on a double.
            if rolling:
                slot_values = list(dice)
                slot_offsets = [(0, 0), (0, 80)]
                used = 0
            else:
                slot_values = dice_slot_values(dice)
                slot_offsets = dice_slot_offsets(dice)
                used = self._used_slots_for(color)
            for i, d in enumerate(slot_values):
                ox, oy = slot_offsets[i]
                # Per-die jitter only applies while rolling and only to the
                # two live slots; the extra (3rd/4th) slots don't exist
                # during a spin so indexing stays safe.
                if rolling and i < len(self._dice_offsets[color]):
                    jx, jy = self._dice_offsets[color][i]
                else:
                    jx, jy = 0, 0
                x = base_x + ox + jx
                y = base_y + oy + jy
                is_used = i < used
                fill = bg_used if is_used else bg
                text = text_used if is_used else text_color
                pygame.draw.rect(screen, fill, (x, y, DICE_SIZE, DICE_SIZE))
                pygame.draw.rect(screen, (0, 0, 0),
                                 (x, y, DICE_SIZE, DICE_SIZE), 2)
                lbl = self.dice_font.render(str(d), True, text)
                screen.blit(lbl, (x + 32 - lbl.get_width() // 2,
                                  y + 32 - lbl.get_height() // 2))
                if is_used:
                    # Diagonal strike so "used" is readable without having
                    # to compare greyscale shades side-by-side.
                    pygame.draw.line(screen, strike, (x + 6, y + 6),
                                     (x + DICE_SIZE - 6, y + DICE_SIZE - 6), 3)


class ReplayScreen:
    def __init__(self, app, path: str):
        self.app = app
        self.layout = BoardLayout()
        self.renderer = Renderer(app.screen, self.layout)
        self.font = pygame.font.SysFont("sans", 18)
        with open(path, "r", encoding="utf-8") as f:
            self.parsed = parse_game(f.read())
        self.replay = Replay(self.parsed)
        self.autoplay = False
        self._last_tick = pygame.time.get_ticks()
        print(f"[REPLAY] Loaded {path} "
              f"({self.replay.total_steps()} half-moves)")
        self.buttons = [
            Button((20, 660, 80, 40), "<<",
                   lambda: self.replay.step_backward()),
            Button((110, 660, 80, 40), "Play/Pause",
                   lambda: self._toggle()),
            Button((200, 660, 80, 40), ">>",
                   lambda: self.replay.step_forward()),
            Button((290, 660, 80, 40), "Menu",
                   lambda: app.goto_menu()),
        ]

    def _toggle(self):
        self.autoplay = not self.autoplay
        print(f"[REPLAY] autoplay={'on' if self.autoplay else 'off'}")

    def tick(self, now_ms):
        if self.autoplay and now_ms - self._last_tick > 1000:
            self.replay.step_forward()
            self._last_tick = now_ms

    def handle(self, event):
        for b in self.buttons:
            b.handle(event)

    def draw(self, screen):
        self.renderer.draw(self.replay.board, dice=self.replay.current_dice)
        for b in self.buttons:
            b.draw(screen, self.font)
        _draw_text(
            screen, self.font,
            f"Step {self.replay.current_move_index + 1}/"
            f"{self.replay.total_steps()}",
            (400, 670),
        )
