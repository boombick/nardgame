import os
from datetime import date
from typing import Optional

import pygame

from engine.board import Color
from engine.game import Game
from notation.parser import parse_game
from notation.replay import Replay
from notation.writer import save_game
from models.base import BaseModel
from opponents.local_model import RandomLocalModel
from ui.input import InputState, hit_test
from ui.layout import BoardLayout
from ui.renderer import Renderer


FPS = 30


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

    def handle(self, event):
        for b in self.buttons:
            b.handle(event)

    def draw(self, screen):
        screen.fill((240, 230, 210))
        _draw_text(screen, self.font, "Long Backgammon (nardy)", (520, 120))
        for b in self.buttons:
            b.draw(screen, self.font)


class GameScreen:
    def __init__(self, app, white_model: Optional[BaseModel],
                 black_model: Optional[BaseModel]):
        self.app = app
        self.layout = BoardLayout()
        self.renderer = Renderer(app.screen, self.layout)
        self.font = pygame.font.SysFont("sans", 18)
        self.game = Game(
            white_name="Human" if white_model is None
            else white_model.__class__.__name__,
            black_name="Human" if black_model is None
            else black_model.__class__.__name__,
        )
        self.game.determine_starter()
        self.models = {Color.WHITE: white_model, Color.BLACK: black_model}
        self.input_state: Optional[InputState] = None
        self.status = ""
        self.roll_button = Button((1090, 560, 160, 40), "Roll", self._on_roll)

    def _on_roll(self):
        if self.game.dice is not None or self.game.is_over():
            return
        self.game.roll()
        seqs = self.game.legal_sequences()
        current = self.game.current_player
        model = self.models[current]
        if model is not None:
            chosen = model.choose_move(self.game.board, current,
                                       self.game.dice, seqs)
            self.game.play(chosen)
            self._maybe_save()
        else:
            self.input_state = InputState(color=current, sequences=seqs)

    def _maybe_save(self):
        if self.game.is_over():
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

    def handle(self, event):
        self.roll_button.handle(event)
        if self.input_state and event.type == pygame.MOUSEBUTTONDOWN:
            pt = hit_test(event.pos, self.layout)
            if pt is not None:
                sequence = self.input_state.click_point(pt, self.game.board)
                if sequence is not None:
                    self.game.play(sequence)
                    self.input_state = None
                    self._maybe_save()

    def draw(self, screen):
        selected = (self.input_state.selected_from
                    if self.input_state else None)
        targets = (list(self.input_state.highlight_targets)
                   if self.input_state else [])
        targets_on_board = [t for t in targets if t != 0]
        self.renderer.draw(self.game.board, dice=self.game.dice,
                           selected_point=selected,
                           highlight_targets=targets_on_board)
        self.roll_button.draw(screen, self.font)
        turn_lbl = (f"Turn: {self.game.current_player.value}"
                    if self.game.current_player else "")
        _draw_text(screen, self.font, turn_lbl, (1090, 520))
        if self.status:
            _draw_text(screen, self.font, self.status, (20, 690))
        if self.game.is_over():
            _draw_text(
                screen, self.font,
                f"Game over: {self.game.winner().value} "
                f"wins {self.game.score()}",
                (480, 680),
            )


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

    def handle(self, event):
        for b in self.buttons:
            b.handle(event)

    def draw(self, screen):
        if self.autoplay:
            now = pygame.time.get_ticks()
            if now - self._last_tick > 1000:
                self.replay.step_forward()
                self._last_tick = now
        self.renderer.draw(self.replay.board, dice=self.replay.current_dice)
        for b in self.buttons:
            b.draw(screen, self.font)
        _draw_text(
            screen, self.font,
            f"Step {self.replay.current_move_index + 1}/"
            f"{self.replay.total_steps()}",
            (400, 670),
        )
