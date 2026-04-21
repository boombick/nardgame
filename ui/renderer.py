import pygame

from engine.board import Board, Color
from ui.layout import BoardLayout, checker_positions


WOOD = (196, 164, 132)
DARK_POINT = (120, 80, 50)
LIGHT_POINT = (230, 200, 160)
BAR = (80, 50, 30)
WHITE_CHECKER = (240, 240, 240)
BLACK_CHECKER = (30, 30, 30)
TEXT = (20, 20, 20)
HIGHLIGHT = (255, 215, 0)


class Renderer:
    """Draws the board, checkers, and dice. The only module that imports
    pygame and does drawing — layout math lives in ui/layout.py."""

    def __init__(self, screen: pygame.Surface, layout: BoardLayout):
        self.screen = screen
        self.layout = layout
        self.font = pygame.font.SysFont("sans", 14)
        self.dice_font = pygame.font.SysFont("sans", 36, bold=True)

    def draw(self, board: Board, dice=None, selected_point: int = None,
             highlight_targets: list = None) -> None:
        self._draw_board()
        self._draw_checkers(board, selected_point or 0, highlight_targets or [])
        if dice is not None:
            self._draw_dice(dice)

    def _draw_board(self) -> None:
        L = self.layout
        self.screen.fill(WOOD)
        for pt in range(1, 25):
            rect = L.point_rect(pt)
            color = DARK_POINT if (pt % 2 == 0) else LIGHT_POINT
            top_row = pt >= 13
            if top_row:
                apex = (rect.x + rect.w // 2, rect.y + rect.h)
                base_l = (rect.x, rect.y)
                base_r = (rect.x + rect.w, rect.y)
            else:
                apex = (rect.x + rect.w // 2, rect.y)
                base_l = (rect.x, rect.y + rect.h)
                base_r = (rect.x + rect.w, rect.y + rect.h)
            pygame.draw.polygon(self.screen, color, [apex, base_l, base_r])
            label = self.font.render(str(pt), True, TEXT)
            lx = rect.x + rect.w // 2 - label.get_width() // 2
            ly = rect.y - 16 if top_row else rect.y + rect.h + 2
            self.screen.blit(label, (lx, ly))
        bar_x = L.board_left + L.quadrant_width
        pygame.draw.rect(self.screen, BAR,
                         (bar_x, L.board_top, L.bar_width, L.board_height))

    def _draw_checkers(self, board: Board, selected_point: int,
                       highlight_targets: list) -> None:
        L = self.layout
        for pt in range(1, 25):
            ps = board.points[pt]
            if ps.count == 0 or ps.color is None:
                continue
            positions = checker_positions(pt, ps.count, L, top_row=(pt >= 13))
            fill = WHITE_CHECKER if ps.color == Color.WHITE else BLACK_CHECKER
            border = (30, 30, 30) if ps.color == Color.WHITE else (220, 220, 220)
            radius = min(L.point_width // 2 - 2, 22)
            for (cx, cy) in positions:
                pygame.draw.circle(self.screen, fill, (cx, cy), radius)
                pygame.draw.circle(self.screen, border, (cx, cy), radius, 2)
            if pt == selected_point:
                rect = L.point_rect(pt)
                pygame.draw.rect(self.screen, HIGHLIGHT,
                                 (rect.x, rect.y, rect.w, rect.h), 3)
        for tpt in highlight_targets:
            rect = L.point_rect(tpt)
            pygame.draw.rect(self.screen, HIGHLIGHT,
                             (rect.x, rect.y, rect.w, rect.h), 3)

    def _draw_dice(self, dice) -> None:
        L = self.layout
        x = L.board_left + L.board_width + 16
        y = L.board_top + 16
        for i, d in enumerate(dice):
            pygame.draw.rect(self.screen, (245, 245, 245),
                             (x, y + i * 80, 64, 64))
            pygame.draw.rect(self.screen, (0, 0, 0),
                             (x, y + i * 80, 64, 64), 2)
            lbl = self.dice_font.render(str(d), True, TEXT)
            self.screen.blit(lbl, (x + 32 - lbl.get_width() // 2,
                                   y + i * 80 + 32 - lbl.get_height() // 2))
