import os

import pygame

from opponents.openrouter import OpenRouterModel
from ui.screens import GameScreen, MenuScreen, ReplayScreen


class App:
    def __init__(self):
        self.screen = None
        self.active = None

    def make_openrouter(self):
        key = os.environ.get("OPENROUTER_API_KEY", "")
        model = os.environ.get("OPENROUTER_MODEL")  # override default if set
        kwargs = {"api_key": key}
        if model:
            kwargs["model"] = model
        return OpenRouterModel(**kwargs)

    def start_game(self, white, black):
        self.active = GameScreen(self, white, black)

    def open_replay_prompt(self):
        path = os.environ.get("NARDGAME_REPLAY_PATH")
        if not path:
            if os.path.isdir("games"):
                files = sorted(
                    (os.path.join("games", f)
                     for f in os.listdir("games")
                     if f.endswith(".narde")),
                    key=os.path.getmtime, reverse=True,
                )
                if files:
                    path = files[0]
        if path and os.path.exists(path):
            self.active = ReplayScreen(self, path)

    def goto_menu(self):
        self.active = MenuScreen(self)


def run():
    headless = os.environ.get("NARDGAME_HEADLESS") == "1"
    if headless:
        # Import-only smoke: just construct the app, don't open a window.
        app = App()
        return app
    pygame.init()
    app = App()
    app.screen = pygame.display.set_mode((1360, 960))
    pygame.display.set_caption("Long Backgammon")
    app.active = MenuScreen(app)
    clock = pygame.time.Clock()
    running = True
    while running:
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                app.active.handle(event)
        app.active.tick(now)
        app.active.draw(app.screen)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    run()
