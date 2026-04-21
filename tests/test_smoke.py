import subprocess
import sys


def test_import_chain():
    # All engine/notation/model modules import cleanly together.
    from engine.board import Board
    from engine.moves import generate_move_sequences  # noqa: F401
    from engine.game import Game  # noqa: F401
    from notation.writer import format_game  # noqa: F401
    from notation.parser import parse_game  # noqa: F401
    from notation.replay import Replay  # noqa: F401
    from models.base import BaseModel  # noqa: F401
    from opponents.local_model import RandomLocalModel  # noqa: F401
    from opponents.openrouter import (  # noqa: F401
        OpenRouterModel, build_prompt, parse_reply,
    )
    assert Board()  # constructs


def test_main_imports(tmp_path, monkeypatch):
    # main.py must be importable without launching pygame window.
    monkeypatch.setenv("NARDGAME_HEADLESS", "1")
    proc = subprocess.run(
        [sys.executable, "-c", "import main; assert hasattr(main, 'run')"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
