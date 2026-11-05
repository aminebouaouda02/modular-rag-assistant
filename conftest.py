"""Make the project root importable so `pytest` works from any cwd."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
