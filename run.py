"""
Convenience entrypoint to launch the Spatial GUI from the repository root.
Equivalent to running `python app/run.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.gui import main


if __name__ == "__main__":
    main()
