"""
Point d'entree simplifie pour lancer l'interface graphique Spatial.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed from app/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gui import main


if __name__ == "__main__":
    main()
