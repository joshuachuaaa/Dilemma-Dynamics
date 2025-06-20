# ────────────────────────────────────────────────────────────
# File   : Utils/plot_helpers.py
# Author : Joshua Chua Han Wei – 32781555
# Purpose: Re-usable helpers for saving Matplotlib figures
# ────────────────────────────────────────────────────────────
from pathlib import Path
import matplotlib.pyplot as plt

# Project-root = one level above this Utils folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

def save_fig(fname: str, dpi: int = 300, show: bool = False) -> Path:
    """
    Save current Matplotlib figure into <project_root>/figures
    and optionally display a window if `show` is True.
    Returns the Path to the saved file.
    """
    path = FIG_DIR / fname
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
    return path
