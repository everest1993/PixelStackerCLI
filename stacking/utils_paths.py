from pathlib import Path
import sys

def res_path(rel: str) -> Path:
    # in bundle PyInstaller -> sys._MEIPASS
    # altrimenti usa la root del progetto
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return (base / rel).resolve()