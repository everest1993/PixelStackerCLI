"""
Contiene funzioni di IO necessarie alla lettura delle immagini in ingresso
"""
from pathlib import Path
from typing import Iterable, List

import cv2
import rawpy    # per ARW/RAW
import numpy as np

from stacking.common_utils import autocrop_trailing_zeros

RAW_EXTS = {".arw", ".nef", ".cr2", ".cr3", ".rw2", ".orf", ".raf", ".dng"} # formati RAW dei vari manufacturer

def read_raw(path: Path) -> np.ndarray:
    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1)
        )

    bgr16 = cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)

    bgr16 = autocrop_trailing_zeros(bgr16)  # crop della porzione valida dell'immagine
    return bgr16


# legge un'immagine a colori dal percorso specificato
def read_color(path: Path) -> np.ndarray:   # la funzione restituisce un array NumPy (type hint)
    ext = path.suffix.lower()   # legge il formato
    if ext in RAW_EXTS: # se il formato è RAW
        img = read_raw(path)    # utilizza la funzione rawpy per leggere le immagini RAW
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)   # altrimenti legge l'immagine con OpenCV

    if img is None:
        raise RuntimeError(f"Impossibile leggere l'immagine: {path}")
    return img


# funzione per leggere una lista immagini in input
def read_imgs(paths: Iterable[Path]) -> List[np.ndarray]:
    imgs = [read_color(p) for p in paths]
    return imgs