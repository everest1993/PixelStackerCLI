"""
Contiene funzioni di utility necessarie all'allineamento, allo stacking e all'esecuzione.
"""

import cv2
import numpy as np
import sys

from pathlib import Path

"""
Funzione per aumentare l'esposizione dell'immagine finale
"""
def to_display_srgb(img16: np.ndarray, ev: float = 0.0) -> np.ndarray:
    im = img16.astype(np.float32)   # conversione in float32

    """
    Exposure Shift
    ev=+1 = raddoppia la luminosità (+1 stop)
	ev=-1 = dimezza la luminosità (-1 stop)
    """
    im *= (2.0 ** ev)

    im = np.clip(im, 0, 65535)  # clamp valori
    x = im / 65535.0    # normalizzazione in [0, 1]

    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1/2.4) - 0.055) # conversione da linear RGB a sRGB

    return np.clip(srgb * 65535.0, 0, 65535).astype(np.uint16)  # denormalizzazione


"""
Molte fotocamere lasciano righe/colonne OB (optical black) a destra e in basso. 
rawpy.postprocess() in alcuni modelli non le rimuove. Gli indici raw.raw_image_visible_area non sempre corrispondono 
allo spazio già demosaicizzato/ruotato, quindi il crop “a indici RAW” può non riflettersi sul RGB finale.

La funzione rimuove le bande nere a livello RGB. Funziona per immagini mono o 3 canali
"""
def autocrop_trailing_zeros(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:   # immagine greyscale
        valid = img > 0 # costruzione di una maschera booleana dove sono validi (True) i pixel > 0
    else:
        valid = np.any(img > 0, axis=2) # in un'immagine a più canali un pixel valido se almeno un canale > 0

    rows = np.where(valid.any(axis=1))[0]   # per ogni riga controlla se almeno un pixel è valido
    cols = np.where(valid.any(axis=0))[0]   # per ogni colonna controlla se almeno un pixel è valido

    if rows.size == 0 or cols.size == 0:    # se ogni pixel è nero viene restituita l'immagine senza ritagliare
        return img

    y1 = int(rows[-1] + 1)  # individua l'ultima riga valida (+ 1 perche slicing esclusivo)
    x1 = int(cols[-1] + 1)  # individua l'ultima colonna valida
    return img[:y1, :x1]    # crop fino alla riga e alla colonna valida (dall'alto verso il basso)


"""
Normalizza i valori dei pixel nell'intervallo [0, 1].
Supporta uint8, uint16 e float* (mantiene scala se già in [0,1])
"""
def normalize_img(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0

    imgf = img.astype(np.float32)   # se l'immagine non è uint8 ne uint16 converte in float32
    mx = float(np.max(imgf))    # trova il massimo
    return imgf if mx <= 1.0 or mx == 0.0 else (imgf / mx)  # normalizza secondo l'intervallo di valori trovato


"""
Converte un'immagine BGR/BGRA o mono in grayscale float32.
Se normalize=True, riporta i valori in [0,1]
"""
def to_gray_f32(img: np.ndarray, normalize: bool = False) -> np.ndarray:
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        g = cv2.cvtColor(img, code)
    else:
        raise ValueError(f"Formato immagine non supportato: {img.shape}")
    g = g.astype(np.float32)
    return normalize_img(g) if normalize else g


"""
restituisce il percorso assoluto alla cartella o file della risorsa, sia eseguendo il codice da 
Python direttamente sia da un eseguibile PyInstaller
"""
def res_path(rel: str) -> Path:
    # in bundle PyInstaller -> sys._MEIPASS
    # altrimenti usa la root del progetto
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return (base / rel).resolve()