"""
Contiene la logica per eseguire noise stacking attraverso il metodo Sigma Clipping (rimuove oggetti mobili)
"""
from typing import List
from abc import ABC, abstractmethod
from stacking.concurrency import AstroStackerWorker, process_map_workers
from astropy.stats import sigma_clip, mad_std
from functools import partial

import numpy as np
import logging
import math

"""
In un blocco (tile) di immagini sovrapposte dello stesso punto del cielo possono essere presenti:
- segnale buono (stelle, nebulose, galassie) presente quasi tutti i frame
- rumore casuale che si distribuisce intorno al segnale
- outliers come satelliti o aerei che appaiono solo in pochi frame

L’obiettivo del metodo è tenere il segnale, togliere gli outliers luminosi positivi e ridurre il rumore:
- aggiunge tutti i valori dei pixel del tile in un array
- stabilisce quanti valori devono sopravvivere così da non basarsi su pochi campioni isolati
- utilizza il metodo di sigma clipping di astropy per eseguire l'elaborazione
- se il conteggio dei valori non è sufficiente utilizza la mediana come fallback
"""
def sigma_clip_tile(
    tile_stack: np.ndarray,
    sigma_low: float,
    sigma_hi: float,
    min_keep: int,
    min_keep_frac: float,
    iterations: int) -> np.ndarray:

    arr = tile_stack.astype(np.float32)
    n = arr.shape[0]

    # calcolo sicuro del min_keep_abs
    min_keep_abs = max(min_keep, int(np.ceil(min_keep_frac * n)))
    min_keep_abs = min(max(1, min_keep_abs), n)  # cap tra 1 e n

    # caso N piccolo o clipping disattivato: media
    if sigma_low is None and sigma_hi is None:
        return np.mean(arr, axis=0, dtype=np.float32)

    # sigma-clipping con lati eventualmente disattivati (None)
    masked = sigma_clip(
        data=arr,
        sigma_lower=sigma_low,
        sigma_upper=sigma_hi,
        maxiters=iterations,
        cenfunc=np.median,
        stdfunc=partial(mad_std, ignore_nan=True),
        axis=0,
        grow=False
    )

    # media dei valori non mascherati
    mean_clipped = masked.mean(axis=0).filled(0.0).astype(np.float32)
    count_kept = (~np.ma.getmaskarray(masked)).sum(axis=0)

    # fallback se non ci sono abbastanza valori validi
    med_all = np.median(arr, axis=0).astype(np.float32)
    use_fallback = (count_kept < min_keep_abs)
    out = np.where(use_fallback, med_all, mean_clipped)

    return out


# classe astratta da cui ereditano tutti i tipi di stacker
class BaseStacker(ABC):
    @abstractmethod
    def stack_all(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


# stacker concreto che utilizza la funzione sigma clipping per ridurre il rumore in una lista di foto
class SigmaClippingNoiseStacker(BaseStacker):
    def __init__(self, sigma_low: float, sigma_hi: float, min_keep: int, min_keep_frac: float,
                 iterations: int, tile: int = 512):
        self.sigma_low = sigma_low
        self.sigma_hi = sigma_hi  # intensità del filtro per gli outliers
        self.min_keep = min_keep    # minimo di campioni buoni
        self.min_keep_frac = min_keep_frac
        self.iterations = iterations
        self.tile = tile    # dimensione del tile (quadrato)

    # implementazione del metodo astratto della super classe
    def stack_all(self, images: List[np.ndarray]) -> np.ndarray:
        stk = np.stack(images, axis=0).astype(np.float32)  # array 4D e conversione in float32

        h, w = stk.shape[1], stk.shape[2]   # dimensioni
        c = stk.shape[3]  # colore

        out = np.empty((h, w, c), dtype=np.float32) # inizializzazione del risultato

        workers = []    # inizializzazione dei workers per il calcolo in parallelo su tile separati
        t = self.tile
        for y in range(0, h, t):
            for x in range(0, w, t):
                y2, x2 = min(y + t, h), min(x + t, w)   # evita di uscire dai bordi dell'immagine

                """
                Sull'asse 0 prende tutti i pixel da y a y2 - 1 e da x a x2 - 1
                ... dice a NumPy di prendere tutto quello che resta negli assi successivi (colore se presente)
                """
                tile_stack = stk[:, y:y2, x:x2, ...]
                workers.append( # viene aggiunto il worker con i rispettivi indici
                    AstroStackerWorker(
                        y1=y, y2=y2, x1=x, x2=x2,
                        tile_stack=tile_stack,
                        sigma_low=self.sigma_low,
                        sigma_hi=self.sigma_hi,
                        min_keep=self.min_keep, min_keep_frac=self.min_keep_frac,
                        iterations=self.iterations
                    )
                )

        for (y1, y2, x1, x2), tile_out in process_map_workers(workers): # applicazione del sigma clipping parallelo
            out[y1:y2, x1:x2, ...] = tile_out

        logging.info("Stacking process successfully completed.")
        return out.astype(np.float32)


# adatta i parametri del sigma clipping al numero di immagini in input
def adaptive_params(n: int):
    # n < 4: nessun clipping, media
    if n < 4:
        sigma_low = None
        sigma_hi = None
        min_keep_frac = 1.0
        min_keep = n
        return sigma_low, sigma_hi, min_keep, min_keep_frac

    # 4 < n < 8: transizione lineare morbida
    if 4 <= n <= 8:
        t = (n - 4) / 4.0
        sigma_hi = 7.5 - t * (7.5 - 6.5)
        sigma_low = 4.5 - t * (4.5 - 3.6)
        min_keep_frac = 0.70 - t * (0.70 - 0.60)
        min_keep = max(4, int(np.ceil(min_keep_frac * n)))
        return sigma_low, sigma_hi, min_keep, min_keep_frac

    # 9 < n < 19: curva log per adattarsi a più frame
    if 9 <= n < 20:
        sigma_hi = 6.5 - 0.7 * math.log2(n / 8)
        sigma_low = 3.6 - 0.3 * math.log2(n / 8)
        min_keep_frac = 0.60 - 0.10 * math.log2(n / 8)

    # n ≥ 20: preset ottimizzato per molti frame
    else:  # n >= 20
        sigma_hi = 4.9
        sigma_low = 3.1
        min_keep_frac = 0.54

    # clamp
    sigma_hi = min(max(sigma_hi, 4.5), 7.5)
    sigma_low = min(max(sigma_low, 3.0), 4.5)
    min_keep_frac = min(max(min_keep_frac, 0.50), 0.70)
    min_keep = max(4, int(np.ceil(min_keep_frac * n)))

    return sigma_low, sigma_hi, min_keep, min_keep_frac


class FocusStacker(BaseStacker):
    def stack_all(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("FocusStacker non ancora implementato")


class ExposureStacker(BaseStacker):
    def stack_all(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("ExposureStacker non ancora implementato")