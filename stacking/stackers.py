"""
Contiene la logica per eseguire noise stacking attraverso il metodo Sigma Clipping (rimuove oggetti mobili)
"""

from typing import List
from abc import ABC, abstractmethod
from stacking.concurrency import AstroStackerWorker, process_map_workers
from astropy.stats import sigma_clip, mad_std

import numpy as np
import logging

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
    iterations: int
) -> np.ndarray:

    arr = tile_stack.astype(np.float32)

    # caso semplice: nessun clipping richiesto
    if sigma_low is None and sigma_hi is None:
        return np.mean(arr, axis=0, dtype=np.float32)

    # sigma clipping usando Astropy
    masked = sigma_clip(
        data=arr,
        sigma_lower=sigma_low,
        sigma_upper=sigma_hi,
        maxiters=iterations,
        cenfunc=np.median,
        stdfunc=mad_std,
        axis=0,
        grow=True
    )

    # media sui valori non mascherati (validi)
    mean_clipped = masked.mean(axis=0)
    # mediana sui tutti i pixel
    med_all = np.median(arr, axis=0)
    # ricostruzione dell'immagine coerente
    out = mean_clipped.filled(med_all).astype(np.float32)

    return out


# classe astratta da cui ereditano tutti i tipi di stacker
class BaseStacker(ABC):
    @abstractmethod
    def stack_all(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


# stacker concreto che utilizza la funzione sigma clipping per ridurre il rumore in una lista di foto
class SigmaClippingNoiseStacker(BaseStacker):
    def __init__(self, sigma_low: float, sigma_hi: float, iterations: int, tile: int = 512):
        self.sigma_low = sigma_low
        self.sigma_hi = sigma_hi  # intensità del filtro per gli outliers
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
                        iterations=self.iterations
                    )
                )

        for (y1, y2, x1, x2), tile_out in process_map_workers(workers): # applicazione del sigma clipping parallelo
            out[y1:y2, x1:x2, ...] = tile_out

        logging.info("Stacking process successfully completed.")
        return out.astype(np.float32)


"""
Restituisce parametri sigma-clipping adattivi al numero di immagini n in ingresso (sigma_low, sigma_hi)
"""
def adaptive_params(n: int):
    if n < 5:
        return None, None   # media
    if 5 <= n <= 20:
        return np.inf, 5.0
    else:
        return np.inf, 10.0


class FocusStacker(BaseStacker):
    def stack_all(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("FocusStacker non ancora implementato")


class ExposureStacker(BaseStacker):
    def stack_all(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("ExposureStacker non ancora implementato")