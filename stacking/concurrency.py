"""
Classe contenente la logica per la parallelizzazione

Il GIL (Global Interpreter Lock) permette a un solo thread alla volta di eseguire bytecode Python.
Dunque i Thread sono ottimi per operazioni I/O-bound (lettura/scrittura disco, rete) perché mentre un thread aspetta
un altro esegue. I processi sono preferibili per operazioni CPU-bound (calcoli numerici, loop intensi), perché ognuno di
essi possiede il proprio GIL (vero parallelismo).

I Thread condividono oggetti (scambio dati economico). I processi non condividono oggetti e i dati passano per pickle
(la serializzazione può costare molto con array grandi)
"""
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Any, Iterable, Optional, Tuple, Iterator

import logging
import numpy as np
import os
import multiprocessing as mp

"""
Funzione che viene passata come initializer al ProcessPoolExecutor in process_map, prima di eseguire i job.
"""
def worker_init() -> None:
    """
    Blocco di codice che limita il threading interno nei worker (OpenCV/BLAS)

    os.environ.setdefault("OMP_NUM_THREADS", "1"): se la variabile d’ambiente non è già impostata nel processo worker,
    la imposta a "1". Come effetto le librerie che usano OpenMP (NumPy/Scipy/numexpr/alcuni backend) useranno 1 thread
    interno per operazione in quel processo.

    os.environ.setdefault("MKL_NUM_THREADS", "1"): limita i thread per intel MKL usato da alcune build di NumPy.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import cv2
        cv2.setNumThreads(1)    # richiesta esplicita di utilizzo di 1 thread interno nel processo worker
    except Exception:
        pass


"""
Wrapper per istanziare i workers
"""
def _exec_worker(worker: Callable[[], Any]) -> Any:
    return worker()


"""
Funzione per implementare la programmazione concorrente nell'allineamento delle immagini
"""
def process_map_workers(workers: Iterable[Callable[[], Any]],
                        max_workers: Optional[int] = None) -> Iterator[Any]:
    if max_workers is None:
        try:
            # vengono utilizzati tutti i core del processore - 1 (per sistema e UI)
            max_workers = max(os.cpu_count() - 1, 1)
            logging.info(f"Concurrency: {max_workers} workers.")
        except Exception:
            max_workers = 1
            logging.info(f"Concurrency unavailable: {max_workers} worker.")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init, mp_context=ctx) as ex:
        """
        Ogni worker esegue la funzione target sugli argomenti passati
        
        ex.map è una API che garantisce la restituzione dei risultati nello stesso ordine in cui vengono passati
        ex.submit() + concurrent.futures.as_completed() restituisce i risultati appena termina il calcolo (non ordine)
        
        ex.map(func, args_iter) restituisce un iterator sui risultati dei task che viene consumato con for res in ...
        yield res fa di process_map un generatore: passa uno alla volta i risultati al chiamante, senza accumularli in 
        una lista.
        """
        for res in ex.map(_exec_worker, list(workers)):
            yield res


# worker astratto
class Worker(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError


# classe concreta per implementare il parallelismo del processo di allineamento
class AstroAlignerWorker(Worker):
    def __init__(self, i: int, tgt_gray: np.ndarray, ref_gray: np.ndarray,):
        self.i = i
        self.tgt_gray = tgt_gray
        self.ref_gray = ref_gray

    def __call__(self) -> Tuple[int, Optional[np.ndarray]]:
        from .aligners import find_transform   # import LAZY per evitare import circolare
        matrix = find_transform(self.tgt_gray, self.ref_gray)
        return self.i, matrix


# classe concreta per implementare il parallelismo del processo di stacking
class AstroStackerWorker(Worker):
    def __init__(self, y1: int, y2: int, x1: int, x2: int,
                 tile_stack: np.ndarray, sigma_low: float, sigma_hi: float, iterations: int):
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

        self.tile_stack = tile_stack
        self.sigma_low = sigma_low
        self.sigma_hi = sigma_hi

        self.iterations = iterations

    def __call__(self) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        from .stackers import sigma_clip_tile   # lazy import
        out_tile = sigma_clip_tile(self.tile_stack, self.sigma_low, self.sigma_hi, self.iterations)
        return (self.y1, self.y2, self.x1, self.x2), out_tile