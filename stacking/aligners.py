"""
Contiene la logica per l'allineamento delle immagini
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from stacking.common_utils import to_gray_f32
from stacking.concurrency import AstroAlignerWorker, process_map_workers

import cv2
import logging
import numpy as np

"""
Funzione che accetta in input due immagini in scala di grigi e restituisce la matrice di trasformazione H necessaria 
ad allinearle
"""
def find_transform(tgt_gray: np.ndarray, ref_gray: np.ndarray) -> Optional[np.ndarray]:
    try:
        ref_u8 = (np.clip(ref_gray, 0, 1) * 255).astype(np.uint8)   # trasformazione in uint8 per ORB
        tgt_u8 = (np.clip(tgt_gray, 0, 1) * 255).astype(np.uint8)

        orb = cv2.ORB.create(4000)  # individua i punti caratteristici e le loro features
        mask: Optional[np.ndarray] = None
        k1, d1 = orb.detectAndCompute(ref_u8, mask) # k sono le coordinate dei punti individuati
        k2, d2 = orb.detectAndCompute(tgt_u8, mask) # d sono i descrittori per il confronto
        if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   # confronta i descrittori e trova le coppie simili
        matches = bf.match(d1, d2)
        if len(matches) < 4:
            return None

        matches = sorted(matches, key=lambda m: m.distance)[:200]   # ordina per qualità e usa i migliori 200
        src = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)    # punti di tgt
        dst = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)    # punti di ref
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        matrix, _ = cv2.findHomography(src, dst, method, 3.0)   # calcola la trasformazione per l'allineamento
        return matrix
    except Exception:
        return None


# classe astratta da cui ereditano tutte le tipologie di aligner
class BaseAligner(ABC):
    @abstractmethod
    def align_all(self, images: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError


# aligner concreto per allineare le stelle in immagini traslate l'una rispetto all'altra (rotazione terrestre)
class AstroAligner(BaseAligner):
    def __init__(self, ref_idx: Optional[int] = None) -> None:
        self.ref_idx = ref_idx  # se None, usa l'immagine centrale (vedi align_all)

    """
    Applica find_transform a tutte le immagini della lista images, lasciando invariata quella di riferimento
    """
    def align_all(self, images: List[np.ndarray]) -> List[np.ndarray]:

        ref_idx = self.ref_idx if self.ref_idx is not None else len(images) // 2
        ref_img = images[ref_idx]

        ref_gray = to_gray_f32(ref_img, normalize=True)

        # preparazione dei workers
        workers = []
        for i, img in enumerate(images):
            if i == ref_idx:
                continue

            tgt_gray = to_gray_f32(img, normalize=True)
            workers.append(AstroAlignerWorker(i = i, tgt_gray = tgt_gray, ref_gray = ref_gray))

        aligned: List[np.ndarray] = list(images)
        aligned[ref_idx] = ref_img  # reference invariata

        """
        process_map_workers usa ProcessPoolExecutor con ~os.cpu_count()-1 processi.
        Ogni processo ha il proprio GIL, quindi i job possono essere calcolati in parallelo su core diversi
        """
        for i, matrix in process_map_workers(workers):  # calcolo in parallelo delle matrici di trasformazione
            logging.info(f"Finding transformation for image {i}")

            if matrix is None:
                logging.info(f"Transformation finding failed for image {i}")
            else:
                """
                L'unico momento in cui avviene programmazione concorrente è nel calcolo della matrice stessa. 
                In questo modo non vengono spostate immagini di grandi dimensioni tra processi in modo da migliorare 
                l'efficienza 
                """
                ref_h, ref_w = ref_img.shape[:2]    # dimensioni delle immagini
                # cv2.INTER_LANCZOS4 per migliori risultati -> lentezza
                # cv2.INTER_CUBIC per compromesso risultato/velocità
                aligned[i] = cv2.warpPerspective(images[i], matrix, (ref_w, ref_h),
                                      flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                logging.info(f"Transformation applied to image {i}.")

        return aligned


class FocusAligner(BaseAligner):
    def align_all(self, images: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError("FocusAligner non ancora implementato")


class ExposureAligner(BaseAligner):
    def align_all(self, images: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError("ExposureAligner non ancora implementato")