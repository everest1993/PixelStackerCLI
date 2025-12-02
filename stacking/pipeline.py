"""
Contiene la logica per l'implementazione delle pipeline logiche
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from stacking.aligners import AstroAligner
from stacking.stackers import SigmaClippingNoiseStacker, adaptive_params
from stacking.mask_generator import detect_sky_mask
from stacking.common_utils import to_display_srgb
from stacking.common_utils import res_path

import numpy as np
import cv2
import logging

# classe astratta da cui ereditano le varie pipeline
class BasePipeline(ABC):
    @abstractmethod
    def run(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


"""
Pipeline di esecuzione del noise stacking

LOGICA ATTUALE:
1) ESTRAE IL CIELO DELL'IMMAGINE DI REFERENCE ATTRAVERSO LA SKY MASK
2) ALLINEA LE IMMAGINI RESTANTI AL CIELO DELLA REFERENCE
3) ESEGUE LO STACKING DEL CIELO
4) COMPONE L'IMMAGINE FINALE UTILIZZANDO IL FOREGROUND DELLA REFERENCE E IL CIELO STACKED
5) RIPRISTINA L'ESPOSIZIONE DIMINUITA IN FASE DI STACKING
"""
class NoiseStackingPipeline(BasePipeline):

    def __init__(self, ref_idx: Optional[int] = None,
                sigma_low: float = None, sigma_hi: float = None, iterations: int = None) -> None:
        self.ref_idx = ref_idx
        self.sigma_low = sigma_low
        self.sigma_hi = sigma_hi
        self.iterations = iterations

    def run(self, images: List[np.ndarray]) -> np.ndarray:
        if not images: raise ValueError("NoiseStackingPipeline: lista 'images' vuota.")

        logging.info("Noise Stacking process started...")

        n = len(images)
        # calcola parametri sigma clipping se non definiti
        ad_sigma_low, ad_sigma_hi = adaptive_params(n)

        # override da costruttore
        self.sigma_low = self.sigma_low if self.sigma_low is not None else ad_sigma_low
        self.sigma_hi = self.sigma_hi if self.sigma_hi is not None else ad_sigma_hi

        if self.iterations is None:
            self.iterations = 1 if n < 4 else 2

        if n < 4: logging.warning("N < 4: sigma-clipping disattivato, uso media/mediana diretta.")

        # immagine di reference
        ref_img_index = self.ref_idx if self.ref_idx is not None else (len(images) // 2)
        ref_img = images[ref_img_index]

        logging.info(f"Reference image index: {ref_img_index}")
        logging.info("Generating sky mask...")

        # 1) Generazione della sky mask sull'immagine di reference
        sky_mask = detect_sky_mask(
            ref_img,
            prob_thresh=0.6,
            max_side=1024,
            model_id=str(res_path("assets/m2f/mask2former-swin-tiny-ade-semantic")), # modello
            # save_path="/Users/luigivannozzi/Desktop/sky_mask.tif"   # output della sky mask per debugging
        )

        # erosione della sky mask (evita leaking sul foreground)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        sky_mask_safe = cv2.erode(sky_mask, k)

        print("STEP 1/3", flush=True)   # aggiornamento GUI

        # 2) Allineamento delle immagini restanti sulle stelle della reference (ECC)
        aligner = AstroAligner()
        logging.info("Starting alignment...")
        aligned = aligner.align_all(images)  # lista di immagini allineate alla reference

        print("STEP 2/3", flush=True)   # aggiornamento GUI

        # 3) Noise stacking
        stacker = SigmaClippingNoiseStacker(sigma_low=self.sigma_low,
                                            sigma_hi=self.sigma_hi,
                                            iterations=self.iterations)
        logging.info(f"Stacking images using Sigma Clipping ({self.iterations} iterations)...")
        stacked = stacker.stack_all(aligned)  # valori in float32


        # 4) Ricomposizione: cielo (stacked) + foreground (reference)
        final = ref_img.copy()
        logging.info("Generating result...")
        cv2.copyTo(stacked, sky_mask_safe, final)   # copia il cielo (stacked) solo nelle zone individuate dalla mask

        # 5) Ripristino esposizione
        final_display = to_display_srgb(final.astype(np.float32), ev=+0.6)  # ev = +0.6 restituisce risultati più fedeli

        print("STEP 3/3", flush=True)   # aggiornamento GUI

        return final_display


"""
Pipeline di esecuzione del focus stacking
Implementazione futura
"""
class FocusPipeline(BasePipeline):
    def __init__(self) -> None:
        pass

    def run(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("FocusPipeline non ancora implementata")


"""
Pipeline di esecuzione del focus stacking
Implementazione futura
"""
class ExposurePipeline(BasePipeline):
    def __init__(self) -> None:
        pass

    def run(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("ExposurePipeline non ancora implementata")