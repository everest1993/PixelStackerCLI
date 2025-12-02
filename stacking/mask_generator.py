"""
Contiene funzioni per generare una maschera del cielo su cui applicare il noise stacking
"""

from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image

import logging
import numpy as np
import cv2

# dizionario globale per cache modello AI
_M2F: Dict[str, Any] = {
    "loaded": False,
    "model": None,
    "processor": None,
    "device": None,
    "id2label": None,
    "torch_mod": None,
}

"""
Carica Mask2Former (Hugging Face) su CPU e lo memorizza in cache.
"""
def _load_mask2former_cpu(model_id: str) -> None:
    if _M2F.get("loaded", False):
        return

    try:
        import torch as torch_mod
        from transformers import (Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation)

    except ImportError as e:
        raise RuntimeError(
            "Mask2Former non disponibile."
        ) from e

    # definisce il device (CPU) prima di usarlo per il caching
    device = torch_mod.device("cpu")

    # fallback
    local_path = Path(model_id)
    download_repo_id = "facebook/mask2former-swin-tiny-ade-semantic"  # repository da scaricare

    # inizializza a None. Popolamento nel caricamento/download
    processor = None
    model = None

    # tentativo di caricamento locale
    try:
        if local_path.exists() and (local_path / "model.safetensors").exists():
            processor = Mask2FormerImageProcessor.from_pretrained(local_path, local_files_only=True)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(local_path, local_files_only=True)
            logging.info(f"Modello locale presente in '{local_path}'. Download non necessario.")
    except Exception as e:
        logging.warning(f"Errore nel caricamento locale ({local_path}): {e}. Scarico da Hugging Face...")
        processor = None
        model = None

    # scarica da Hugging Face se il caricamento locale non è riuscito
    if processor is None or model is None:
        print(f"Modello non trovato o caricamento locale fallito, scarico da Hugging Face come {download_repo_id}...")

        processor = Mask2FormerImageProcessor.from_pretrained(download_repo_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(download_repo_id)

        processor.save_pretrained(local_path)
        model.save_pretrained(local_path)

    _M2F.update(
        dict(
            loaded=True,
            processor=processor,
            model=model,
            device=device,
            id2label=model.config.id2label,
            torch_mod=torch_mod,
        )
    )


"""
Esegue la segmentazione del cielo su un'immagine BGR:
1) Ridimensionamento (se necessario) mantenendo il lato max = max_side
2) Segmentazione con modello
3) Estrazione della classe "sky"
4) Scaling alla dimensione originale
5) Generazione di una maschera del cielo binaria uint8 {0,255}

Parametri
- prob_thresh: soglia su probabilità (usata quando disponibili i logits)
- max_side: lato massimo per la segmentazione
- model_id: id del modello
- save_path: se fornito, salva la maschera nel percorso indicato
"""
def detect_sky_mask(
        bgr: np.ndarray,
        prob_thresh: float,
        max_side: int,
        model_id: str,
        save_path: Optional[str] = None,
) -> np.ndarray:
    if (not _M2F.get("loaded")) or any(  # se il modello non è caricato
            _M2F.get(k) is None for k in ("model", "processor", "device", "id2label", "torch_mod")
    ):
        _load_mask2former_cpu(model_id)  # invoca la funzione per la gestione del modello

    proc = _M2F["processor"]  # recupero attributi dal dizionario globale
    model = _M2F["model"]
    device = _M2F["device"]
    id2label = _M2F["id2label"]
    t = _M2F["torch_mod"]

    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:  # scaling immagine per ridurre il consumo delle risorse
        scale = max_side / float(max(h, w))
        bgr_s = cv2.resize(
            bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        bgr_s = bgr

    rgb = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2RGB)  # conversione da BGR in RGB per l'utilizzo del modello

    if rgb.dtype == np.uint16:  # conversione uint16 -> uint8
        rgb_u8 = (rgb / 257.0).astype(np.uint8)  # 65535 / 257 = 255
    elif rgb.dtype in (np.float32, np.float64):  # conversione float32/float64 -> uint8
        rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb.astype(np.uint8)  # immagine importata in uint8

    rgb_u8 = np.ascontiguousarray(rgb_u8)  # trasforma l'immagine in PIL (Python Imaging Library)
    inputs = proc(images=Image.fromarray(rgb_u8), return_tensors="pt").to(device)  # ottiene tensori PyTorch

    with t.no_grad():  # calcola le predizioni senza tenere traccia del gradiente (più veloce, meno memoria)
        out = model(**inputs)

    """
    I logits sono l’output grezzo del modello, prima che venga applicata una funzione di attivazione (come softmax).
    Sono valori reali che possono essere positivi o negativi, non sono ancora probabilità
    """
    logits = out.sem_seg if hasattr(out, "sem_seg") else None
    if logits is not None:
        logits = t.nn.functional.interpolate(
            logits, size=bgr_s.shape[:2], mode="bilinear", align_corners=False
        )[0]

    sky_ids = (  # cerca tutte le classi il cui nome contiene "sky"
        [i for i, n in id2label.items() if isinstance(n, str) and "sky" in n.lower()]
    )

    if logits is not None:
        """
        La softmax è una funzione che accetta un vettore di logits e lo trasforma in probabilità normalizzate tra 0 e 1.
        Calcola le probabilità per pixel, normalizzando lungo la dimensione delle classi
        """
        probs = t.softmax(logits, dim=0).float().cpu().numpy()  # calcola la probabilità softmax per ogni pixel
        if sky_ids:
            sky_prob = probs[sky_ids].sum(0) if len(sky_ids) > 1 else probs[sky_ids[0]]
        else:
            sky_prob = np.zeros(bgr_s.shape[:2], np.float32)
        mask_s = (sky_prob >= prob_thresh).astype(np.uint8) * 255  # se sky_prob >= prob_thresh → 255, altrimenti 0
    else:
        seg = proc.post_process_semantic_segmentation(
            out, target_sizes=[bgr_s.shape[:2]]
        )[0]
        if hasattr(seg, "cpu"):
            seg = seg.cpu().numpy()
        if sky_ids:
            mask_s = (np.isin(seg, np.array(sky_ids))).astype(np.uint8) * 255
        else:
            mask_s = np.zeros(bgr_s.shape[:2], np.uint8)

    # Riporta alla dimensione originale
    if scale != 1.0:
        mask_out = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_out = mask_s

    # Cast esplicito e clamp a {0,255}
    mask_out = (mask_out > 0).astype(np.uint8) * 255

    logging.info("Sky mask generated.")

    if save_path:
        cv2.imwrite(save_path, mask_out)

    return mask_out