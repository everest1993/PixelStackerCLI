import sys
import pytest
import stacking.mask_generator as mg
import numpy as np

from contextlib import nullcontext
from types import SimpleNamespace

"""
Funzione finta che imita l’ImageProcessor di Hugging Face
- accetta due parametri (images, return_tensors) solo per compatibilità di firma
- ritorna un oggetto vuoto che chiama il metodo to
- to crea un finto tensore con shape
    - 1 = batch size
    - 3 = canali (RGB)
    - 8x8 = dimensione
"""
def _processor(images, return_tensors):
    return SimpleNamespace(to=lambda device: {"pixel_values": np.zeros((1, 3, 8, 8), dtype=np.float32)})


"""
Simula processor.post_process_semantic_segmentation()
- crea una mappa di segmentazione
- prima riga = 0 (classe “sky”)
- righe successive = 1 (non sky)
"""
def _post_process(_out, target_sizes):
    H, W = target_sizes[0]
    seg = np.zeros((H, W), dtype=np.int64)  # crea una matrice di dimensioni HxW piena di zeri
    seg[1:, :] = 1  # uguaglia a 1 tutte le righe dall'indice 1 in poi (indice 0 = 0)
    return [seg]


# prepara la cache globale _M2F del modulo mask_generator con oggetti finti
def _prime_cache_with_fakes(id2label):
    _processor.post_process_semantic_segmentation = _post_process   # sostituisce post_process_semantic_segmentation

    model = lambda **_kw: SimpleNamespace() # modello finto che restituisce un SimpleNamespace vuoto
    model.config = SimpleNamespace(id2label=id2label)

    torch_mod = SimpleNamespace(    # imita un sottoinsieme di torch
        device=lambda *_a, **_k: "cpu",
        no_grad=nullcontext
    )

    mg._M2F.update(dict(    # aggiorna il dizionario globale di mask_generator
        loaded=True,
        processor=_processor,
        model=model,
        device="cpu",
        id2label=id2label,
        torch_mod=torch_mod,
    ))


def test_load_mask2former_cpu_raises_without_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="Mask2Former non disponibile"):
        mg._load_mask2former_cpu("dummy/model")


def test_detect_sky_mask_resizes_back_to_original():
    _prime_cache_with_fakes({0: "sky", 1: "ground"})

    bgr = np.zeros((800, 1200, 3), dtype=np.uint8)
    out = mg.detect_sky_mask(bgr, prob_thresh=0.5, max_side=200, model_id="ignored")

    assert out.shape == (800, 1200)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_detect_sky_mask_when_no_sky_in_labels_returns_zeros():
    _prime_cache_with_fakes({0: "road", 1: "ground"})

    bgr = np.zeros((5, 7, 3), dtype=np.uint8)
    out = mg.detect_sky_mask(bgr, prob_thresh=0.5, max_side=1024, model_id="ignored")

    assert out.shape == (5, 7)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_detect_sky_mask_accepts_uint16_input():
    _prime_cache_with_fakes({0: "sky", 1: "ground"})

    bgr16 = np.zeros((3, 4, 3), dtype=np.uint16)
    bgr16[0, 0, :] = 10000
    out = mg.detect_sky_mask(bgr16, prob_thresh=0.5, max_side=1024, model_id="ignored")

    assert out.shape == (3, 4)
    assert out.dtype == np.uint8
    assert np.all(out[0, :] == 255)
    assert np.all(out[1:, :] == 0)