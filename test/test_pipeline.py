import numpy as np
import pytest
import stacking.pipeline as pipe

def _fake_img(h=10, w=12, c=3, value=0, dtype=np.uint8):    # genera immagini finte
    return np.full((h, w, c), value, dtype=dtype)   # con valori value costanti


imgs = [_fake_img(value=10),    # lista di immagini
        _fake_img(value=20),
        _fake_img(value=30),
        _fake_img(value=40),
        _fake_img(value=50)]

called = {  # dizionario per memorizzare i parametri delle chiamate alla pipeline
    "detect": {},
    "align_all": [],
    "stack_creator": None,
    "stack_all": False,
}

fake_mask = np.ones(imgs[0].shape[:2], dtype=np.uint8)  # crea una maschera corrispondente all'intera immagine


def fake_detect(bgr, prob_thresh, max_side, model_id, save_path=None):  # sostituisce detect_sky_mask
    called["detect"] = {
        "bgr": bgr,
        "prob_thresh": prob_thresh,
        "max_side": max_side,
        "model_id": model_id,
        "save_path": save_path,
    }
    return fake_mask    # restituisce fake_mask


# classe fake per testare la pipeline
class FakeAligner:
    def align_all(self, images):
        called["align_all"] = list(images)
        return images


# classe fake per testare la pipeline
class FakeStacker:
    def __init__(self, sigma_low, sigma_hi, min_keep, min_keep_frac, iterations):
        called["stack_creator"] = dict(sigma_low=sigma_low, sigma_hi=sigma_hi,
                                       min_keep=min_keep, min_keep_frac=min_keep_frac, iterations=iterations)
    def stack_all(self, im):
        called["stack_all"] = True
        return _fake_img(value=123, dtype=np.float32)


def test_noise_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(pipe, "detect_sky_mask", fake_detect)
    monkeypatch.setattr(pipe, "AstroAligner", FakeAligner)
    monkeypatch.setattr(pipe, "SigmaClippingNoiseStacker", FakeStacker)

    p = pipe.NoiseStackingPipeline(ref_idx=None, sigma_low=None, sigma_hi=None,
                                   min_keep=None, min_keep_frac=None, iterations=None)
    out = p.run(imgs)

    assert called["detect"]["bgr"] is imgs[2], "La sky mask deve essere generata sulla ref centrale"
    assert called["detect"]["prob_thresh"] == 0.6
    assert called["detect"]["max_side"] == 1024
    assert "mask2former" in called["detect"]["model_id"]
    assert called["detect"]["save_path"] is None
    assert called["align_all"] == imgs

    std = capsys.readouterr().out
    assert "STEP 1/3" in std and "STEP 2/3" in std and "STEP 3/3" in std


def test_noise_pipeline_respects_explicit_ref_idx(monkeypatch):
    imgs = [_fake_img(value=i*10) for i in range(4)]    # 0, 10, 20, 30
    explicit_ref = 1    # seconda immagine

    chosen = {"ref": None}  # dizionario

    """
    Aggiorna il dizionario e ritorna la fake_mask per il metodo erode di pipeline.py. Il metodo update ritorna sempre 
    false dunque, dopo aver aggiornato ref, il risultato della lambda sarà sempre fake_mask
    """
    monkeypatch.setattr(pipe, "detect_sky_mask", lambda bgr, **kw: (chosen.update(ref=bgr) or fake_mask))
    monkeypatch.setattr(pipe, "AstroAligner", FakeAligner)
    monkeypatch.setattr(pipe, "SigmaClippingNoiseStacker", FakeStacker)

    p = pipe.NoiseStackingPipeline(ref_idx=explicit_ref)
    _ = p.run(imgs)

    assert chosen["ref"] is imgs[explicit_ref], "La ref deve rispettare ref_idx esplicito"


def test_focus_and_exposure_pipelines_raise():
    with pytest.raises(NotImplementedError):
        pipe.FocusPipeline().run([_fake_img()])

    with pytest.raises(NotImplementedError):
        pipe.ExposurePipeline().run([_fake_img()])