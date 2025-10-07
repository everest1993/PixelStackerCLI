import numpy as np
import pytest
import stacking.stackers as stackers

def generate_tile_stack():  # genera 5 immagini formate da 2 righe e 2 colonne
    stack = np.array(
        [[[1, 100], [3, 4]],
         [[1, 100], [3, 4]],
         [[1, 100], [3, 4]],
         [[1, 100], [3, 4]],
         [[9, 999], [3, 4]]],
        dtype=np.float32,
    )

    return stack


def generate_mask():  # genera una maschera con la stessa struttura delle immagini del tile stack
    mask = np.array([
        [[False, True], [False, False]],
        [[False, True], [False, False]],
        [[False, True], [False, False]],
        [[False, False], [False, False]],
        [[False, False], [False, False]], ],
        dtype=bool,
    )

    return mask


def test_sigma_clip_tile_uses_sigma_clip_and_fallback(monkeypatch):
    tile_stack = generate_tile_stack()
    mask = generate_mask()

    masked_return = np.ma.MaskedArray(tile_stack, mask=mask)  # nasconde True e mostra False
    monkeypatch.setattr(stackers, "sigma_clip", lambda **kw: masked_return)

    out = stackers.sigma_clip_tile(tile_stack, sigma_low=3.0, sigma_hi=3.0, min_keep=3, min_keep_frac=0.1, iterations=1)
    assert out.shape == (2, 2)  # dalle 5 immagini iniziali ne risulta una con 2 righe e 2 colonne
    assert np.isclose(out[0, 0], 2.6, atol=1e-5)  # tutti i valori validi -> media (1,1,1,1,9)/5
    assert out[0, 1] == 100.0  # la maschera esclude i primi 3 valori -> fallback a mediana (100 100 !100! 100 999)
    assert out[1, 0] == 3.0  # tutti i valori sono validi -> media
    assert out[1, 1] == 4.0  # tutti i valori sono validi -> media
    assert out.dtype == np.float32


def test_SigmaClippingNoiseStacker_stacking():
    # genera una lista di 4 immagini da 6×4 pixel con un solo canale di colore
    imgs = [np.ones((6, 4, 1), dtype=np.uint16) * v for v in (10, 20, 30, 40)]

    s = stackers.SigmaClippingNoiseStacker(sigma_low=1.0, sigma_hi=1.0,
                                           min_keep=1, min_keep_frac=0.1, iterations=1, tile=3)
    out = s.stack_all(imgs)

    assert out.dtype == np.float32
    assert np.all(out == 25)    # tutti i pixel devono valere 25
    assert out.shape == (6, 4, 1)   # la shape deve essere uguale a quella delle immagini in input


def test_focus_and_exposure_stackers_raise():
    with pytest.raises(NotImplementedError):
        stackers.FocusStacker().stack_all([np.zeros((2, 2, 3), np.uint8)])

    with pytest.raises(NotImplementedError):
        stackers.ExposureStacker().stack_all([np.zeros((2, 2, 3), np.uint8)])