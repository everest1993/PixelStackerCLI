import numpy as np
import pytest
import stacking.utils as utils

called = {"code": None}

def fake_cvtColor(img, code):   # funzione che ritorna una matrice di valori 42
    called["code"] = code
    return np.full((img.shape[0], img.shape[1]), 42, dtype=np.uint8)


def test_to_display_srgb_dtype_and_clamp():
    img16 = np.array([[0, 1, 65535, 70000, -50]], dtype=np.int32).astype(np.uint16) # array uint16 di forma 1, 5
    out = utils.to_display_srgb(img16, ev=0.0)

    assert out.dtype == np.uint16   # conformità valori
    assert out.shape == img16.shape # forma uguale all'input
    assert out.min() >= 0 and out.max() <= 65535    # clamp valori in uint16 [0, 65535]


def test_to_display_srgb_exposure_shift():
    img16 = np.full((4, 4), 10000, dtype=np.uint16) # matrice 4x4 con valore 10000 in ogni cella

    out_ev0 = utils.to_display_srgb(img16, ev=0.0)
    out_ev2 = utils.to_display_srgb(img16, ev=2.0)
    out_ev3 = utils.to_display_srgb(img16, ev=-1.0)

    assert out_ev0.shape == out_ev2.shape
    assert out_ev2.mean() > out_ev0.mean()
    assert out_ev3.mean() < out_ev0.mean()


def test_autocrop_trailing_zeros_gray():
    img = np.zeros((5, 6), dtype=np.uint16) # crea una matrice di zeri con 5 righe e 6 colonne
    img[:4, :4] = 123   # assegna il valore 123 ai pixel del quadrato 0, 1, 2, 3 x 0, 1, 2, 3
    out = utils.autocrop_trailing_zeros(img)    # esecuzione della funzione autocrop

    assert out.shape == (4, 4)  # validazione dimensioni output


def test_autocrop_trailing_zeros_color():
    img = np.zeros((4, 5, 3), dtype=np.uint16)  # crea una matrice di zeri con 4 righe e 5 colonne
    img[:3, :4, :] = 999    # assegna il valore 999 ai pixel del quadrato 0, 1, 2 x 0, 1, 2, 3
    out = utils.autocrop_trailing_zeros(img)

    assert out.shape == (3, 4, 3)
    assert np.all(out == 999)


def test_autocrop_trailing_zeros_all_black_returns_same():
    img = np.zeros((7, 9, 3), dtype=np.uint16)  # genera una matrice di zeri con 3 canali di colore
    out = utils.autocrop_trailing_zeros(img)

    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_normalize_img_uint8_and_uint16():
    u8 = np.array([[0, 255]], dtype=np.uint8)
    u16 = np.array([[0, 65535]], dtype=np.uint16)

    n8 = utils.normalize_img(u8)
    n16 = utils.normalize_img(u16)

    assert n8.dtype == np.float32 and n16.dtype == np.float32
    assert np.array_equal(n8, np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(n16, np.array([[0.0, 1.0]], dtype=np.float32))


def test_to_gray_f32_gray_passthrough_and_normalize():
    g = np.array([[0, 128, 255]], dtype=np.uint8)   # 2 canali (non usa cv2)
    out = utils.to_gray_f32(g, normalize=False)

    assert out.ndim == 2 and out.dtype == np.float32    # to_gray_f32 esegue una conversione di tipo in float32
    assert np.array_equal(out, g.astype(np.float32))    # i valori restano uguali dal momento che normalize=False

    outn = utils.to_gray_f32(g, normalize=True)
    assert outn.dtype == np.float32 # to_gray_f32 esegue una conversione di tipo in float32
    assert np.isclose(outn.min(), 0.0) and np.isclose(outn.max(), 1.0)  # verifica normalizzazione


def test_to_gray_f32_bgr_uses_cvtColor(monkeypatch):
    bgr = np.zeros((3, 5, 3), dtype=np.uint8)   # 3 canali (usa cv2)

    monkeypatch.setattr(utils.cv2, "cvtColor", fake_cvtColor)

    out = utils.to_gray_f32(bgr, normalize=False)

    assert called["code"] == utils.cv2.COLOR_BGR2GRAY   # controllo codice
    assert out.dtype == np.float32
    assert out.shape == (3, 5)
    assert np.all(out == 42.0)


def test_to_gray_f32_bgra_uses_cvtColor(monkeypatch):
    bgra = np.zeros((4, 6, 4), dtype=np.uint8)  # 4 canali (usa cv2)

    monkeypatch.setattr(utils.cv2, "cvtColor", fake_cvtColor)

    out = utils.to_gray_f32(bgra, normalize=True)

    assert called["code"] == utils.cv2.COLOR_BGRA2GRAY  # controllo codice
    assert out.dtype == np.float32
    assert out.shape == (4, 6)
    assert np.isclose(out.max(), 1.0)


def test_to_gray_f32_unsupported_raises():
    bad = np.zeros((2, 3, 2), dtype=np.uint8)   # converte solo immagini BGR/BGRA o mono in grayscale float32

    with pytest.raises(ValueError, match="Formato immagine non supportato"):
        _ = utils.to_gray_f32(bad, normalize=False)