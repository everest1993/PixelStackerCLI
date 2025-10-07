import numpy as np
import pytest
import stacking.aligners as alignment

def test_find_transform_returns_none_on_flat_images():
    ref = np.zeros((64, 64), dtype=np.float32)  # matrice di zeri (immagine piatta)
    tgt = np.zeros((64, 64), dtype=np.float32)

    H = alignment.find_transform(tgt_gray=tgt, ref_gray=ref)    # se non individua coordinate o il numero di matches < 4
    assert H is None    # la matrice di trasformazione è nulla


def test_focus_and_exposure_aligners_are_placeholders():
    with pytest.raises(NotImplementedError):
        alignment.FocusAligner().align_all([np.zeros((10, 10, 3), dtype=np.uint8)])

    with pytest.raises(NotImplementedError):
        alignment.ExposureAligner().align_all([np.zeros((10, 10, 3), dtype=np.uint8)])