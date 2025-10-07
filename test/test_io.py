from stacking.io import read_color

import pytest

def test_read_color_non_raw_raises_when_none(monkeypatch, tmp_path):
    p = tmp_path / "broken.jpg"

    monkeypatch.setattr("stacking.io.cv2.imread", lambda *_: None)

    with pytest.raises(RuntimeError, match="Impossibile leggere l'immagine"):
        read_color(p)