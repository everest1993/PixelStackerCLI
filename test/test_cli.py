from types import SimpleNamespace
from stacking.pipeline import NoiseStackingPipeline, FocusPipeline, ExposurePipeline

import numpy as np
import pytest
import stacking.cli as cli

class FakeNoisePipe:
    def __init__(self, **kwargs): pass
    def run(self, imgs): return _fake_img()


def _fake_img(h=10, w=10, c=3):
    return np.ones((h, w, c), dtype=np.uint8)   # genera un'immagine come matrice di 1 per i test


# il subparser mode è required=True dunque se viene generato un parser senza mode il processo termina con exit code 2
def test_parser_requires_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# verifica mapping
def test_parser_noise_args_minimal():
    parser = cli.build_parser()
    args = parser.parse_args(["noise", "-o", "out.tif", "img1.png", "img2.png"])
    assert args.mode == "noise"
    assert args.output == "out.tif"
    assert args.inputs == ["img1.png", "img2.png"]


def test_parser_focus_args_minimal():
    parser = cli.build_parser()
    args = parser.parse_args(["focus", "-o", "out", "a.jpg", "b.jpg"])
    assert args.mode == "focus"
    assert args.output == "out"
    assert args.inputs == ["a.jpg", "b.jpg"]


def test_parser_exposure_args_minimal():
    parser = cli.build_parser()
    args = parser.parse_args(["exposure", "-o", "hdr", "a.png", "b.png"])
    assert args.mode == "exposure"
    assert args.output == "hdr"
    assert args.inputs == ["a.png", "b.png"]


# test di binding mode - pipeline
def test_make_pipeline_returns_expected_types():
    args_noise = SimpleNamespace(mode="noise", sigma_low=None, ref_idx=None, sigma_hi=None,
                                 min_keep=None, min_keep_frac=None, iterations=None)
    p = cli.make_pipeline(args_noise)
    assert isinstance(p, NoiseStackingPipeline)

    args_focus = SimpleNamespace(mode="focus")
    p = cli.make_pipeline(args_focus)
    assert isinstance(p, FocusPipeline)

    args_expo = SimpleNamespace(mode="exposure")
    p = cli.make_pipeline(args_expo)
    assert isinstance(p, ExposurePipeline)


def test_make_pipeline_unknown_raises():
    with pytest.raises(ValueError):
        cli.make_pipeline(SimpleNamespace(mode="???"))


def test_main_less_than_two_images_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["script", "noise", "-o", str(tmp_path / "out"), "only_one.png"])

    """
    nel namespace del modulo stacking.cli, esiste un simbolo locale chiamato read_imgs che punta alla funzione definita 
    in stacking/io.py. monkeypatch non chiama stacking.io.read_imgs direttamente, ma il riferimento locale
    """
    monkeypatch.setattr(cli, "read_imgs", lambda paths: [_fake_img()])

    with pytest.raises(ValueError, match=r"Minimum images number\s*=\s*2\."):
        cli.main()


def test_main_mismatched_sizes_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["script", "noise", "-o", str(tmp_path / "out"), "a.png", "b.png"])
    monkeypatch.setattr("stacking.cli.read_imgs",
                        lambda paths: [_fake_img(10, 10), _fake_img(9, 10)])     # shape diversa

    with pytest.raises(ValueError, match="same size"):
        cli.main()


def test_main_imwrite_failure_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["prog", "noise", "-o", str(tmp_path / "out"), "a.png", "b.png"])
    monkeypatch.setattr("stacking.cli.read_imgs", lambda paths: [_fake_img(), _fake_img()])
    monkeypatch.setattr("stacking.cli.NoiseStackingPipeline", FakeNoisePipe)
    monkeypatch.setattr("stacking.cli.cv2.imwrite", lambda p, im: False)    # forza il fallimento della scrittura

    with pytest.raises(RuntimeError, match="Impossible to save file"):
        cli.main()