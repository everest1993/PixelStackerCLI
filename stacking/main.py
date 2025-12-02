"""
Punto di ingresso da riga di comando per eseguire la pipeline logica in modo automatico.
Viene utilizzata la libreria argparse per leggere gli argomenti forniti da terminale.

Il blocco if __name__ == "__main__": main() fa sì che lo script venga eseguito solo se lanciato direttamente da
terminale, non se importato come modulo in un altro programma.
"""
import logging
import argparse
import os

from pathlib import Path
from stacking.pipeline import NoiseStackingPipeline
from stacking.pipeline import FocusPipeline
from stacking.pipeline import ExposurePipeline


# evita qualunque accesso a internet da HF/transformers
def _set_runtime_env() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("DISABLE_TELEMETRY", "1")

    # evita warning e sovraccarico
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # limita i thread BLAS/OpenMP nel processo principale
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # (opzionale) evita uso non voluto di MPS/CUDA (su Mac evita crash se MPS non disponibile)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


logging.basicConfig(
    level=logging.DEBUG,  # mostra i logging DEBUG
    format="%(levelname)s: %(message)s"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PixelStacker CLI – stacking pipelines"
    )

    sub = parser.add_subparsers(dest="mode", required=True, help="select pipeline")

    def add_common_args(p: argparse.ArgumentParser):
        p.add_argument("-o", "--output", required=True, help="output file (TIF)")
        p.add_argument("inputs", nargs="+", help="input images (2 min)")
        p.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging")

    # noise pipeline
    p_noise = sub.add_parser("noise", help="Noise stacking (sigma clipping)")
    add_common_args(p_noise)
    p_noise.add_argument("--ref-idx", type=int, default=None, help="reference img idx (default: central)")
    p_noise.add_argument("--sigma-low", type=float, default=None, help="lower sigma clipping value")
    p_noise.add_argument("--sigma-hi", type=float, default=None, help="upper sigma clipping value")
    p_noise.add_argument("--iterations", type=int, default=None, help="sigma clipping iterations")

    # focus pipeline
    p_focus = sub.add_parser("focus", help="Focus stacking")
    add_common_args(p_focus)

    # exposure pipeline
    p_expo = sub.add_parser("exposure", help="Exposure stacking / HDR")
    add_common_args(p_expo)

    return parser


def make_pipeline(args):
    if args.mode == "noise":
        return NoiseStackingPipeline(
            ref_idx=args.ref_idx,
            sigma_low=args.sigma_low,
            sigma_hi=args.sigma_hi,
            iterations=args.iterations
        )
    elif args.mode == "focus":
        return FocusPipeline()
    elif args.mode == "exposure":
        return ExposurePipeline()
    else:
        raise ValueError(f"Unknown pipeline: {args.mode}")


def main():
    # importa librerie pesanti solo dopo che l'ambiente runtime è stato impostato
    import cv2
    from stacking.io import read_imgs

    parser = build_parser()
    args = parser.parse_args()

    output = Path(args.output)
    inputs = [Path(p) for p in args.inputs]

    imgs = read_imgs(inputs)
    if len(imgs) < 2:
        raise ValueError("Minimum images number = 2.")

    h, w = imgs[0].shape[:2]
    if not all(im.shape[:2] == (h, w) for im in imgs):
        raise ValueError("Each image must have the same size.")

    pipeline = make_pipeline(args)  # costruzione pipeline
    final = pipeline.run(imgs)

    out_path = output.with_suffix(".tif")

    if not cv2.imwrite(str(out_path), final):
        raise RuntimeError(f"Impossible to save file: {out_path}")
    print(f"Completed: {out_path}")


"""
Blocco di codice che serve per garantire che i worker dei processi si avviino in modo sicuro e pulito
"""
if __name__ == "__main__":  # entrypoint
    import multiprocessing as mp

    # imposta variabili d'ambiente prima degli import pesanti
    _set_runtime_env()

    mp.freeze_support()
    try:
        """
        spawn è default su Windows e compatibile con macOs e Linux
        Avvia un nuovo interprete python e ricarica i moduli
        """
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()