"""cmapss-rul command line: download data, run the benchmark, make figures."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .benchmark import run, write_report
from .data import SUBSETS, download
from .models import MODELS


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cmapss-rul")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--root", default="data/cmapss")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download the four C-MAPSS subsets")
    b = sub.add_parser("benchmark", help="train and evaluate models")
    b.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    b.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    b.add_argument("--window", type=int, default=30)
    b.add_argument("--seeds", type=int, nargs="+", default=[42])
    b.add_argument("--out", default="artifacts/benchmark")
    b.add_argument("--figures", default=None, help="directory for figures (needs matplotlib)")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.cmd == "download":
        print(f"C-MAPSS available in {download(args.root)}")
        return

    results, all_preds = [], {}
    for seed in args.seeds:
        for subset in args.subsets:
            r, preds = run(subset, args.models, args.root, args.window, seed=seed)
            results += r
            all_preds.setdefault(subset, preds)  # figures use the first seed
            for x in r:
                print(
                    f"seed {seed} {subset} {x.model:9s} RMSE={x.rmse:6.2f}  "
                    f"score={x.nasa_score:9.0f}  ({x.seconds:.0f}s)"
                )
    print("\n" + write_report(results, args.out))
    if args.figures:
        from .plots import prediction_figure

        for subset, preds in all_preds.items():
            prediction_figure(preds, subset, Path(args.figures) / f"pred_{subset}.png")


if __name__ == "__main__":
    main()
