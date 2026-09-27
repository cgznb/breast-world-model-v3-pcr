import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import METRIC_KEYS, compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/experiments")
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="aggregate only these exact seed artifacts instead of every preds_seed*.npz",
    )
    args = ap.parse_args()

    if args.seeds is not None:
        if len(args.seeds) != len(set(args.seeds)):
            raise SystemExit("--seeds must not contain duplicates")
        files = [os.path.join(args.dir, f"preds_seed{seed}.npz") for seed in args.seeds]
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise SystemExit(f"Missing requested prediction artifact: {missing[0]}")
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "preds_seed*.npz")))
    if not files:
        raise SystemExit(f"No preds_seed*.npz in {args.dir}")

    per = defaultdict(lambda: defaultdict(list))   # config -> metric -> [per-seed values]
    order = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        cfgs = [k[:-len("::test_prob")] for k in d.files if k.endswith("::test_prob")]
        for c in cfgs:
            if c not in order:
                order.append(c)
            m = compute_metrics(d[f"{c}::test_y"], d[f"{c}::test_prob"])
            for k in METRIC_KEYS:
                per[c][k].append(m[k])

    print(f"{len(files)} seeds from {args.dir}\n")
    hdr = "config".ljust(34) + "".join(k.upper().rjust(14) for k in METRIC_KEYS)
    print(hdr)
    for c in order:
        cells = "".join(f"{np.nanmean(per[c][k]):.3f}±{np.nanstd(per[c][k]):.3f}".rjust(14)
                        for k in METRIC_KEYS)
        print(f"{c:34s}{cells}")

    base = order[0]
    print(f"\nmean test-AUROC delta vs '{base}':")
    for c in order[1:]:
        delta = np.nanmean(per[c]["auroc"]) - np.nanmean(per[base]["auroc"])
        wins = sum(a > b for a, b in zip(per[c]["auroc"], per[base]["auroc"]))
        print(f"  {c:34s} {delta:+.3f}   (beats baseline in {wins}/{len(files)} seeds)")


if __name__ == "__main__":
    main()
