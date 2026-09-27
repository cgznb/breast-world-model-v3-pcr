import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import METRIC_KEYS, compute_metrics
from src.data import TABULAR_FEATURE_NAMES, build_tabular_full, load_ids


def build_xy(meta_by_pid, ids):
    X = np.stack([build_tabular_full(meta_by_pid[str(p)]) for p in ids])
    y = np.array([float(meta_by_pid[str(p)]["pCR"]) for p in ids], dtype=np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", required=True, help="enriched metadata CSV")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--output-dir", default="results/clinical")
    args = ap.parse_args()
    if args.n_seeds <= 0:
        raise SystemExit("--n-seeds must be positive")
    os.makedirs(args.output_dir, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    if "pid" not in meta.columns or "pCR" not in meta.columns:
        raise SystemExit("metadata must contain pid and pCR columns")
    if meta["pid"].astype(str).duplicated().any():
        raise SystemExit("metadata contains duplicate patient IDs")
    meta_by_pid = {str(r["pid"]): r for _, r in meta.iterrows()}
    split_ids = {
        split: load_ids(os.path.join(args.splits_dir, f"{split}_ids.txt"))
        for split in ("train", "val", "test")
    }
    for split, ids in split_ids.items():
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{split} split contains duplicate patient IDs")
        missing = set(ids) - set(meta_by_pid)
        if missing:
            raise SystemExit(f"{split} split contains patients missing from metadata")
    if any(
        set(split_ids[left]) & set(split_ids[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise SystemExit("train, validation, and test patient splits overlap")
    tr, va, te = split_ids["train"], split_ids["val"], split_ids["test"]
    Xtr, ytr = build_xy(meta_by_pid, tr)
    Xte, yte = build_xy(meta_by_pid, te)
    split_labels = np.asarray(
        [float(meta_by_pid[pid]["pCR"]) for ids in split_ids.values() for pid in ids]
    )
    if not np.isfinite(split_labels).all() or not set(np.unique(split_labels)).issubset(
        {0.0, 1.0}
    ):
        raise SystemExit("pCR labels must be binary")
    print(f"train={len(ytr)} test={len(yte)} | pCR={ytr.mean():.3f}/{yte.mean():.3f} "
          f"| clinical_dim={Xtr.shape[1]}")

    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    per = {k: [] for k in METRIC_KEYS}
    seed_preds = {}
    seed_models = {}
    for seed in seeds:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        seed_preds[seed] = p
        seed_models[seed] = {
            "classes": clf.classes_.tolist(),
            "coef": clf.coef_.astype(float).tolist(),
            "intercept": clf.intercept_.astype(float).tolist(),
            "n_features_in": int(clf.n_features_in_),
            "feature_names": list(TABULAR_FEATURE_NAMES),
            "C": float(clf.C),
            "solver": str(clf.solver),
            "class_weight": clf.class_weight,
            "max_iter": int(clf.max_iter),
        }
        for k, v in compute_metrics(yte, p).items():
            per[k].append(v)

    print(f"\n{'clinical | LR':24s}" + "".join(k.upper().rjust(14) for k in METRIC_KEYS))
    cells = "".join(f"{np.nanmean(per[k]):.3f}±{np.nanstd(per[k]):.3f}".rjust(14) for k in METRIC_KEYS)
    print(f"{'clinical | LR':24s}{cells}")

    for seed, prob in seed_preds.items():
        output = Path(args.output_dir)
        npz_path = output / f"preds_seed{seed}.npz"
        temporary = npz_path.with_suffix(npz_path.suffix + f".tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                test_patient_ids=np.asarray(te),
                model_classes=np.asarray(seed_models[seed]["classes"]),
                model_coef=np.asarray(seed_models[seed]["coef"]),
                model_intercept=np.asarray(seed_models[seed]["intercept"]),
                **{"clinical | LR::test_y": yte, "clinical | LR::test_prob": prob},
            )
        os.replace(temporary, npz_path)
        frame = pd.DataFrame(
            {
                "patient_id": te,
                "label": yte.astype(int),
                "probability": prob,
                "seed": seed,
                "split": "test",
            }
        )
        csv_path = output / f"test_predictions_seed{seed}.csv"
        temporary = csv_path.with_suffix(csv_path.suffix + f".tmp.{os.getpid()}")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, csv_path)
    summary = {
        "model": "clinical_logistic_regression",
        "threshold": 0.5,
        "train_patients": len(ytr),
        "validation_patients": len(va),
        "test_patients": len(yte),
        "clinical_dim": int(Xtr.shape[1]),
        "seeds": seeds,
        "models": {str(seed): model for seed, model in seed_models.items()},
        "inputs": {
            "metadata_csv": str(Path(args.metadata_csv).expanduser().resolve()),
            "splits_dir": str(Path(args.splits_dir).expanduser().resolve()),
        },
        "metrics": {
            key: {"mean": float(np.nanmean(per[key])), "std": float(np.nanstd(per[key]))}
            for key in METRIC_KEYS
        },
    }
    summary_path = Path(args.output_dir) / "summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, summary_path)
    print(f"\nsaved per-seed predictions to {args.output_dir}/preds_seed*.npz")
    print("this is the imaging-free floor: any encoder cell must beat it to justify the imaging.")


if __name__ == "__main__":
    main()
