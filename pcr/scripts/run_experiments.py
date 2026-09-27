import argparse
import copy
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tdn import TDN, T0Head
from src.metrics import compute_metrics
from src.data import TABULAR_FEATURE_NAMES, load_all_splits, load_ids
from src.mewm_data import create_registered_source, patient_key, registered_source_contract

from src.temporal import truncate_temporal_splits
MODELS = {"tdn": TDN, "head": T0Head}
CELLS = [("@T0", "head", False), ("@T0+tab", "head", True),
         ("+temporal", "tdn", False), ("+tab+temporal", "tdn", True)]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_prior(data, seed):
    """Fit the logistic-regression clinical prior on the training split; the TDN adds a foundation-
    imaging residual on top (logit = prior + alpha * TDN)."""
    def feats(split):
        return split["clinical"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
        clf.fit(feats(data["train"]), data["train"]["labels"])
    out = {}
    for s in ("train", "val", "test"):
        lg = clf.decision_function(feats(data[s])).astype(np.float32)
        out[s] = np.clip(np.nan_to_num(lg, nan=0.0, posinf=30.0, neginf=-30.0), -30, 30)
    state = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "classes": clf.classes_.tolist(),
        "coef": clf.coef_.astype(float).tolist(),
        "intercept": clf.intercept_.astype(float).tolist(),
        "n_features_in": int(clf.n_features_in_),
        "feature_names": (
            list(TABULAR_FEATURE_NAMES)
            if int(clf.n_features_in_) == len(TABULAR_FEATURE_NAMES)
            else [f"feature_{index}" for index in range(int(clf.n_features_in_))]
        ),
        "C": float(clf.C),
        "solver": str(clf.solver),
        "class_weight": clf.class_weight,
        "max_iter": int(clf.max_iter),
    }
    return out, state


def _loader(split, prior, batch_size, shuffle):
    n = len(split["labels"])
    prior_arr = prior if prior is not None else np.zeros(n, dtype=np.float32)
    ds = TensorDataset(
        torch.tensor(split["embs"]),
        torch.tensor(split["masks"]),
        torch.tensor(split["clinical"]),
        torch.tensor(split["labels"]),
        torch.tensor(prior_arr),
        torch.tensor(split["days"]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_eval(ds_cfg, data, priors, seed, device):
    set_seed(seed)
    bs = ds_cfg["batch_size"]
    train_loader = _loader(data["train"], priors["train"] if priors else None, bs, True)
    val_loader = _loader(data["val"], priors["val"] if priors else None,
                         len(data["val"]["labels"]), False)
    test_loader = _loader(data["test"], priors["test"] if priors else None,
                          len(data["test"]["labels"]), False)

    n_pos = float((data["train"]["labels"] == 1).sum())
    n_neg = float((data["train"]["labels"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    is_tdn = ds_cfg.get("model_type") == "tdn"
    model = MODELS[ds_cfg.get("model_type", "head")]({"downstream": ds_cfg}).to(device)
    optimizer = Adam(model.parameters(), lr=ds_cfg["lr"], weight_decay=ds_cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ds_cfg["epochs"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def call(emb, mask, clin, pl, dys):
        if is_tdn:
            return model(emb, mask, clin, days=dys, prior_logit=pl)
        return model(emb, mask, clin, prior_logit=pl)

    def forward_all(loader):
        model.eval()
        with torch.no_grad():
            emb, mask, clin, lab, pl, dys = next(iter(loader))
            logits = call(emb.to(device), mask.to(device), clin.to(device),
                          pl.to(device), dys.to(device))
            return lab.numpy(), torch.sigmoid(logits).cpu().numpy()

    best_val, best_state, best_epoch, patience = -1.0, None, -1, 0
    epochs_trained = 0
    for epoch in range(ds_cfg["epochs"]):
        epochs_trained = epoch + 1
        model.train()
        epoch_loss = 0.0
        for emb, mask, clin, lab, pl, dys in train_loader:
            emb, mask, clin = emb.to(device), mask.to(device), clin.to(device)
            lab, pl, dys = lab.to(device), pl.to(device), dys.to(device)
            optimizer.zero_grad()
            logits = call(emb, mask, clin, pl, dys)
            loss = loss_fn(logits, lab)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gc = ds_cfg.get("grad_clip", 0)
            if gc and gc > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gc)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        val_y, val_p = forward_all(val_loader)
        finite = bool(np.isfinite(val_p).all())
        try:
            val_auroc = roc_auc_score(val_y, val_p) if finite else -1.0
        except ValueError:
            val_auroc = 0.5
        if finite and val_auroc > best_val:
            best_val = val_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= ds_cfg["patience"]:
            break
        floor = ds_cfg.get("train_loss_floor", 0)
        if floor and (epoch_loss / max(1, len(train_loader))) < floor:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        print("  WARN: validation never produced finite AUROC (training diverged every epoch)")
    val_y, val_p = forward_all(val_loader)
    test_y, test_p = forward_all(test_loader)
    return {
        "val_y": val_y,
        "val_prob": val_p,
        "test_y": test_y,
        "test_prob": test_p,
        "_model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "_selection": {
            "criterion": "validation_auroc",
            "best_epoch_zero_based": best_epoch,
            "best_validation_auroc": float(best_val),
            "epochs_trained": epochs_trained,
        },
        "_effective_config": copy.deepcopy(ds_cfg),
    }


def run_cell(cfg, data, model_type, use_tabular, seed, device):
    """Build ds_cfg for one cell, optionally drop the clinical vector, fit the prior (the
    +tab+temporal cell only), then train and evaluate."""
    dim = int(data["train"]["dim"])
    if use_tabular:
        splits = {s: data[s] for s in ("train", "val", "test")}
        clin_dim = int(data["train"]["clinical"].shape[-1])
    else:
        splits = {s: {**data[s], "clinical": np.zeros((len(data[s]["labels"]), 0), np.float32)}
                  for s in ("train", "val", "test")}
        clin_dim = 0

    ds_cfg = copy.deepcopy(cfg["downstream"])
    ds_cfg["model_type"] = model_type
    ds_cfg["input_dim"] = dim
    ds_cfg["clinical_dim"] = clin_dim
    if model_type == "head" or not use_tabular:
        ds_cfg["use_prior"] = False                 # the clinical prior is the +tab+temporal cell only

    # early prediction keeps the patient but masks every later token.
    k = int(ds_cfg.get("max_tp", 4))
    if k < 4:
        splits = truncate_temporal_splits(splits, k)

    if ds_cfg.get("use_prior", False):
        priors, prior_state = fit_prior(splits, seed)
    else:
        priors, prior_state = None, None
    result = train_eval(ds_cfg, splits, priors, seed, device)
    result["_clinical_prior"] = prior_state
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--global-dir", "--encoder-dir", dest="global_dir", required=True,
                   help="encoder embeddings: dir OR .zip with {pid}/{pid}_T{t}.pt "
                        "(emb_global for 3D Pillar-0, emb_vit2d for the 2D baseline)")
    p.add_argument("--metadata-csv", required=True, help="enriched metadata CSV")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/experiments")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a downstream config key (repeatable), e.g. "
                        "--set max_tp=2 --set sig_dim=64")
    return p.parse_args()


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_npz(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


def _resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return device


def _validate_mewm_embedding_store(cfg, global_dir, metadata_csv, splits_dir):
    adapter = cfg.get("mewm_adapter")
    if not isinstance(adapter, dict):
        return
    root = Path(global_dir)
    summary_path = root / "extraction_summary.json"
    contract_path = root / "extraction_contract.json"
    if not summary_path.is_file() or not contract_path.is_file():
        raise SystemExit("MeWM Pillar embeddings are not extracted with an auditable contract")
    try:
        summary = json.loads(summary_path.read_text())
        contract = json.loads(contract_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SystemExit("MeWM Pillar extraction metadata is unreadable") from None
    expected_dim = int(adapter.get("embedding_dim", 1152))
    expected_splits = {
        split: load_ids(Path(splits_dir) / f"{split}_ids.txt")
        for split in ("train", "val", "test")
    }
    metadata = pd.read_csv(metadata_csv)
    if "pid" not in metadata.columns or "pCR" not in metadata.columns:
        raise SystemExit("MeWM adapted metadata is missing pid or pCR")
    expected_ids = [
        pid for split in ("train", "val", "test") for pid in expected_splits[split]
    ]
    try:
        metadata_by_key = {}
        for _, row in metadata.iterrows():
            key = patient_key(row["pid"])
            if key in metadata_by_key:
                raise ValueError("duplicate patient identities")
            metadata_by_key[key] = row
        expected_keys = [patient_key(pid) for pid in expected_ids]
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"MeWM adapted cohort identities are invalid: {error}") from None
    if len(expected_keys) != len(set(expected_keys)):
        raise SystemExit("MeWM adapted patient splits overlap or contain duplicates")
    if set(expected_keys) != set(metadata_by_key):
        raise SystemExit("MeWM adapted metadata and patient splits do not match")

    try:
        source = create_registered_source(adapter)
        source_contract = registered_source_contract(adapter)
    except Exception as error:
        raise SystemExit(
            f"MeWM registered source cannot be reconstructed from current inputs: {error}"
        ) from None

    current_labels = {}
    current_visits = []
    current_phase_counts = {}
    for pid, key in zip(expected_ids, expected_keys):
        row = metadata_by_key[key]
        try:
            label = float(row["pCR"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise SystemExit(f"MeWM adapted metadata has an invalid pCR label for {pid}") from None
        if not np.isfinite(label) or label not in {0.0, 1.0}:
            raise SystemExit(f"MeWM adapted metadata has an invalid pCR label for {pid}")
        current_labels[str(pid)] = int(label)
        tokens = [
            token.strip()
            for token in str(row.get("registered_timepoints", "")).split(";")
            if token.strip()
        ]
        if not tokens or any(len(token) != 2 or token[0] != "T" for token in tokens):
            raise SystemExit("MeWM adapted metadata has invalid registered_timepoints")
        try:
            timepoints = [int(token[1]) for token in tokens]
        except ValueError:
            raise SystemExit("MeWM adapted metadata has invalid registered_timepoints") from None
        if (
            len(timepoints) != len(set(timepoints))
            or any(timepoint not in range(4) for timepoint in timepoints)
        ):
            raise SystemExit("MeWM adapted metadata has invalid registered_timepoints")
        for timepoint in timepoints:
            try:
                members, policy = source.phase_selection(pid, timepoint, row)
            except Exception as error:
                raise SystemExit(
                    f"MeWM registered phase selection failed for {pid} T{timepoint}: {error}"
                ) from None
            if members is None or policy is None:
                raise SystemExit(
                    f"MeWM registered source is missing selected visit {pid} T{timepoint}"
                )
            visit = source.visits.get((key, timepoint))
            if visit is None:
                raise SystemExit(
                    f"MeWM registered source is missing selected visit {pid} T{timepoint}"
                )
            current_phase_counts[policy] = current_phase_counts.get(policy, 0) + 1
            current_visits.append(
                {
                    "patient_id": str(pid),
                    "timepoint": int(timepoint),
                    "phase_paths": list(members),
                    "phase_selection": policy,
                    "registered_spacing_zyx": list(visit.spacing_zyx),
                    "registered_shape_zyx": list(visit.shape_zyx),
                }
            )
    required_policy = adapter.get("required_phase_selection")
    if (
        summary.get("complete") is not True
        or summary.get("contract") != contract_path.name
        or int(summary.get("expected_visits", -1))
        != int(summary.get("processed_visits", 0))
        + int(summary.get("validated_existing_visits", 0))
        or int(summary.get("embedding_dim", -1)) != expected_dim
        or int(contract.get("expected_visits", -1))
        != int(summary.get("expected_visits", -2))
        or contract.get("model") != "YalaLab/Pillar0-BreastMRI"
        or contract.get("input_mode") != adapter.get("input_mode")
        or int(contract.get("embedding_dim", -1)) != expected_dim
        or contract.get("model_revision") != str(adapter.get("model_revision", "main"))
        or any(
            contract.get(key) != value for key, value in source_contract.items()
        )
        or contract.get("cohort_dir")
        != str(Path(metadata_csv).expanduser().resolve().parent)
        or contract.get("target_shape_hwd") != list(adapter.get("target_shape_hwd", []))
        or contract.get("target_spacing_zyx")
        != [float(value) for value in adapter.get("target_spacing_zyx", [])]
        or contract.get("metadata_csv")
        != str(Path(metadata_csv).expanduser().resolve())
        or contract.get("splits") != expected_splits
        or contract.get("labels") != current_labels
        or contract.get("visits") != current_visits
        or int(contract.get("expected_visits", -1)) != len(current_visits)
        or contract.get("phase_selection_counts") != current_phase_counts
        or (
            required_policy
            and set(current_phase_counts) != {str(required_policy)}
        )
    ):
        raise SystemExit("MeWM Pillar extraction metadata does not match the experiment config")


def save_cell_artifacts(output_dir, name, predictions, patient_ids, seed):
    """Persist an auditable checkpoint, patient predictions, and fixed-threshold metrics."""
    cell_dir = Path(output_dir) / f"seed_{seed}" / _slug(name)
    checkpoint = {
        "cell": name,
        "seed": int(seed),
        "model_state": predictions["_model_state"],
        "selection": predictions["_selection"],
        "effective_config": predictions["_effective_config"],
        "clinical_prior": predictions.get("_clinical_prior"),
        "run_context": predictions.get("_run_context"),
    }
    _atomic_torch_save(cell_dir / "best.pt", checkpoint)

    summary = {
        "cell": name,
        "seed": int(seed),
        "threshold": 0.5,
        "selection": predictions["_selection"],
        "clinical_prior": predictions.get("_clinical_prior"),
        "run_context": predictions.get("_run_context"),
        "metrics": {},
    }
    for split in ("val", "test"):
        labels = predictions[f"{split}_y"]
        probabilities = predictions[f"{split}_prob"]
        pids = list(patient_ids[split])
        if not (len(pids) == len(labels) == len(probabilities)):
            raise RuntimeError(f"{split} patient predictions are misaligned")
        frame = pd.DataFrame(
            {
                "patient_id": pids,
                "label": labels.astype(int),
                "probability": probabilities,
                "seed": int(seed),
                "split": split,
            }
        )
        csv_path = cell_dir / f"{split}_predictions.csv"
        temporary = csv_path.with_suffix(csv_path.suffix + f".tmp.{os.getpid()}")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, csv_path)
        summary["metrics"][split] = compute_metrics(labels, probabilities, threshold=0.5)
    _atomic_text(cell_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = _resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for ov in args.overrides:                      # CLI overrides of downstream config keys
        if "=" not in ov:
            raise SystemExit(f"--set expects KEY=VALUE, got {ov!r}")
        k, v = ov.split("=", 1)
        cfg["downstream"][k] = yaml.safe_load(v)   # parses bool / int / float / str
    if args.overrides:
        print("overrides:", " ".join(args.overrides))
    run_context = {
        "seed": int(args.seed),
        "requested_device": args.device,
        "resolved_device": str(device),
        "global_dir": str(Path(args.global_dir).expanduser().resolve()),
        "metadata_csv": str(Path(args.metadata_csv).expanduser().resolve()),
        "splits_dir": str(Path(args.splits_dir).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "torch_version": str(torch.__version__),
    }
    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg["runtime"] = run_context
    ids = (load_ids(os.path.join(args.splits_dir, "train_ids.txt")),
           load_ids(os.path.join(args.splits_dir, "val_ids.txt")),
           load_ids(os.path.join(args.splits_dir, "test_ids.txt")))
    print(f"Splits: train={len(ids[0])} val={len(ids[1])} test={len(ids[2])} | seed={args.seed}")

    _validate_mewm_embedding_store(
        cfg, args.global_dir, args.metadata_csv, args.splits_dir
    )
    data = load_all_splits(args.global_dir, args.metadata_csv, *ids)
    adapter = cfg.get("mewm_adapter")
    if isinstance(adapter, dict) and data["train"]["dim"] != int(
        adapter.get("embedding_dim", 1152)
    ):
        raise SystemExit("loaded embedding dimension does not match the MeWM config")
    _atomic_text(
        Path(args.output_dir) / f"resolved_config_seed{args.seed}.yaml",
        yaml.safe_dump(resolved_cfg, sort_keys=True),
    )
    print(f"  embedding_dim={data['train']['dim']} tabular_dim={data['train']['clinical'].shape[-1]}")

    results = {}
    print(f"\n{'config':22s} {'val':>6} {'test':>6} {'bacc':>6} {'prauc':>6}")
    for cell, mt, use_tab in CELLS:
        name = f"global | {cell}"
        preds = run_cell(cfg, data, mt, use_tab, args.seed, device)
        preds["_run_context"] = run_context
        for split in ("val", "test"):
            p = preds[f"{split}_prob"]
            if not np.isfinite(p).all():
                raise RuntimeError(f"{name} produced non-finite {split} probabilities")
        m = compute_metrics(preds["test_y"], preds["test_prob"])
        try:
            vauc = roc_auc_score(preds["val_y"], preds["val_prob"])
        except ValueError:
            vauc = float("nan")
        save_cell_artifacts(
            args.output_dir,
            name,
            preds,
            {"val": data["val"]["pids"], "test": data["test"]["pids"]},
            args.seed,
        )
        results[name] = preds
        print(f"{name:22s} {vauc:6.3f} {m['auroc']:6.3f} {m['bacc']:6.3f} {m['prauc']:6.3f}")

    prediction_keys = ("val_y", "val_prob", "test_y", "test_prob")
    npz_payload = {
        "val_patient_ids": np.asarray(data["val"]["pids"]),
        "test_patient_ids": np.asarray(data["test"]["pids"]),
        **{
            f"{name}::{key}": predictions[key]
            for name, predictions in results.items()
            for key in prediction_keys
        },
    }
    _atomic_npz(
        Path(args.output_dir) / f"preds_seed{args.seed}.npz",
        npz_payload,
    )
    print(f"\nPredictions saved to {args.output_dir}/preds_seed{args.seed}.npz")


if __name__ == "__main__":
    main()
