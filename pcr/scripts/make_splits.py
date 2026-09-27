import argparse
import os

import pandas as pd

SPLIT_NAME = {0: "train", 1: "val", 2: "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="BreastDCEDL_ISPY2_metadata[_longitudinal].csv")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--pid-col", default="pid")
    ap.add_argument("--test-col", default="test")
    ap.add_argument("--label-col", default="pCR")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out, exist_ok=True)

    parts = {name: df[df[args.test_col] == code] for code, name in SPLIT_NAME.items()}
    for name in ("train", "val", "test"):
        sub = parts[name]
        pids = sub[args.pid_col].astype(str).tolist()
        path = os.path.join(args.out, f"{name}_ids.txt")
        with open(path, "w") as f:
            f.write("\n".join(pids) + ("\n" if pids else ""))
        pcr = sub[args.label_col].mean() if len(sub) else float("nan")
        print(f"{name:5s}: n={len(pids):4d}  pCR={pcr:.3f}  -> {path}")

    print(f"total={len(df)}")


if __name__ == "__main__":
    main()
