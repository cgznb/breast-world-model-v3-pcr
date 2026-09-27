import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tabular import DRUG_COLS, arm_column_row


def numid(pid):
    m = re.search(r"(\d+)$", str(pid))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="BreastDCEDL_ISPY2_metadata_longitudinal.csv")
    ap.add_argument("--ispy2-xlsx", required=True, help="ISPY2-metadata.xlsx (Arm, MP)")
    ap.add_argument("--out", required=True, help="output enriched CSV path")
    ap.add_argument("--pid-col", default="pid")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    xl = pd.read_excel(args.ispy2_xlsx)

    arm_mp = {}
    for _, r in xl.iterrows():
        try:
            arm_mp[int(r["Patient_ID"])] = (r.get("Arm"), r.get("MP"))
        except (ValueError, TypeError):
            continue

    nid = df[args.pid_col].map(numid)
    df["Arm"] = nid.map(lambda k: arm_mp.get(k, (None, None))[0])
    df["MP"] = nid.map(lambda k: arm_mp.get(k, (None, None))[1])

    n_arm = int(df["Arm"].notna().sum())
    n_mp = int(df["MP"].notna().sum())
    print(f"rows={len(df)}  Arm matched={n_arm}/{len(df)}  MP matched={n_mp}/{len(df)}")

    # MATERIALIZE the 12 binary drug features as columns (preprocess, not in-memory at train)
    drug_df = pd.DataFrame([arm_column_row(a) for a in df["Arm"]], index=df.index)
    df = pd.concat([df, drug_df], axis=1)
    arms = sorted({a for a in df["Arm"].dropna().unique()})
    print(f"unique arms={len(arms)}  drug columns={len(DRUG_COLS)}")
    print(df[DRUG_COLS].sum().astype(int).to_string())

    # segmentation-free clinical vector: drop the tumor-volume (FTV) columns (never used).
    tumvol = [f"tum_vol_T{t}" for t in range(4)]
    dropped = [c for c in tumvol if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        print(f"dropped {len(dropped)} tumor-volume columns (segmentation-free clinical vector)")

    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  (added Arm, MP, {len(DRUG_COLS)} drug_* columns; tum_vol dropped)")


if __name__ == "__main__":
    main()
