import argparse
import re

import numpy as np
import pandas as pd


def numid(x):
    m = re.search(r"(\d+)$", str(x))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pid-col", default="pid")
    args = ap.parse_args()

    frames = []
    for f in args.manifests:
        frames.append(pd.read_excel(f, usecols=["Study ID", "Study Date"]))
    man = pd.concat(frames, ignore_index=True)
    sid = man["Study ID"].astype(str).str.extract(r"(\d+)_T(\d)")
    man["numid"] = pd.to_numeric(sid[0], errors="coerce")
    man["t"] = pd.to_numeric(sid[1], errors="coerce")
    man["date"] = pd.to_datetime(man["Study Date"], errors="coerce")
    man = man.dropna(subset=["numid", "t", "date"])
    # one date per (patient, timepoint)
    date = man.groupby(["numid", "t"])["date"].min()

    df = pd.read_csv(args.csv)
    nid = df[args.pid_col].map(numid)

    def days(n, t):
        try:
            return int((date[(n, t)] - date[(n, 0)]).days)
        except (KeyError, TypeError):
            return np.nan

    cov = 0
    for t in range(4):
        df[f"days_T{t}"] = [days(n, t) for n in nid]
    cov = int(df["days_T1"].notna().sum())
    print(f"rows={len(df)}  days_T1 matched={cov}/{len(df)}")
    for t in (1, 2, 3):
        v = df[f"days_T{t}"].dropna()
        if len(v):
            print(f"  days_T{t}: n={len(v)} median={v.median():.0f}d "
                  f"IQR=[{v.quantile(.25):.0f},{v.quantile(.75):.0f}]")
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  (+ days_T0..days_T3)")


if __name__ == "__main__":
    main()
