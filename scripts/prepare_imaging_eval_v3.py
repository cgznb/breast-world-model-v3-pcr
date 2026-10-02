#!/usr/bin/env python3
"""Prepare the fixed, portable decoded MRI evaluation cohort (no training)."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from responsewm.imaging_evaluation_v3 import prepare_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.manifest, args.data_root, args.output,
                                   count=args.cases, split=args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
