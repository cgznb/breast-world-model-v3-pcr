import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mewm_data import (
    INPUT_MODES,
    create_registered_source,
    build_adapted_cohort,
    write_adapted_cohort,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an audited Pillar/TDN cohort from the registered MeWM data."
    )
    parser.add_argument("--config", default="configs/mewm_ispy2_registered.yaml")
    parser.add_argument("--input-mode", choices=INPUT_MODES)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--check-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require every selected phase file to exist (default: true)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    adapter = config.get("mewm_adapter")
    if not isinstance(adapter, dict):
        raise SystemExit("config must contain a mewm_adapter mapping")

    input_mode = args.input_mode or adapter.get("input_mode")
    output_dir = args.output_dir or adapter.get("output_dir")
    if not output_dir:
        raise SystemExit("mewm_adapter.output_dir is required")
    if args.input_mode and not args.output_dir and args.input_mode != adapter.get("input_mode"):
        raise SystemExit("an input-mode override also requires --output-dir")

    source = create_registered_source(adapter, input_mode=input_mode)
    metadata, splits, summary = build_adapted_cohort(
        adapter["metadata_csv"],
        adapter.get("source_splits_dir"),
        source,
        reference_manifest_json=adapter.get("reference_manifest_json"),
        min_timepoints=int(adapter.get("minimum_timepoints", 1)),
        check_files=args.check_files,
    )
    required_policy = adapter.get("required_phase_selection")
    if required_policy and set(summary["phase_selection_counts"]) != {required_policy}:
        raise SystemExit(
            "selected visits do not satisfy mewm_adapter.required_phase_selection"
        )
    write_adapted_cohort(output_dir, metadata, splits, summary)

    print(
        f"mode={summary['input_mode']} patients={summary['patients']} "
        f"checked_files={summary['files_checked']}"
    )
    for split in ("train", "val", "test"):
        row = summary["splits"][split]
        print(
            f"{split:5s}: n={row['patients']} "
            f"non-pCR/pCR={row['pcr_negative']}/{row['pcr_positive']} "
            f"timepoints={row['registered_timepoint_counts']}"
        )
    print(f"phase selection: {summary['phase_selection_counts']}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
