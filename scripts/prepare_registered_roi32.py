from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from symm_observation.registered_roi32 import convert_registered_roi32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codec-checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(convert_registered_roi32(args.root, args.output, args.codec_checkpoint), indent=2))
