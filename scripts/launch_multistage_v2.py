"""Start the authorized four-stage run in a persistent independent process."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from responsewm.io import write_json,digest,read_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True);p.add_argument("--config",required=True)
    p.add_argument("--output",required=True);p.add_argument("--preflight",required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    preflight=read_json(args.preflight)
    if set(preflight.get("results",{}))!={"representation","flow","readout","joint"}:
        raise ValueError("A/B/C/D real-size preflight required")
    if preflight["manifest_digest"]!=digest(args.manifest):
        raise ValueError("Preflight was performed with different data")
    receipt=output/"launch.json"
    if receipt.exists():
        previous=read_json(receipt)
        try:
            os.kill(previous["pid"],0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Previous training controller is still running")
    command=[sys.executable,"-u",str(root/"joint.py"),"train-v2","--stage","all","--resume",
             "--config",str(Path(args.config).resolve()),"--manifest",str(Path(args.manifest).resolve()),
             "--output",str(output)]
    with (output/"console.log").open("ab",buffering=0) as log:
        process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True)
    info={"pid":process.pid,"process_group":process.pid,"command":command,"cwd":str(root),
          "started_unix":time.time(),"config_sha256":digest(args.config),"manifest_sha256":digest(args.manifest),
          "preflight_sha256":digest(args.preflight),"log":str(output/"console.log")}
    write_json(receipt,info)
    print(info,flush=True)


if __name__=="__main__":
    main()
