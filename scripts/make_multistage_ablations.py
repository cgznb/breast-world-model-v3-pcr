"""Emit strict, complete configs for implemented ablations, never run them."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import yaml
from responsewm.config import load_config,from_dict


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="configs/ispy2_multistage_native_v2.yaml")
    p.add_argument("--output",default="configs/ablations_v2")
    args=p.parse_args()
    base=load_config(args.base).to_dict()
    changes={"no_generated_future_readout":("network","use_future",False),
             "no_multiscale_coupling":("network","coupling",False),
             "reencoded_image_state":("network","readout_source","reencode"),
             "no_deep_jepa":("multistage","deep_jepa_mix",0.),
             "no_distribution":("loss","energy",0.),
             "no_grounding":("loss","grounding",0.),
             "all_edges_reference":("multistage","edge_sampling","all_edges"),
             "no_assimilation_reconstruction":("multistage","assimilation_reconstruction",0.)}
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    for name,(group,key,value) in changes.items():
        config=copy.deepcopy(base);config[group][key]=value
        from_dict(config)
        (root/(name+".yaml")).write_text(yaml.safe_dump(config,sort_keys=False))
        print(name)


if __name__=="__main__":
    main()
