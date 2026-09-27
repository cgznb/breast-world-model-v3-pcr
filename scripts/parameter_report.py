"""Count instantiated production modules; not a GPU memory benchmark."""
from pathlib import Path
import argparse
import gc
import json
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from symm_observation.config import load_config
from symm_observation.observation_ssl import ObservationPretrainer
from symm_observation.world_model import ThreePhaseWorldModel
from symm_observation.conditioning import ConditionSchema
from symm_observation.utils import write_json


def count(m):
    return {'total_parameters':sum(p.numel() for p in m.parameters()),
            'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad)}


def report(config):
    cfg=load_config(config)
    statistics={'mean':[0.]*24,'std':[1.]*24,'fit_split':'train'}
    a=ObservationPretrainer(cfg,statistics)
    result={'config':str(config),'observation_encoder':count(a.encoder),'A_system':count(a),
            'same_visit_predictor':count(a.predictor),'observation_readout':count(a.readout),
            'input_volume_forward_tested':False,'GPU_memory_measured':False}
    encoder=a.target_encoder
    schema=ConditionSchema.fit([{'split':'train','conditions':{'stage_i':'T0','stage_j':'T1',
                        'treatment_arm':'example_only','interval_verified':False}}])
    try:
        b=ThreePhaseWorldModel(cfg,encoder,schema)
        result['B_system_example_vocabulary']=count(b)
        result['velocity']=count(b.velocity_model)
    except ImportError as e:
        result['B_instantiation_skipped']=str(e)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/cache_native.yaml'));p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    value=report(args.config);write_json(args.output,value);print(json.dumps(value,indent=2))
