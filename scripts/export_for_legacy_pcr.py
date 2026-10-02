#!/usr/bin/env python
"""Export decoded generated visits to the original pcr/run.py encode input shape.

Does not copy/average Pillar features, hallucinate observed visits or transplant
TDN weights. Each future visit retains the same K trajectory ordering.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
with np.load(a.prediction,allow_pickle=False) as data:
    if 'images' not in data: raise ValueError('Run joint.py predict with --codec before image-based independent evaluation')
    images=data['images']
    if images.ndim!=7 or images.shape[0]!=1 or images.shape[3]!=3:
        raise ValueError('Expected one-patient images [1,K,F,3,D,H,W]')
    rows=[]
    for j in range(images.shape[2]):
        path=root/f'future_{j+1:02d}.npz'
        np.savez_compressed(path,images=images[0,:,j])
        rows.append({'future_index':j,'file':path.name,'shape':list(images[0,:,j].shape)})
    (root/'index.json').write_text(json.dumps({'schema':'responsewm_legacy_image_exports_v1','visits':rows,
        'trajectory_alignment':'k indexes the same complete trajectory across all visits',
        'normalization':'Use original VQ-matched physical-spacing and foreground conventions in pcr/run.py encode',
        'classifier_inputs':'Observed visit Pillar features, days, masks, clinical vector and original classifier bundles remain required'},indent=2))
