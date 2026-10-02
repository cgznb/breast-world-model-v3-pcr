#!/usr/bin/env python
"""Full-width native CPU forward evidence, not a one-step performance claim."""
import sys,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from responsewm.config import Config
from responsewm.model import ResponseWorldModel
from responsewm.contracts import ForecastInput
from responsewm.io import write_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
cfg=Config();cfg.network.backend='native';cfg.training.device='cpu';cfg.training.precision='fp32'
torch.set_num_threads(2);torch.manual_seed(7)
m=ResponseWorldModel(cfg.validate(),17,8).eval().requires_grad_(False)
z=torch.randn(1,1,24,8,32,32)
inp=ForecastInput(z,torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1),torch.zeros(1,17),
    torch.ones(1,17,dtype=torch.bool),torch.ones(1,1)*90,torch.ones(1,1,dtype=torch.bool),
    torch.zeros(1,1,8),torch.ones(1,1,8,dtype=torch.bool))
start=time.monotonic()
with torch.no_grad(): out=m.forecast(inp,samples=1,steps=1,method='heun',generator=torch.Generator().manual_seed(9))
write_json(a.output,{'engineering_only':True,'clinical_validation':False,'backend':'native','device':'cpu','precision':'fp32',
    'full_production_widths':True,'latent_shape':list(z.shape),'generated_latent_shape':list(out.latent.shape),
    'generated_state_shape':list(out.state.shape),'parameters_total':sum(p.numel() for p in m.parameters()),
    'parameters_by_component':{k:sum(p.numel() for p in getattr(m,k).parameters()) for k in ('encoder','target_encoder','history','velocity','pcr')},
    'seconds':time.monotonic()-start,'finite_outputs':all(bool(torch.isfinite(x).all()) for x in (out.latent,out.state,out.probability)),
    'solver':'heun','steps':1,'note':'Random weights; one-step shape/finite forward test, not a tested one-step inference recipe.'})
