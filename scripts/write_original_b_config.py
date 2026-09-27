"""Create a new opt-in configuration; never edit an old run's configuration."""
import argparse
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
p.add_argument('--base-config',required=True)
p.add_argument('--a-checkpoint',required=True)
p.add_argument('--output-root',required=True)
p.add_argument('--output-config',required=True)
a=p.parse_args()
base=Path(a.base_config).resolve()
cfg=yaml.safe_load(base.read_text())
if cfg.get('schema') != 'three_phase_symmflow_all_pairs_v1':
    raise ValueError('Expected the original formal three-phase config')
ckpt=Path(a.a_checkpoint).resolve()
if not ckpt.is_file():
    raise FileNotFoundError(ckpt)
out=Path(a.output_config).resolve()
if out.exists():
    raise FileExistsError(out)
newroot=Path(a.output_root).resolve()
if newroot.exists() and any(newroot.iterdir()):
    raise ValueError('Use an empty NEW experiment output directory')
cfg['observation_checkpoint']=str(ckpt)
cfg['output_root']=str(newroot)
# Existing preparation-reuse settings remain subject to the original code's checks.
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
print(out)
