#!/usr/bin/env python
"""Package tracked SOURCE from a local V2 clone + this upgrade, without modifying it.

No data/weights/caches are included. Review your tracked source for PHI before
sharing; extension filtering cannot certify de-identification of source text.
"""
from pathlib import Path
import argparse,subprocess,zipfile,hashlib
p=argparse.ArgumentParser();p.add_argument('--v2-repo',required=True);p.add_argument('--output',required=True)
a=p.parse_args();repo=Path(a.v2_repo).resolve();upgrade=Path(__file__).resolve().parents[1]
allowed={'.py','.md','.yaml','.yml','.json','.toml','.sh','.txt','.cff','.ini','.cfg','.rst','.ipynb'}
# Do not copy notebooks: outputs may contain patient images or identifiers.
allowed.discard('.ipynb')
named={'LICENSE','NOTICE','.gitignore','Dockerfile','Makefile'}
blocked={'__pycache__','.pytest_cache','.git','data','runs','weights','checkpoints','artifacts','results'}
paths=subprocess.check_output(['git','-C',str(repo),'ls-files','-z']).decode().split('\0')
output=Path(a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
    for name in paths:
        path=repo/name
        if not name or not path.is_file() or path.is_symlink() or blocked.intersection(Path(name).parts): continue
        if path.suffix not in allowed and path.name not in named: continue
        if path.stat().st_size>10*1024*1024: raise ValueError(f'Unexpected large tracked source: {name}')
        z.write(path,'breast-world-model-v2-pcr/'+name)
    for path in sorted(upgrade.rglob('*')):
        rel=path.relative_to(upgrade)
        if not path.is_file() or path.is_symlink() or blocked.intersection(rel.parts): continue
        if path.suffix not in allowed and path.name not in named: continue
        z.write(path,'breast-world-model-v2-pcr/joint_upgrade/'+str(rel))
print(output)
