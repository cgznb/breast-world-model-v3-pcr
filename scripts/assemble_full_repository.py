"""Combine a LOCAL git repository with v3 into a full upstream+v3 source ZIP.

Only git-tracked upstream files are considered, and binary patient assets,
credentials and model weights are excluded even when accidentally tracked.
No external download or GitHub credentials are needed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

DENIED_SUFFIXES=('.pt','.pth','.ckpt','.safetensors','.npy','.npz','.nii','.nii.gz','.dcm','.dicom',
                 '.h5','.hdf5','.csv','.tsv','.xlsx','.xls','.parquet','.key','.pem','.zip','.tar','.gz')
DENIED_PARTS={'__pycache__','.git','.venv','node_modules','artifacts','outputs','datasets','checkpoints','logs','wandb'}


def allowed(path):
    p=Path(path)
    name=p.name.lower()
    return not (p.is_absolute() or '..' in p.parts or any(x in DENIED_PARTS for x in p.parts)
                or name.startswith('.env') or name.startswith('id_rsa') or name.startswith('id_ed25519')
                or name in {'paths.local.yaml','credentials.json','secrets.json'} or name.endswith('.local.yaml')
                or name.endswith(DENIED_SUFFIXES))


def assemble(upstream,output,enable_original_b=True):
    repo=Path(upstream).resolve(); output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    listing=subprocess.check_output(['git','-C',str(repo),'ls-files','-z']).decode().split('\0')
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)/'symm-fm-observation-v3-full';dest.mkdir()
        skipped=[]; copied=[]
        for name in filter(None,listing):
            src=repo/name
            if not allowed(name) or src.is_symlink():
                skipped.append(name);continue
            if not src.is_file():
                raise ValueError(f'Tracked file missing: {name}')
            target=dest/name; target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);copied.append(name)
        spec=importlib.util.spec_from_file_location('installer',Path(__file__).with_name('install_into_repo.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.install(dest,enable_original_b)
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
        report={'upstream_commit':commit,'tracked_files_copied':len(copied),'excluded_for_safety':skipped,
                'uncommitted_tracked_edits_included':True,'upstream_all_history_included':False}
        (dest/'OBSERVATION_V3_ASSEMBLY.json').write_text(json.dumps(report,indent=2))
        output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(dest.rglob('*')):
                if file.is_file(): z.write(file,file.relative_to(dest.parent))
    return {'archive':str(output),**report}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local-upstream',required=True);p.add_argument('--output',required=True)
    p.add_argument('--no-original-b-patch',action='store_true');a=p.parse_args()
    print(json.dumps(assemble(a.local_upstream,a.output,not a.no_original_b_patch),indent=2))
