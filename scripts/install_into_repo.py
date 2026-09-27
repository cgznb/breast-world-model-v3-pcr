"""Install the complete v3 workflow without replacing historical upstream runs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = '    model = ThreePhaseSymmFlow(training.records, source=cfg["symm_repo"]).to("cuda:0")'
NEW = '''    # observation_v3: source-only semantic conditioning, opt-in per new run.
    if cfg.get("observation_checkpoint"):
        from .three_phase_observation_bridge import ObservationConditionedSymmFlow
        model = ObservationConditionedSymmFlow(
            training.records, source=cfg["symm_repo"],
            observation_checkpoint=cfg["observation_checkpoint"], statistics=statistics,
            codec_checkpoint=cfg["codec_checkpoint"],
            patient_splits={v["patient_id"]: v["split"] for v in inventory["views"]},
        ).to("cuda:0")
    else:
        model = ThreePhaseSymmFlow(training.records, source=cfg["symm_repo"]).to("cuda:0")'''


def install(repo, enable_original_b=False):
    repo = Path(repo).resolve()
    original = repo/'workflows/first_post_three_phase/mewm_ispy2'
    if not (original/'three_phase_symmflow.py').is_file():
        raise ValueError('Expected cgznb/symm-fm with its original three-phase workflow')
    destination = repo/'workflows/observation_v3'
    if destination.exists():
        raise FileExistsError('observation_v3 already exists; installation will not overwrite it')
    training = original/'three_phase_all_pairs_training.py'
    text = training.read_text(encoding='utf-8') if enable_original_b else ''
    # Validate patches before any file changes. An upstream refactor is not guessed.
    if enable_original_b and (text.count(OLD) != 1 or 'observation_v3:' in text):
        raise ValueError('Original model-construction block changed or already patched; inspect manually')
    launcher = repo/'run_observation_v3.py'
    bridge = original/'three_phase_observation_bridge.py'
    backup = training.with_suffix('.py.observation_v3_backup')
    if launcher.exists() or bridge.exists() or (enable_original_b and backup.exists()):
        raise FileExistsError('An integration destination already exists')
    files = []
    try:
        for folder in ('src','configs','scripts','docs','licenses'):
            shutil.copytree(ROOT/folder, destination/folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('world.py','pyproject.toml','README.md','LICENSE','requirements.txt'):
            if (ROOT/name).exists():
                shutil.copy2(ROOT/name, destination/name)
        launcher.write_text("from pathlib import Path\nimport runpy\nrunpy.run_path(str(Path(__file__).resolve().parent/'workflows/observation_v3/world.py'), run_name='__main__')\n", encoding='utf-8')
        files = [str(destination), str(launcher)]
        if enable_original_b:
            shutil.copy2(training, backup)
            shutil.copy2(ROOT/'scripts/three_phase_observation_bridge.py', bridge)
            training.write_text(text.replace(OLD,NEW), encoding='utf-8')
            files += [str(training),str(bridge),str(backup)]
        report = {'installed': files, 'original_b_enabled': enable_original_b,
                  'baseline_preserved': True, 'overwrite_historical_checkpoints': False}
        (destination/'integration_report.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
        return report
    except BaseException:
        if backup.exists() and enable_original_b:
            shutil.copy2(backup,training)
            backup.unlink()
        if bridge.exists():
            bridge.unlink()
        launcher.unlink(missing_ok=True)
        shutil.rmtree(destination,ignore_errors=True)
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--enable-original-b',action='store_true')
    a=p.parse_args()
    print(json.dumps(install(a.repo,a.enable_original_b),indent=2))
