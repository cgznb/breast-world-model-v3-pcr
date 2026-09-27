from pathlib import Path
import importlib.util
import json
import subprocess
import zipfile
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def mock_repo(tmp_path):
    repo=tmp_path/'repo';core=repo/'workflows/first_post_three_phase/mewm_ispy2';core.mkdir(parents=True)
    (core/'three_phase_symmflow.py').write_text('# original baseline\n')
    installer=load_script('install_into_repo')
    (core/'three_phase_all_pairs_training.py').write_text('def original():\n'+installer.OLD+'\n')
    (repo/'README.md').write_text('original repo')
    return repo,core


def test_install_preserves_baseline_and_exact_opt_in_patch(tmp_path):
    repo,core=mock_repo(tmp_path);installer=load_script('install_into_repo')
    text=(core/'three_phase_all_pairs_training.py').read_text()
    result=installer.install(repo,True)
    assert result['baseline_preserved'] and result['original_b_enabled']
    assert (core/'three_phase_symmflow.py').read_text()=='# original baseline\n'
    assert (core/'three_phase_all_pairs_training.py.observation_v3_backup').read_text()==text
    assert 'ObservationConditionedSymmFlow' in (core/'three_phase_all_pairs_training.py').read_text()
    assert (repo/'run_observation_v3.py').is_file()
    assert (repo/'workflows/observation_v3/src/symm_observation/observation_ssl.py').is_file()
    with pytest.raises(FileExistsError):installer.install(repo,True)
    subprocess.run(['python',str(repo/'run_observation_v3.py'),'--help'],check=True,capture_output=True)


def test_changed_upstream_fails_before_writing(tmp_path):
    repo,core=mock_repo(tmp_path)
    (core/'three_phase_all_pairs_training.py').write_text('# changed upstream\n')
    with pytest.raises(ValueError):load_script('install_into_repo').install(repo,True)
    assert not (repo/'workflows/observation_v3').exists()


def test_local_full_repository_assembler_excludes_private_assets(tmp_path):
    repo,core=mock_repo(tmp_path)
    (repo/'weights.pt').write_bytes(b'private')
    (repo/'paths.local.yaml').write_text('private: true')
    (repo/'.env').write_text('secret=not_a_real_secret')
    for args in (['init'],['config','user.email','test@example.invalid'],['config','user.name','Test'],
                 ['add','.'],['commit','-m','synthetic fixture']):
        subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    output=tmp_path/'full.zip'
    result=load_script('assemble_full_repository').assemble(repo,output)
    assert result['tracked_files_copied']>=3
    with zipfile.ZipFile(output) as z:
        names=z.namelist()
        assert not any(n.endswith('weights.pt') or n.endswith('paths.local.yaml') or n.endswith('/.env') for n in names)
        assert any(n.endswith('/src/symm_observation/observation_ssl.py') for n in names)


def test_bridge_file_compiles():
    compile((ROOT/'scripts/three_phase_observation_bridge.py').read_text(),'<bridge>','exec')
