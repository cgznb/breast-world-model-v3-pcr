from pathlib import Path
import hashlib
import re
import pytest
from symm_observation.config import load_config
from symm_observation.cli import source_input, build_parser
from symm_observation.utils import read_json

ROOT=Path(__file__).resolve().parents[1]


def test_upstream_reference_matches_fetched_git_blob():
    b=(ROOT/'upstream_reference/three_phase_symmflow.py').read_bytes()
    observed=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
    assert observed=='6ca00eb2a2bdfd7039dbda8688a88674fd970179'


def test_all_configs_validate():
    for file in (ROOT/'configs').rglob('*.yaml'):
        cfg=load_config(file)
        assert cfg.schema=='three_phase_observation_v3'
        assert 'future' not in cfg.observation.__dict__


def test_A_source_requires_explicit_codec_identity(visits):
    row=visits.records('train')[0];path=visits.base/row['latent_path']
    with pytest.raises(ValueError):source_input(path,visits.codec_id)
    with pytest.raises(ValueError):source_input(path,visits.codec_id,'wrong')
    z,v=source_input(path,visits.codec_id,visits.codec_id)
    assert z.shape==(24,8,8,8)


def test_C_not_an_available_stage():
    parser=build_parser()
    with pytest.raises(SystemExit):parser.parse_args(['train','--stage','C','--config','x','--output','x'])


def test_bridge_identity_survives_upstream_public_filter():
    # Exact upstream filtering semantics, not an assertion that its hash-stripping
    # is desirable. The new bridge deliberately records byte integers instead.
    def public(v):
        if isinstance(v,dict):
            return {str(k):public(x) for k,x in v.items() if not any(s in str(k).lower() for s in ('sha256','checksum','fingerprint','hash','revision'))}
        if isinstance(v,(list,tuple)):return [public(x) for x in v]
        if isinstance(v,str):return re.sub(r'(?<![a-zA-Z0-9])[a-fA-F0-9]{40,64}(?![a-zA-Z0-9])','<legacy-digest>',v)
        return v
    a={'observation_asset':{'content_identity_bytes':list(bytes.fromhex('00'*32))}}
    b={'observation_asset':{'content_identity_bytes':list(bytes.fromhex('01'*32))}}
    assert public(a)!=public(b)
    bridge=(ROOT/'scripts/three_phase_observation_bridge.py').read_text()
    assert 'content_identity_bytes' in bridge
