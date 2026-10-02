import copy
from pathlib import Path
import numpy as np
import pytest
import torch

from responsewm.synthetic import make_synthetic
from responsewm.data import ManifestStore,PatientSampler
from responsewm.io import read_json,write_json,load_checkpoint
from responsewm.training import train_stage,STAGES,initialize_run
from responsewm.inference import read_request,predict
from responsewm.checkpoints import migrate_v2


def cohort(tmp_path):
    path=make_synthetic(tmp_path/'data')
    store=ManifestStore(path,allow_synthetic=True)
    store.fit_statistics()
    return path,store


def mutate_manifest(path,mutate):
    value=read_json(path); mutate(value); write_json(path,value)


def test_patient_split_leak_is_rejected(tmp_path):
    path,store=cohort(tmp_path)
    mutate_manifest(path,lambda x:x['cases'][1].update(split='test'))
    with pytest.raises(ValueError,match='overlaps'):
        ManifestStore(path,True)


def test_future_clinical_availability_rejected(tmp_path):
    path,store=cohort(tmp_path)
    mutate_manifest(path,lambda x:x['cases'][0]['input']['clinical_known_at'].__setitem__(0,99))
    with pytest.raises(ValueError,match='unavailable'):
        ManifestStore(path,True)


def test_target_mismatch_and_unknown_fields_rejected(tmp_path):
    path,store=cohort(tmp_path)
    original=read_json(path)
    value=copy.deepcopy(original); value['cases'][0]['input']['pcr']=1; write_json(path,value)
    with pytest.raises(ValueError,match='Unexpected'):
        ManifestStore(path,True)
    value=copy.deepcopy(original); value['cases'][0]['target']['future'][-1]['day']=91; write_json(path,value)
    with pytest.raises(ValueError,match='paired target'):
        ManifestStore(path,True)


def test_no_synthetic_silent_fallback(tmp_path):
    path,store=cohort(tmp_path)
    with pytest.raises(ValueError,match='Synthetic'):
        ManifestStore(path)


def test_statistics_do_not_use_validation_test(tmp_path):
    path,store=cohort(tmp_path)
    before=store.statistics
    for p in store.all_assets():
        if any(f'synthetic_{i:03d}' in p for i in (4,5,6,7)):
            x=np.load(p); np.save(p,x+12345)
    fresh=ManifestStore(path,True); after=fresh.fit_statistics()
    assert before==after
    with pytest.raises(ValueError,match='training split'):
        fresh.set_statistics({**before,'fit_split':'test'})


def test_source_only_reader_never_opens_future(tmp_path):
    path,store=cohort(tmp_path)
    i=store.by_split['test'][0]
    future=store.cases[i]['target']['future']
    for v in future:
        if v is not None:
            Path(v['latent']).unlink()
    store.cache.clear()
    inp=store.batch([i],supervised=False)
    assert inp.observed.shape[:2]==(1,1)
    with pytest.raises(FileNotFoundError):
        store.batch([i],supervised=True)


def test_patient_sampler_rng_resume(tmp_path):
    _,store=cohort(tmp_path)
    sampler=PatientSampler(store,1); sampler.sample(3)
    state=sampler.generator.get_state()
    expected=sampler.sample(8)
    other=PatientSampler(store,123); other.generator.set_state(state)
    assert other.sample(8)==expected


def test_missing_targets_are_masked_not_negative_labels(tmp_path):
    _,store=cohort(tmp_path)
    i=store.by_split['train'][0]; store.cases[i]['target']['pcr']=None
    inp,sup=store.batch([i])
    assert not sup.label_mask[0]
    assert not sup.future_mask[0,0]
    assert inp.future_mask[0,0]  # Forecast is requested despite absent supervision.


def test_checkpoint_resume_exact_and_mutation_guard(tmp_path,cfg):
    path,store=cohort(tmp_path)
    cfg.training.strict_determinism=True
    full=tmp_path/'full'; interrupted=tmp_path/'resumed'
    train_stage(store,cfg,full,'representation')
    train_stage(ManifestStore(path,True),cfg,interrupted,'representation',stop_after=1)
    train_stage(ManifestStore(path,True),cfg,interrupted,'representation',resume=True)
    a=load_checkpoint(full/'representation'/'last.pt')
    b=load_checkpoint(interrupted/'representation'/'last.pt')
    assert a['completed'] and b['completed']
    for k in a['model']:
        assert torch.equal(a['model'][k],b['model'][k]),k
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    with pytest.raises(FileExistsError):
        train_stage(store,cfg,full,'representation')
    p=Path(store.all_assets()[0]); np.save(p,np.load(p)+1.)
    with pytest.raises(ValueError,match='changed'):
        initialize_run(ManifestStore(path,True),cfg,full)


def test_four_stage_training_and_input_only_deployment(tmp_path,cfg):
    path,store=cohort(tmp_path); root=tmp_path/'run'
    for stage in STAGES:
        result=train_stage(store,cfg,root,stage)
        assert result['completed']
    cp=root/'joint'/'best.pt'; output=tmp_path/'prediction.npz'
    info=predict(cp,path.parent/'request.json',output,samples=2,steps=2)
    assert info['clinical_validation'] is False
    with np.load(output) as x:
        assert x['latent'].shape==(1,2,3,24,4,8,8)
        assert np.allclose(x['pcr_probability'],x['trajectory_probabilities'].mean(1))
    payload=load_checkpoint(cp)
    request=read_json(path.parent/'request.json'); request['target']={'pcr':1}
    bad=tmp_path/'bad.json'; write_json(bad,request)
    with pytest.raises(ValueError,match='Unexpected'):
        read_request(bad,payload)
    with pytest.raises(ValueError,match='no trained'):
        predict(root/'flow'/'best.pt',path.parent/'request.json',tmp_path/'bad.npz')


def test_strict_v2_migration_preserves_current_statistics(tmp_path,model):
    cp=tmp_path/'legacy.pt'; state={}
    for prefix,module in [('encoder.',model.encoder),('heads.',model.state_heads),('velocity.',model.velocity.image)]:
        state.update({prefix+k:v.clone() for k,v in module.state_dict().items()})
    state['encoder.latent_mean'].fill_(987)
    torch.save({'model':state},cp)
    mean=model.encoder.latent_mean.clone()
    report=migrate_v2(model,cp)
    assert torch.equal(model.encoder.latent_mean,mean)
    assert report['image_backbone']['strict']
    del state[next(k for k in state if k.startswith('velocity.'))]
    torch.save({'model':state},cp)
    with pytest.raises(ValueError,match='parameter keys differ'):
        migrate_v2(model,cp)


def test_stage_index_is_explicit_and_integral(tmp_path,cfg):
    path,store=cohort(tmp_path)
    value=read_json(path); value['time_basis']='stage_index'
    for case in value['cases']:
        inp=case['input']; inp['landmark_day']/=30
        for v in inp['observed']:
            v['day']/=30; v['available_at']/=30
        for q in inp['queries']: q['day']/=30
        for v in case['target']['future']:
            if v is not None: v['day']/=30
    write_json(path,value)
    store=ManifestStore(path,True)
    with pytest.raises(ValueError,match='time basis'):
        initialize_run(store,cfg,tmp_path/'run')
    cfg.network.time_basis='stage_index'
    initialize_run(store,cfg,tmp_path/'run2')
    value['cases'][0]['input']['queries'][0]['day']=.5
    write_json(path,value)
    with pytest.raises(ValueError,match='integers'):
        ManifestStore(path,True)


def test_flow_refuses_cohort_without_actual_adjacent_pairs(tmp_path,cfg):
    path,store=cohort(tmp_path)
    value=read_json(path)
    for case in value['cases']:
        if case['split']=='train':
            future=case['target']['future']
            for j in range(len(future)-1): future[j]=None
    write_json(path,value)
    with pytest.raises(ValueError,match='No eligible adjacent'):
        train_stage(ManifestStore(path,True),cfg,tmp_path/'empty_flow','flow')


def test_all_shipped_model_configs_validate():
    from responsewm.config import load_config
    root=Path(__file__).resolve().parents[1]
    for path in (root/'configs').rglob('*.yaml'):
        load_config(path).validate()
