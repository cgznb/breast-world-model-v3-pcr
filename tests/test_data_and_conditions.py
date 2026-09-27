from pathlib import Path
import copy
import json
import numpy as np
import pytest
import torch
from symm_observation.data import VisitStore, PairStore, PatientSampler, convert_legacy, PHASES
from symm_observation.conditioning import validate_condition, ConditionSchema, ClinicalEncoder
from symm_observation.utils import read_json, write_json
from symm_observation.observation_ssl import ObservationPretrainer
from symm_observation.observation_views import observation_batch


def rewrite(store, obj):
    path=store.path.parent/'altered.json';write_json(path,obj);return path

@pytest.mark.parametrize('key',['pcr','future_treatment','next_visit','target_latent','recurrence'])
def test_A_manifest_refuses_future_fields(visits,key):
    m=copy.deepcopy(visits.manifest);m['visits'][0][key]='invalid'
    with pytest.raises(ValueError): VisitStore(rewrite(visits,m))


def test_A_rejects_pair_manifest(dataset):
    with pytest.raises(ValueError): VisitStore(dataset['pairs'])


def test_A_patient_visit_deduplication(visits):
    m=copy.deepcopy(visits.manifest);row=copy.deepcopy(m['visits'][0]);row['id']='other_crop'
    m['visits'].append(row)
    with pytest.raises(ValueError): VisitStore(rewrite(visits,m))


def test_patient_split_leak_rejected(visits):
    m=copy.deepcopy(visits.manifest);m['visits'][1]['split']='test'
    with pytest.raises(ValueError): VisitStore(rewrite(visits,m))


def test_fit_statistics_train_only(visits):
    before=visits.fit_statistics()
    for r in visits.records('val')+visits.records('test'):
        path=visits.base/r['latent_path'];a=np.load(path);np.save(path,a+1000)
    after=visits.fit_statistics()
    assert before==after and before['fit_split']=='train'


def test_sampler_counter_replay_and_patient_balance(visits):
    sampler=PatientSampler(visits.records('train'),100)
    assert sampler.batch(19,4)==sampler.batch(19,4)
    counts={r['patient_id']:0 for r in visits.records('train')}
    for i in range(400):counts[sampler.batch(i,1)[0]['patient_id']]+=1
    assert min(counts.values())>60


def test_missing_aux_not_automatically_negative(visits,cfg):
    m=copy.deepcopy(visits.manifest)
    for row in m['visits']:
        row.pop('kinetic_path',None);row.pop('kinetic_provenance',None);row.pop('segmentation_path',None)
        row['phenotype_label']=-1;row['domain_label']=-1
    store=VisitStore(rewrite(visits,m))
    model=ObservationPretrainer(cfg,store.fit_statistics())
    batch=observation_batch(store,store.records('train')[:1],cfg,0,'cpu')
    _,p=model(batch,0)
    assert p['kinetics_labels']==0 and p['segmentation_labels']==0
    assert p['kinetics']==0 and p['segmentation']==0 and p['arcface']==0 and p['dann']==0


def test_all_zero_segmentation_is_valid_label(visits,cfg):
    row=visits.records('train')[0]
    file=visits.base/row['segmentation_path'];np.save(file,np.zeros_like(np.load(file)))
    m=ObservationPretrainer(cfg,visits.fit_statistics())
    batch=observation_batch(visits,[row],cfg,0,'cpu')
    loss,p=m(batch,0)
    assert p['segmentation_labels']==1 and torch.isfinite(loss) and p['segmentation']>0

@pytest.mark.parametrize('field',['pcr','pathology','rcb','recurrence'])
def test_B_condition_outcomes_rejected(field):
    with pytest.raises(ValueError): validate_condition({field:1})


def test_unverified_dates_removed_not_zero_fabricated():
    assert validate_condition({'delta_days':91})['delta_days'] is None
    assert validate_condition({'delta_days':91,'interval_verified':True})['delta_days']==91


def test_unknown_future_actual_treatment_rejected():
    with pytest.raises(ValueError): validate_condition({'action_segments':[{'drug':'A','start':0,'end':2,'dose':1,'known_at_source':False}]})


def test_condition_validation_no_inplace_change():
    c={'action_segments':[{'drug':'A','start':'0','end':'2','dose':'1','known_at_source':True}]}
    original=copy.deepcopy(c);result=validate_condition(c)
    assert c==original and result['action_segments'][0]['start']==0.


def test_clinical_padding_not_extra_drug(cfg,pairs):
    schema=ConditionSchema.fit(pairs.records('train'))
    c=ClinicalEncoder(schema,48,4).eval()
    one=copy.deepcopy(pairs.records('train')[0]['conditions'])
    many=copy.deepcopy(one)
    many['action_segments']=[{'drug':'A','start':i,'end':i+1,'dose':1.,'known_at_source':True} for i in range(3)]
    alone=c([one]);together=c([one,many])
    assert alone.shape[1]==together.shape[1]
    torch.testing.assert_close(alone[0],together[0],atol=1e-5,rtol=1e-5)


def test_original_inventory_conversion_and_dedup(tmp_path):
    root=tmp_path/'legacy';root.mkdir();(root/'latents').mkdir()
    for name in ['a','b','b_alt']:
        np.save(root/'latents'/f'{name}.npy',np.random.default_rng(3).normal(size=(24,8,8,8)).astype('float16'))
    inv={'phase_order':PHASES,'visits':[{'visit_id':'v0','visit':'T0'},{'visit_id':'v1','visit':'T1'}],
         'views':[{'view_id':'a','visit_id':'v0','source_visit_id':'v0','patient_id':'p','split':'train','latent_file':'latents/a.npy'},
                  {'view_id':'b','visit_id':'v1','source_visit_id':'v0','patient_id':'p','split':'train','latent_file':'latents/b.npy'},
                  {'view_id':'b_alt','visit_id':'v1','source_visit_id':'v1','patient_id':'p','split':'train','latent_file':'latents/b_alt.npy'}],
         'pairs':[{'pair_id':'01','patient_id':'p','split':'train','source_view':'a','target_view':'b','earlier_stage':'T0','later_stage':'T1'}]}
    write_json(root/'admitted_inventory.json',inv)
    result=convert_legacy(root,tmp_path/'new','declared-codec')
    a=VisitStore(result['visits_manifest']);b=PairStore(result['pairs_manifest'])
    assert len(a.visits)==2 and len(b.pairs)==1
    assert a.visits[1]['id']=='b_alt' and b.pairs[0]['target']['id']=='b'
    assert not a.visits[0]['phase_alignment_verified']
    with pytest.raises(FileExistsError):convert_legacy(root,tmp_path/'new','declared-codec')
