"""Command line for independent observation pretraining and source-only dynamics."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import platform
import sys
import numpy as np
import torch
from .config import load_config, from_dict
from .data import VisitStore, PairStore, array, convert_legacy
from .training import train, load_observation, load_world, evaluate_b
from .utils import read_json, write_json, file_identity, codec_file_id, load_checkpoint, autocast_context


def atomic_npz(path, **arrays):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return str(path)


def source_input(path, expected_codec_id, asserted_codec_id=None, valid_path=None):
    """A raw NPY requires an explicit codec assertion; an encoded NPZ is self-describing."""
    z = array(path, 'latent')
    embedded_id = None
    if str(path).lower().endswith('.npz'):
        with np.load(path, allow_pickle=False) as a:
            if 'codec_id' in a:
                embedded_id = str(a['codec_id'].item())
    if embedded_id and asserted_codec_id and embedded_id != asserted_codec_id:
        raise ValueError('Supplied codec ID contradicts the encoded source archive')
    actual = embedded_id or asserted_codec_id
    if actual != expected_codec_id:
        raise ValueError('Source codec provenance differs/missing; for raw NPY supply --codec-id matching the checkpoint')
    if z.ndim != 4 or z.shape[0] != 24:
        raise ValueError('Source must be raw continuous VQ coordinates [24,D,H,W], not standardized values')
    valid = array(valid_path, 'valid') if valid_path else torch.ones(1, *z.shape[1:])
    if valid.ndim == 3:
        valid = valid[None]
    if valid.ndim != 4 or valid.shape[0] not in (1, 3) or valid.shape[1:] != z.shape[1:] or ((valid < 0) | (valid > 1)).any():
        raise ValueError('Source validity mask must have shape [1|3,D,H,W] and values in [0,1]')
    return z, valid


@torch.no_grad()
def sample_file(checkpoint, source, conditions_path, output, *, device='cpu', codec_id=None,
                valid_path=None, samples=None, steps=None, method=None, seed=2026,
                direction='forward', codec_checkpoint=None, trusted_legacy=False):
    model, meta = load_world(checkpoint, device)
    cfg = model.cfg
    torch.set_num_threads(cfg.training.cpu_threads)
    count = cfg.sampling.samples if samples is None else samples
    if count < 1 or (steps is not None and steps < 1):
        raise ValueError('samples/steps must be positive')
    if direction not in ('forward', 'reverse'):
        raise ValueError('Unknown sampling direction')
    if meta['directions_trained'].get(direction, 0) < 1:
        raise ValueError(f'This checkpoint has not trained the {direction} conditional task')
    raw, valid = source_input(source, meta['codec_id'], codec_id, valid_path)
    conditions = read_json(conditions_path)
    from .conditioning import validate_condition
    conditions = validate_condition(conditions)
    raw, valid = raw[None].to(device), valid[None].to(device)
    normalized = model.observation_encoder.normalize(raw)
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions = []
    precision = cfg.training.precision if torch.device(device).type == 'cuda' else 'fp32'
    for _ in range(count):
        noise = torch.randn(normalized.shape, device=device, dtype=normalized.dtype, generator=generator)
        with autocast_context(device, precision):
            pred = model.sample(normalized, [conditions], noise, valid=valid, steps=steps, method=method,
                                direction=1 if direction == 'forward' else -1)
        predictions.append(pred.float())
    standardized = torch.cat(predictions, 0)
    result = model.observation_encoder.denormalize(standardized)
    arrays = {'latent': result.cpu().numpy(), 'standardized_latent': standardized.cpu().numpy(),
              'codec_id': np.asarray(meta['codec_id']), 'phase_order': np.asarray(['pre_aqc0','first_post_aqc1','metadata_late'])}
    if codec_checkpoint:
        expected = codec_file_id(codec_checkpoint)
        if expected != meta['codec_id']:
            raise ValueError('Image reconstruction requires the exact VQ checkpoint recorded in the data manifest')
        from .codec import load_codec
        codec = load_codec(codec_checkpoint, device, trusted_legacy=trusted_legacy)
        arrays['images'] = torch.cat([codec.decode(x[None]).float().cpu() for x in result], 0).numpy()
    if not np.isfinite(arrays['latent']).all():
        raise FloatingPointError('Generated latent contains nonfinite values')
    destination = atomic_npz(output, **arrays)
    provenance = {'checkpoint_identity': file_identity(checkpoint), 'source_identity': file_identity(source),
                  'codec_id': meta['codec_id'], 'seed': seed, 'direction': direction,
                  'samples': count, 'steps': steps or cfg.sampling.steps, 'method': method or cfg.sampling.method,
                  'conditions': conditions, 'uses_A_ssl_predictor': False, 'future_inputs': [],
                  'array_space': 'raw_continuous_VQ', 'support_assumed': valid_path is None,
                  'warning': 'No target geometry inferred; registration/crop validity and clinical utility require independent validation.'}
    write_json(str(output)+'.json', provenance)
    return {'prediction': destination, 'provenance': str(output)+'.json', 'shape': list(result.shape)}


@torch.no_grad()
def extract_file(checkpoint, source, output, *, device='cpu', codec_id=None, valid_path=None):
    value = load_checkpoint(checkpoint)
    if value.get('stage') == 'A':
        encoder, meta = load_observation(checkpoint, device)
    else:
        world, meta = load_world(checkpoint, device)
        encoder = world.observation_encoder
    torch.set_num_threads(meta['config']['training']['cpu_threads'])
    raw, valid = source_input(source, meta['codec_id'], codec_id, valid_path)
    features = encoder(raw[None].to(device), valid[None].to(device), teacher_average=True,
                       minimum_coverage=meta['config']['observation']['minimum_patch_coverage'])
    return {'features': atomic_npz(output, tokens=features.tokens.float().cpu().numpy(),
                                  padding=features.padding.cpu().numpy(), positions=features.positions.cpu().numpy(),
                                  indices=features.indices.cpu().numpy(), grid=np.asarray(features.grid),
                                  pooled=features.pooled().float().cpu().numpy()),
            'visit_only': True, 'uses_ssl_predictor': False}


def smoke(output, config_path):
    from .synthetic import make_synthetic
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Smoke output must be a new/empty directory')
    cfg = load_config(config_path)
    paths = make_synthetic(output/'data')
    a = train('A', cfg, paths['visits'], output/'run')
    b = train('B', cfg, paths['pairs'], output/'run', observation_checkpoint=output/'run/A/best.pt')
    pair = PairStore(paths['pairs']).records('test')[0]
    condition_path = output/'source_conditions.json'
    write_json(condition_path, pair['conditions'])
    base = Path(paths['pairs']).parent
    source = base/pair['source']['latent_path']
    # Physically remove the held-out future file before source-only inference.
    future = base/pair['target']['latent_path']
    future.unlink()
    pred = sample_file(output/'run/B/best.pt', source, condition_path, output/'prediction.npz',
                       codec_id=PairStore(paths['pairs']).codec_id,
                       valid_path=base/pair['source']['valid_path'], device=cfg.training.device,
                       direction='forward', samples=2)
    extraction = extract_file(output/'run/A/best.pt', source, output/'features.npz',
                              codec_id=PairStore(paths['pairs']).codec_id, device=cfg.training.device)
    report = {'passed': a['complete'] and b['complete'], 'A': a, 'B': b,
              'generation_after_target_deletion': pred, 'target_deleted': not future.exists(),
              'extraction': extraction, 'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_available': torch.cuda.is_available(), 'synthetic_only': True,
              'full_size_patient_gpu_training_tested': False}
    write_json(output/'report.json', report)
    return report


def build_parser():
    p = argparse.ArgumentParser(description='Observation-only Stage A + longitudinal SymmFlow Stage B (v3)')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('smoke', help='Run real reduced A/B networks and source-only inference on synthetic data')
    s.add_argument('--output', required=True)
    s.add_argument('--config', default=str(Path(__file__).resolve().parents[2]/'configs/smoke.yaml'))
    c = sub.add_parser('convert-legacy', help='Convert original admitted_inventory and latent caches into separate A/B manifests')
    c.add_argument('--root', required=True); c.add_argument('--output', required=True)
    group = c.add_mutually_exclusive_group(required=True)
    group.add_argument('--codec-id'); group.add_argument('--codec-checkpoint')
    c.add_argument('--qc', help='Optional audited visit sidecar metadata JSON')
    a = sub.add_parser('audit'); a.add_argument('--visits', required=True)
    a.add_argument('--pairs'); a.add_argument('--output')
    d = sub.add_parser('prepare-sidecars', help='Encode image-space nuisance augmentations with the matching frozen VQ')
    for key in ('visits','codec-checkpoint','output'):
        d.add_argument('--'+key, required=True)
    d.add_argument('--samples', type=int, default=2); d.add_argument('--seed', type=int, default=2026)
    d.add_argument('--device', default='cpu'); d.add_argument('--cache-tolerance', type=float, default=.03)
    d.add_argument('--trusted-legacy', action='store_true', help='Explicitly allow unsafe loading of a trusted legacy VQ file')
    t = sub.add_parser('train')
    t.add_argument('--stage', choices=['A','B','all'], required=True)
    t.add_argument('--config', required=True); t.add_argument('--visits'); t.add_argument('--pairs')
    t.add_argument('--output', required=True); t.add_argument('--a-checkpoint')
    t.add_argument('--resume', action='store_true'); t.add_argument('--stop-after', type=int)
    for name in ('sample', 'extract'):
        x = sub.add_parser(name)
        for key in ('checkpoint','source','output'):
            x.add_argument('--'+key, required=True)
        x.add_argument('--device', default='cpu'); x.add_argument('--codec-id'); x.add_argument('--valid')
        if name == 'sample':
            x.add_argument('--conditions', required=True); x.add_argument('--samples', type=int)
            x.add_argument('--steps', type=int); x.add_argument('--method', choices=['euler','heun'])
            x.add_argument('--seed', type=int, default=2026); x.add_argument('--direction', choices=['forward','reverse'], default='forward')
            x.add_argument('--codec-checkpoint'); x.add_argument('--trusted-legacy', action='store_true')
    e = sub.add_parser('evaluate')
    for key in ('checkpoint','pairs','output'):
        e.add_argument('--'+key, required=True)
    e.add_argument('--split', choices=['val','test'], default='test'); e.add_argument('--device', default='cpu')
    e.add_argument('--limit', type=int)
    z = sub.add_parser('encode-source', help='Encode already preprocessed, normalized same-visit MRI [3,D,H,W]')
    for key in ('images','codec-checkpoint','output'):
        z.add_argument('--'+key, required=True)
    z.add_argument('--device', default='cpu'); z.add_argument('--trusted-legacy', action='store_true')
    return p


def main(argv=None):
    p = build_parser(); args = p.parse_args(argv)
    if args.command == 'smoke':
        result = smoke(args.output, args.config)
    elif args.command == 'convert-legacy':
        codecid = args.codec_id or codec_file_id(args.codec_checkpoint)
        result = convert_legacy(args.root, args.output, codec_id=codecid, qc_path=args.qc)
    elif args.command == 'audit':
        store = VisitStore(args.visits); result = store.audit()
        if args.pairs:
            pairs = PairStore(args.pairs)
            result['pair_count'] = len(pairs.pairs)
            if store.codec_id != pairs.codec_id:
                raise ValueError('A/B codec IDs differ')
            for pid, split in pairs.patient_splits.items():
                if pid in store.patient_splits and store.patient_splits[pid] != split:
                    raise ValueError('A/B patient splits differ')
        if args.output:
            write_json(args.output, result)
    elif args.command == 'prepare-sidecars':
        from .preparation import prepare_sidecars
        result = prepare_sidecars(args.visits, args.codec_checkpoint, args.output, samples=args.samples,
                                   device=args.device, seed=args.seed, cache_tolerance=args.cache_tolerance,
                                   trusted_legacy=args.trusted_legacy)
    elif args.command == 'train':
        cfg = load_config(args.config); result = {}
        if args.stage in ('A', 'all'):
            if not args.visits:
                p.error('Stage A requires --visits; it does not accept pair data')
            result['A'] = train('A', cfg, args.visits, args.output, resume=args.resume, stop_after=args.stop_after)
        if args.stage == 'B' or (args.stage == 'all' and result['A']['complete']):
            if not args.pairs:
                p.error('Stage B requires --pairs')
            checkpoint = args.a_checkpoint or str(Path(args.output)/'A/best.pt')
            resume_b = args.resume and (Path(args.output)/'B/last.pt').exists() if args.stage == 'all' else args.resume
            result['B'] = train('B', cfg, args.pairs, args.output, observation_checkpoint=checkpoint,
                                 resume=resume_b, stop_after=args.stop_after)
    elif args.command == 'sample':
        result = sample_file(args.checkpoint, args.source, args.conditions, args.output, device=args.device,
                             codec_id=args.codec_id, valid_path=args.valid, samples=args.samples,
                             steps=args.steps, method=args.method, seed=args.seed, direction=args.direction,
                             codec_checkpoint=args.codec_checkpoint, trusted_legacy=args.trusted_legacy)
    elif args.command == 'extract':
        result = extract_file(args.checkpoint, args.source, args.output, device=args.device,
                              codec_id=args.codec_id, valid_path=args.valid)
    elif args.command == 'evaluate':
        model, meta = load_world(args.checkpoint, args.device)
        torch.set_num_threads(model.cfg.training.cpu_threads)
        store = PairStore(args.pairs)
        if store.codec_id != meta['codec_id']:
            raise ValueError('Evaluation VQ codec differs from the training codec')
        for row in store.records(args.split):
            for exposure in (meta['patient_splits'], meta['A_patient_splits']):
                previous = exposure.get(row['patient_id'])
                if previous == 'train' or (args.split == 'test' and previous == 'val'):
                    raise ValueError('Evaluation patient participated in training or checkpoint selection')
        result = evaluate_b(model, store, model.cfg, args.split, args.limit)
        write_json(args.output, result)
    else:
        from .codec import load_codec
        codec = load_codec(args.codec_checkpoint, args.device, trusted_legacy=args.trusted_legacy)
        images = array(args.images, 'images')
        if images.ndim != 4 or images.shape[0] != 3:
            raise ValueError('Input images must be [3,D,H,W] in the exact preprocessing/normalization of the VQ checkpoint')
        with torch.no_grad():
            latent = codec.encode(images[None].to(args.device)).float()[0].cpu().numpy()
        codecid = codec_file_id(args.codec_checkpoint)
        result = {'source': atomic_npz(args.output, latent=latent, codec_id=np.asarray(codecid)), 'codec_id': codecid}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
