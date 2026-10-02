import copy
from types import SimpleNamespace

import pytest
import torch

from responsewm.backbones import CoupledVelocityV2, NativeImageBackbone
from responsewm.legacy.encoder import PhaseStateEncoder
from responsewm.representation import DeepJEPAPredictor, deep_jepa_loss


def velocity_fixture(cfg, backend="native"):
    cfg = copy.deepcopy(cfg)
    cfg.network.semantic_depth = 6
    cfg.network.backend = backend
    if backend == "monai":
        cfg.network.channels = (16, 32, 32)
    model = CoupledVelocityV2(cfg)
    z = torch.randn(2, 48, 4, 8, 8)
    state = torch.randn(2, 2 * cfg.encoder.disease_tokens, cfg.encoder.dim)
    context = SimpleNamespace(cond_tokens=torch.randn(2, 4, cfg.encoder.dim),
                              cond_global=None, semantic_global=torch.randn(2, cfg.encoder.dim),
                              state_tokens=torch.randn(2, 16, cfg.encoder.dim),
                              clinical_tokens=torch.randn(2, 3, cfg.encoder.dim),
                              interval_action_tokens=torch.randn(2, 1, cfg.encoder.dim))
    return model, (z, state, torch.rand(2), context)


def warm_output(model):
    with torch.no_grad():
        if model.cfg.network.backend == "native":
            model.image.out.weight.normal_(0, .01)
        else:
            for parameter in model.image.network.out[-1].parameters():
                parameter.normal_(0, .01)
        model.semantic_out[-1].weight.normal_(0, .02)


def test_zero_bridge_preserves_native_image_and_parameter_layout(cfg):
    model, (z, state, tau, context) = velocity_fixture(cfg)
    warm_output(model)
    original = NativeImageBackbone(model.cfg.network, model.cfg.encoder.dim)
    original.load_state_dict(model.image.state_dict(), strict=True)
    expected = original(z, tau, context.cond_tokens)
    actual = model(z, state, tau, context)[0]
    assert torch.equal(expected, actual)
    assert len(model.blocks) == 6
    assert all(block.modulation[-1].out_features == 12 * cfg.encoder.dim for block in model.blocks)


def test_early_bridge_changes_deeper_grid_and_decoder(cfg):
    model, inputs = velocity_fixture(cfg)
    warm_output(model)
    baseline = model(*inputs)
    with torch.no_grad():
        model.bridges[0].image_gate.fill_(.5)
    changed = model(*inputs)
    assert not torch.allclose(baseline[2], changed[2])
    assert not torch.allclose(baseline[0], changed[0])
    loss = changed[0].square().mean() + changed[2].square().mean()
    loss.backward()
    assert model.bridges[0].image_gate.grad.abs() > 0
    assert model.bridges[0].image_out.weight.grad.abs().sum() > 0


def test_checkpoint_forward_backward_and_training_determinism(cfg):
    plain, inputs = velocity_fixture(cfg)
    warm_output(plain)
    with torch.no_grad():
        for bridge in plain.bridges:
            bridge.image_gate.fill_(.1)
            bridge.state_gate.fill_(.1)
        for block in plain.blocks:
            block.modulation[-1].bias.fill_(.03)
    recompute = copy.deepcopy(plain)
    recompute.cfg.network.checkpoint_blocks = True
    a, b = plain(*inputs), recompute(*inputs)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    torch.manual_seed(999)
    assert all(torch.equal(x, y) for x, y in zip(a, plain(*inputs)))
    sum(x.square().mean() for x in a).backward()
    sum(x.square().mean() for x in b).backward()
    for (name, p), (_, q) in zip(plain.named_parameters(), recompute.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-6, rtol=1e-4), name


def test_zero_initialized_semantic_path_opens_after_optimizer_steps(cfg):
    model, inputs = velocity_fixture(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = torch.randn_like(inputs[1])
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model(*inputs)[1] - target).square().mean()
        loss.backward()
        optimizer.step()
    assert model.semantic_in.weight.grad.abs().sum() > 0
    assert model.blocks[0].action_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.bridges[0].to_state.in_proj_weight.grad.abs().sum() > 0


def test_monai_v2_zero_bridge_split_matches_actual_backend(cfg):
    monai = pytest.importorskip("monai")
    if monai.__version__ != "1.5.1":
        pytest.skip("Split traversal is pinned to MONAI 1.5.1")
    model, (z, state, tau, context) = velocity_fixture(cfg, "monai")
    warm_output(model)
    model.eval()
    with torch.no_grad():
        expected = model.image.network(z, tau, context=context.cond_tokens)
        actual = model(z, state, tau, context)[0]
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_encoder_pyramid_preserves_default_graph_and_keys(cfg):
    encoder = PhaseStateEncoder(cfg.encoder).eval()
    keys = set(encoder.state_dict())
    z = torch.randn(2, 24, 4, 8, 8, requires_grad=True)
    ordinary = encoder(z)
    state, pyramid = encoder(z, return_pyramid=True)
    assert torch.equal(ordinary.dense, state.dense)
    assert set(encoder.state_dict()) == keys
    assert len(pyramid.tokens) == len(cfg.encoder.widths)
    sum(x.square().mean() for x in pyramid.tokens).backward()
    assert z.grad.abs().sum() > 0


def test_deep_jepa_stopgrad_targets_and_fixed_teacher_input_gradients(cfg):
    encoder = PhaseStateEncoder(cfg.encoder)
    teacher = copy.deepcopy(encoder).eval().requires_grad_(False)
    predictor = DeepJEPAPredictor(cfg.encoder)
    z = torch.randn(2, 24, 4, 8, 8)
    loss, details = deep_jepa_loss(encoder, teacher, predictor, z,
                                   level_weights=(.2, 1.0), mix=.5)
    assert set(details) == {"jepa_fused", "jepa_deep", "jepa_level0", "jepa_level1"}
    loss.backward()
    assert encoder.stem[0].weight.grad.abs().sum() > 0
    assert all(head[-1].weight.grad.abs().sum() > 0 for head in predictor.level_outputs)
    assert all(p.grad is None for p in teacher.parameters())
    generated = z.clone().requires_grad_()
    teacher(generated).disease.square().mean().backward()
    assert generated.grad.abs().sum() > 0
