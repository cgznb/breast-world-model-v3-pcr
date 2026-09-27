import numpy as np
import pytest
import torch

from release_inference import export_bundle, predict
from scripts.run_full978_anti_overfit import _prior_from_state
from src.data import TABULAR_FEATURE_NAMES, _vec_from_tensor
from src.tdn import TDN
from src.temporal import canonicalize_temporal_prefix


def bundle(tmp_path, fold=0, seed=42):
    torch.manual_seed(10 + fold)
    config = dict(input_dim=1152, proj_dim=8, sig_dim=8, clinical_dim=17,
                  n_layers=1, n_heads=2, dropout=0.0, use_prior=True)
    model = TDN({"downstream": config}).eval()
    prior = dict(coef=[[0.1] * 17], intercept=[-0.3], feature_names=list(TABULAR_FEATURE_NAMES))
    source = tmp_path / f"source-{fold}.pt"
    torch.save(dict(schema="registered_three_phase_roi32_pcr_300_50", model_state=model.state_dict(),
                    task=dict(effective=config, candidate="v1", depth=3, seed=seed, outer=fold,
                              train_ids=["synthetic-only"]), clinical_prior=prior), source)
    output = tmp_path / f"bundle-{fold}.pt"
    exported = export_bundle(source, output)
    assert "task" not in exported
    assert "train_ids" not in str(exported["classifier"])
    return output, model, prior


def test_export_and_probability_averaging_replay_original_tdn(tmp_path):
    torch.set_num_threads(1)
    paths, models, priors = zip(*(bundle(tmp_path, fold) for fold in range(2)))
    rng = np.random.default_rng(17)
    images = rng.normal(size=(4, 3, 4, 1152)).astype("float32")
    masks = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 1]], dtype="float32")
    clinical = rng.normal(size=(3, 17)).astype("float32")
    days = np.tile(np.array([0, 30, 90, 180], dtype="float32"), (3, 1))
    result = predict(paths, images, masks, clinical, days)
    expected = []
    with torch.inference_mode():
        for model, prior in zip(models, priors):
            per_draw = []
            for draw in images:
                z = np.stack([_vec_from_tensor(torch.from_numpy(x)) for x in draw.reshape(-1, 1152)])
                z, mask, elapsed = canonicalize_temporal_prefix(z.reshape(3, 4, 1152), masks, days, 3)
                logits = model(torch.from_numpy(z), torch.from_numpy(mask), torch.from_numpy(clinical),
                               days=torch.from_numpy(elapsed),
                               prior_logit=torch.from_numpy(_prior_from_state(clinical, prior)))
                per_draw.append(logits.sigmoid().numpy())
            expected.append(np.mean(per_draw, axis=0))
    np.testing.assert_allclose(result["fold_probabilities"], expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result["probability"], np.mean(expected, axis=0), rtol=0, atol=1e-7)
    changed = images.copy()
    changed[:, :, 3] += 100
    changed[:, 2, 2] -= 100
    np.testing.assert_array_equal(predict(paths, changed, masks, clinical, days)["probability"],
                                  result["probability"])


def test_reject_mixed_seeds_and_invalid_inputs(tmp_path):
    first, _, _ = bundle(tmp_path, 0, 42)
    second, _, _ = bundle(tmp_path, 1, 43)
    x = np.ones((1, 4, 1152), dtype="float32")
    mask = np.ones((1, 4), dtype="float32")
    clinical = np.zeros((1, 17), dtype="float32")
    days = np.zeros((1, 4), dtype="float32")
    with pytest.raises(ValueError, match="one classifier"):
        predict([first, second], x, mask, clinical, days)
    with pytest.raises(ValueError, match="distinct folds"):
        predict([first, first], x, mask, clinical, days)
    mask[0, 0] = 0
    with pytest.raises(ValueError, match="Invalid"):
        predict([first], x, mask, clinical, days)
