# ROI32 pCR Prediction

This directory contains the complete historical longitudinal pCR source,
training scripts, configurations and tests, plus an input-only prediction CLI.
The primary workflow is `registered_three_phase_budget.py`: V1 classifiers for
T0, T0-T1, T0-T2 and T0-T3; V4 for T0-T3 with residual-logit L2 weight 0.1.
The generator version and classifier version are separate names.

Frozen Pillar0-BreastMRI produces 1152-dimensional features from three MRI
phases. A projection and time-aware Transformer TDN read the visit sequence and
17 clinical/treatment features. The final logit adds a learned imaging residual
to a training-fold logistic clinical prior. Full source is in `src/tdn.py` and
`src/first_post_optimization.py`. Learned weights are external inputs.

## Training

From this directory, install the root project's `pcr` and `test` extras, then:

```bash
python scripts/run_registered_three_phase_pcr.py --help
python scripts/run_registered_three_phase_budget.py --help
python -m pytest -q tests/test_registered_three_phase_budget.py tests/test_release_prediction.py
```

The first workflow constructs the real-image feature baseline. The second
retains the original 10 classifier seeds x 5 folds, 300 epochs, patience 50 and
validation-AUC selection. `configs/registered_three_phase_pcr_300_50.yaml`
requires authorized baseline features, fold manifests and preceding frozen
reference runs; these are not included. Historical cross-study scripts and
configs remain available as source reference. Replace `/path/to/research` and
other placeholders with your own local inputs before using them. There is no
claim that a code-only clone contains the historical data or trained models.

## Connect Generated MRI to pCR

1. Run the repository's `world.py sample` with its matching generator and VQ
   checkpoints. Its NPZ contains `images` with shape `[draws,3,32,128,128]`.
2. Encode each observed/generated visit using the original image normalization,
   real T0 foreground, physical-spacing preprocessing and local frozen Pillar:

```bash
python pcr/run.py encode --images /path/to/forecast.npz \
  --foreground /path/to/t0_foreground.npy --normalization /path/to/normalization.json \
  --pillar /path/to/local/pillar-model --output /path/to/visit_features.npy
```

Commands in this section run from the repository root. `encode` reuses the
historical `build_volume` and `pillar_forward`; it performs no weight download.
The matching Pillar model code and authorized weights must already be local.

3. Export each trusted classifier checkpoint into a bundle without patient IDs:

```bash
python pcr/run.py export --checkpoint /path/to/fit/model.pt --output /path/to/fold0.pt
```

4. Assemble NPZ arrays in patient-aligned order: `embeddings` is
   `[draws,patients,4,1152]`, `masks` and `days` are `[patients,4]`, `clinical` is
   `[patients,17]`. T0 is real; T1-T3 use the requested generated trajectories.
   Clinical order is `src.data.TABULAR_FEATURE_NAMES`; use `build_tabular_full`
   to construct it. No label or patient identifier array is accepted.

```bash
python pcr/run.py predict --bundles /path/to/fold0.pt /path/to/fold1.pt \
  /path/to/fold2.pt /path/to/fold3.pt /path/to/fold4.pt \
  --inputs /path/to/trajectories.npz --output /path/to/pcr_predictions.npz
```

The input wrapper retains EmbStore's per-visit L2 normalization and contiguous
prefix masking. It averages probabilities across complete trajectories within
each classifier, then across folds for one classifier seed. It does not average
images, embeddings, logits, or different seeds. It also returns the per-fold
probabilities, so mean-fold AUROC and AUROC of mean probabilities can be
reported separately. One bundle is accepted for individual-fold prediction.

The historical 102-patient evaluation cohort was also used for generator
development. Results are exploratory, not an untouched end-to-end test.
