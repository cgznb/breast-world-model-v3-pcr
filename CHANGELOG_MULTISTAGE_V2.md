# Multistage R1 Upgrade

Implemented in the original training source `/home/cxx/test/breast_world_joint`.
The GitHub publication tree `/home/cxx/test/medical-world-models` was not edited.
The starting source is not a Git working tree; no HEAD or research branch is
invented. Its historical launch snapshot remains at
`/data1/cxx/breast_world_joint_20260928/source`.

- Preserve responsewm_v1 configuration digests, baseline model and training CLI.
- Add strict patient trajectory schema, source-visible prefixes, four separate
  masks, split/codec audit, train-only normalization and unbiased edge sampling.
- Add canonical T0/T1/T2/T3 dynamics, persistent belief, real-observation
  assimilation, evidence deduplication, branch invalidation and independent RNG.
- Add shared pCR readout for observed, simulated and marginalized outcomes.
- Preserve the full VQ/three-phase encoder/native U-Net. Add three interleaved
  image-semantic bridges, six semantic blocks, typed interval conditioning and
  same-visit deep JEPA. The pinned MONAI backend also has a tested split traversal.
- Add paired FM, differentiable free rollouts, stage/joint Energy Scores,
  posterior observation training and posterior-to-next-edge supervision.
- Add four-stage curriculum, validation history, early stopping, controlled
  image learning rate, parameter/RNG/sampler resume, strict migration reports,
  input-only deployment, prequential evaluation and executable ablation configs.

No clinical outcome improvement is claimed. R2 physical-time modeling is disabled.
