# Code Release, 2026-09-27

Source snapshots: Symm-FM Threephase Observation V3;
longitudinal_temporal_pillar; the MeWM-ISPY2 ROI32 dependency package.
The generator and classifier model/training implementations are retained.
Upstream authorship and licenses are retained.

Packaging changes:

- Added the complete external pCR source under `pcr/` and a repository overview.
- Added input-only pCR bundle export, local Pillar encoding and prediction.
- Added optional PCR dependencies to root package metadata.
- Bundled original ROI32 dependency modules and configuration templates under
  `vendor/mewm-ispy2/`.
- Replaced machine-specific roots with generic placeholders; the changed file
  list is in `PATH_REDACTIONS.json`. Historical dataset references still need
  local configuration; the portable prediction CLI uses explicit input paths.
- Marked the historical-fold integration test as requiring private artifacts
  when its input files are absent. Its assertions are unchanged.

V3 contains no internal PCR auxiliary head. The external Pillar/TDN workflow is
the same classifier source as in the V2 release. Generator versions V2/V3 must
not be confused with classifier recipe versions V1/V4.

`GENERATION_GUIDE.md` preserves the original generator guide. Its original
delivery-time reports are distinct from this release's `VALIDATION.md`.
No foundation weights, fitted cohort checkpoints, MRI/CT images, labels, patient
splits, per-patient outputs, credentials or original machine roots are published.
Full trained-weight/real-image end-to-end replay was not performed for this
code release; the new pCR adapter has synthetic parity tests.
