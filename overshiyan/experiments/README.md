# Experiment policy

One experiment changes one primary factor. `EXP006-X0` measures the effect of
YOLO11x on the cleaned Data3 supervision. `EXP007-X1` may later measure rare
ship-class sampling while keeping validation unchanged. Copy-paste, FSC hard
negatives, 1280 fine-tuning, model ensembling, and modality postprocessing each
require separate experiment IDs and evidence.

Official submissions are frozen only after: source audit, fixed-split inference,
D3 and D0 scoring, 3090 Docker smoke/full validation, output-schema validation,
and SHA256 recording.

Current run: `EXP006-X0` is running on RTX 5090. Its immutable launch evidence,
bundle/data/config hashes, transfer incident, and interim health observations are
recorded in
[`run_manifests/20260904_exp006_x0_remote_train.md`](run_manifests/20260904_exp006_x0_remote_train.md).
`EXP007-X1` remains blocked until EXP006 checkpoints complete dual-protocol
evaluation.
