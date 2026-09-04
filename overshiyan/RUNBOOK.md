# Operator Runbook

All commands assume the current directory is the uploaded `overshiyan` folder.

## 1. Training host preflight

```bash
python -m pip install -r requirements-train.txt
python scripts/00_fetch_yolo11x.py
python scripts/00_env_check.py --target train5090
python scripts/01_audit_data3.py --strict
python scripts/02_prepare_data3_view.py
python scripts/02_validate_ultralytics_dataset.py
python scripts/03_train_exp006.py --preflight-only
```

The audit must report 4481 images, 4481 labels, four intentional empty labels,
25 classes, no missing pairs, and no invalid boxes. The prepared view removes
only exact duplicate rows inside a label file; it never edits `data3/`.
Images are copied by default so generated training files cannot mutate the source
through a hard link. Use the optional hard-link mode only when disk constraints
are understood and the workspace is kept read-only.

## 2. EXP006-X0 training

```bash
python scripts/03_train_exp006.py
```

For a remote host without `tmux` or `screen`, start one persistent process and
record its PID:

```bash
mkdir -p runs/train
pgrep -af "03_train_exp006.py" || true
nohup python -u scripts/03_train_exp006.py \
  > runs/train/EXP006_console.log 2>&1 &
echo $! > runs/train/EXP006.pid
```

Only run the `nohup` command when `pgrep` confirms that no EXP006 process exists.
Monitor it with `ps -fp "$(cat runs/train/EXP006.pid)"` and
`tail -f runs/train/EXP006_console.log`; leaving `tail` with `Ctrl+C` does not stop
training.

The schedule is at most 60 epochs, saves a checkpoint every 5 epochs, and has a
3.75-hour wall-clock guard. At the guard it finishes the current epoch, validates,
and saves normally, so total wall time should remain close to four hours. The
default configuration uses `batch=8`. If the 5090 reports CUDA OOM, edit
only `batch` to 6 or 4 in the experiment YAML and record the change before
restarting. Do not reduce image size or change augmentations during the same run.
After each epoch, `epoch_timings.csv` and the console show the measured per-epoch
time and a projected 60-epoch duration. Use the three most recent stable epochs,
not the slower first epoch, when judging the estimate.

Resume only an interrupted run:

```bash
python scripts/03_train_exp006.py --resume runs/train/EXP006_X0_data3_yolo11x_1024/weights/last.pt
```

## 3. Candidate inference

Evaluate a checkpoint on the fixed 897-image Data3 validation view:

```bash
python scripts/04_infer_checkpoint.py \
  --model runs/train/EXP006_X0_data3_yolo11x_1024/weights/best.pt \
  --output-dir runs/eval/EXP006_best
```

For checkpoint comparison, repeat with `last.pt` and useful periodic checkpoints.
The inference code has separate limits for each tile and the full image so a
10k image is not accidentally capped at 300 detections.

## 4. Dual-protocol scoring

```bash
python scripts/05_evaluate_protocols.py \
  --predictions runs/eval/EXP006_best/result.json \
  --timings runs/eval/EXP006_best/timings.json \
  --output-dir runs/eval/EXP006_best/metrics
```

This scores identical predictions against:

- D3: the manually revised validation labels in the prepared view.
- D0: the frozen original-label validation protocol mapped to sequential IDs.

`protocol_comparison.json` contains the worst-protocol Recall/FDR used for model
selection. A high score on only one protocol is not considered enough evidence.
Smoke/sample metrics are marked non-full and are rejected by candidate ranking
and freezing, even when their small-sample gates happen to pass.

## 5. Candidate ranking

```bash
python scripts/06_rank_candidates.py \
  --metrics runs/eval/EXP006_best/metrics/protocol_comparison.json \
  --metrics runs/eval/EXP006_last/metrics/protocol_comparison.json \
  --output runs/eval/candidate_ranking.csv
```

The ranker prioritizes passing both protocols, then the larger gate margin. It
does not use Ultralytics fitness as the final competition criterion.

## 6. Freeze and 3090 submission preparation

```bash
python scripts/07_freeze_candidate.py \
  --model runs/train/EXP006_X0_data3_yolo11x_1024/weights/best.pt \
  --metrics runs/eval/EXP006_best/metrics/protocol_comparison.json \
  --name EXP006_X0_best
```

The command copies the selected model to `models/frozen/`, records hashes, and
copies it into `submit/models/best.pt`. Build and benchmark the Docker image on
an RTX 3090 before any registry push. The submission defaults are provisional
until confidence calibration is complete.

From the `overshiyan` directory:

```powershell
docker build -f submit/Dockerfile -t weixing-submission:x6 .
docker run --rm --gpus all --network none `
  -v "${PWD}\submit\test-input:/input:ro" `
  -v "${PWD}\submit\test-output:/output" `
  weixing-submission:x6 --input /input --output /output
```

Test `tile_batch` values 1, 2, and 4 on the 3090. Keep the fastest setting that
has stable memory headroom on the largest images; never select from 5090 timing.
