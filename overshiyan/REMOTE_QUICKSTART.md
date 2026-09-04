# Remote Quick Start

Keep SSH private keys and passwords local. The operator performs transfer and
starts the GPU job; no credential is required inside this project.

## Package on Windows

From `D:\daima\weixing\overshiyan`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_remote_bundle.ps1
```

The script creates a tar archive and neighboring `.sha256` file under
`D:\daima\weixing`. It includes Data3 and the verified `yolo11x.pt`, while
excluding regenerated workspace files and local debug runs.

## Upload option A: provider web panel

Upload both the `.tar` and `.tar.sha256` files to the instance data disk, then
open its terminal.

## Upload option B: SCP

Use the host, port, user, and key path shown by the provider:

```powershell
scp -P <PORT> -i "C:\path\to\private_key" `
  "D:\daima\weixing\overshiyan_remote_20260904.tar" "D:\daima\weixing\overshiyan_remote_20260904.tar.sha256" `
  <USER>@<HOST>:/data/coding/
```

If the provider uses password login, omit `-i` and enter the password locally.
Do not paste the credential into chat or save it in this directory.

## Verify and unpack on Linux

```bash
cd /data/coding
stat -c '%n %s bytes' overshiyan_remote_20260904.tar
sha256sum -c overshiyan_remote_20260904.tar.sha256
tar -tf overshiyan_remote_20260904.tar >/dev/null && echo "tar archive OK"
tar -xf overshiyan_remote_20260904.tar
cd overshiyan
```

For the 2026-09-04 bundle, the archive size is `1350321152` bytes and SHA256 is
`3476525e3cbc7fc4f9e6d9acef607e9ccfe87ea03718faeaadbd9696f411784f`.
Do not extract or execute a bundle unless both the checksum and tar integrity
checks pass. Provider mount points vary; replace `/data/coding` consistently if
the uploaded files are stored elsewhere.

## Prepare and start training

```bash
python -m pip install -r requirements-train.txt
python scripts/00_fetch_yolo11x.py
python scripts/00_env_check.py --target train5090
python scripts/01_audit_data3.py --strict
python scripts/02_prepare_data3_view.py
python scripts/02_validate_ultralytics_dataset.py
python scripts/03_train_exp006.py --preflight-only
python scripts/03_train_exp006.py
```

The fetch command is idempotent and only verifies the bundled checkpoint when it
already exists. Training is capped at 60 epochs and 3.75 hours. A foreground
training process must not be left attached to a disposable web terminal; use
`tmux`, `screen`, `nohup`, or the provider's persistent job feature.

Recommended `tmux` launch:

```bash
tmux new -s exp006
cd /data/coding/overshiyan
python scripts/03_train_exp006.py 2>&1 | tee runs/train/EXP006_console.log
```

Detach with `Ctrl+B`, then `D`. Reconnect with `tmux attach -t exp006`.

If `tmux` and `screen` are unavailable, use the reviewed `nohup` fallback:

```bash
cd /data/coding/overshiyan
pgrep -af "03_train_exp006.py" || true
mkdir -p runs/train
nohup python -u scripts/03_train_exp006.py \
  > runs/train/EXP006_console.log 2>&1 &
echo $! > runs/train/EXP006.pid
ps -fp "$(cat runs/train/EXP006.pid)"
tail -n 80 runs/train/EXP006_console.log
```

Closing SSH does not stop this `nohup` process. Stopping, restarting, or releasing
the GPU instance does. Never launch the command again while the recorded PID is
still active.
