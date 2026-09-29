# PicoDual → Piper: chunk-relative pi05

This workflow uses `pick_cube_0928_chunk_relative`: 20 Hz, base and wrist images,
1D gripper state in millimetres, and 7D absolute PICO World TCP actions. Each
sampled 50-step action window is converted to `inv(T_action[0]) @ T_action[k]`;
the gripper target remains an absolute width. The first action is same-frame and
is skipped by the live client. The model action representation remains rotvec.

## Data and statistics

The dataset is generated locally and is ignored by Git. If it is absent, create
it from the untracked `pick_cube_0928` source:

```bash
.venv/bin/python examples/piper/prepare_chunk_relative_eef.py \
  --source pick_cube_0928 --output pick_cube_0928_chunk_relative \
  --robot-open-width-mm 20
```

Compute statistics from valid action rows only; this reads no images. Statistics
and the manifest are written under `assets/` and are ignored by Git. The
statistics must travel with the checkpoint or be recomputed from the same data.

```bash
.venv/bin/python examples/piper/chunk_relative/compute_norm_stats.py
```

The dedicated config is `pi05_piper_pick_cube_0928_chunk_relative`. It does not
reuse the older `pick_cube_chunk_relative` statistics. Sync the dataset separately
when training on another machine.

## Train and serve

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_piper_pick_cube_0928_chunk_relative --exp-name=pick_cube_0928

uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
  --policy.config=pi05_piper_pick_cube_0928_chunk_relative \
  --policy.dir=checkpoints/pi05_piper_pick_cube_0928_chunk_relative/pick_cube_0928/<step>
```

Replace `<step>` with a completed checkpoint step. Before live execution,
check the camera order, TCP axes and origin, gripper feedback, and target motion
using the existing offline evaluator or dry-run replay.

## Live client

The focused client fixes the task prompt, action frame, and 50-step model
horizon. It exposes only the server, camera, and execution-step settings:

```bash
.venv/bin/python examples/piper/chunk_relative/infer.py \
  --base-camera <serial> --wrist-camera <index> --steps 3
```

Start with a short execution block; increase `--steps` only after observing
correct motion. For advanced hardware settings and offline evaluation, use
`examples/piper/inference_eef.py` directly.
