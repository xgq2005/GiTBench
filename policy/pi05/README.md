# GITBench pi05

This directory contains vendored OpenPI and the GITBench adapter. Base weights are
at `model/pi05_base/params`; fine-tuned checkpoints are under `checkpoints/`.
SwanLab defaults to project `GITBench`, experiment `pi05`.

## Environment Rules

| Workflow | Environment | Install source |
| --- | --- | --- |
| Train, normalization, smoke test, direct evaluation | `gitbench` | `requirements.txt` in this policy |
| WebSocket server only | separate `openpi` environment | original `pyproject.toml` and `uv.lock` |
| WebSocket benchmark client | `gitbench` | `install_eval_deps.sh` |

All non-WebSocket operations use the common GITBench environment. This policy
`requirements.txt` contains only OpenPI packages missing from the benchmark base
environment; it does not reinstall NumPy, PyTorch, h5py, Pillow, OpenCV,
imageio, PyYAML, tqdm, tyro, filelock, or ManiSkill.

## Common GITBench Setup

```bash
conda activate gitbench
cd /data/wangyi/xgq/GITBenchv4
python -m pip install -r policy/pi05/requirements.txt
```

Training and local inference default to `/data/wangyi/conda/envs/openpi/bin/python`.

## Train And Direct Evaluation

```bash
conda activate gitbench
cd /data/wangyi/xgq/GITBenchv4/policy/pi05
bash smoke_test.sh
bash train.sh lora
# Full fine-tuning:
bash train.sh full

# Direct evaluation:
bash evaluate_local.sh checkpoints/<checkpoint> test bio2
```

The train script uses the LeRobot v3 dataset at
`data/GiTBench_train` and computes normalization statistics before training.
Supply a
different SwanLab experiment name as the second positional argument, e.g.
`bash train.sh lora pi05_run2`.

Training uses LeRobot v3 directly.

## WebSocket Evaluation

Only the policy server uses the original OpenPI locked environment:

```bash
conda create -n openpi python=3.11 -y
conda activate openpi
python -m pip install --upgrade uv
cd /data/wangyi/xgq/GITBenchv4/policy/pi05
GIT_LFS_SKIP_SMUDGE=1 uv venv --python python
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Install the lightweight client in the common benchmark environment, then run
the server and evaluation in separate terminals:

```bash
# In gitbench, once
cd /data/wangyi/xgq/GITBenchv4/policy/pi05
bash install_eval_deps.sh

# In the OpenPI environment
OPENPI_PYTHON="$PWD/.venv/bin/python" bash serve.sh checkpoints/<checkpoint>

# In gitbench
bash evaluate_websocket.sh test bio2
```
