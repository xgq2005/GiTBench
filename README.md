# GiTBench

> GiTBench ：a simulation benchmark for temporal grounding in robotic manipulation.

<p align="center">
  <img src="asset/gitbench.png" alt="GITBench benchmark teaser" width="90%">
</p>

<p align="center">
  <a href="https://kk-stephen.github.io/grounded-in-time/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-GiT-blue" alt="GiT website"></a>
  <a href="https://github.com/xgq2005/GiT"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="GiT GitHub repository"></a>
  <a href="<ARXIV_URL>"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv paper"></a>
  <a href="https://huggingface.co/datasets/XGQ12345/GiTBench_sim_train_lerobotv3"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow" alt="GiTBench LeRobot dataset"></a>
  <a href="https://www.modelscope.cn/datasets/xgq12345/GiTBench_sim_train_lerobotv3"><img src="https://img.shields.io/badge/ModelScope-Dataset-blue" alt="GiTBench LeRobot dataset"></a>
</p>

<p align="center">
  <a href="https://kk-stephen.github.io/grounded-in-time/">Website</a> ·
  <a href="https://github.com/xgq2005/GiTBench">GitHub</a> ·
  <a href="https://arxiv.org/abs/2610.04255">arXiv</a> ·
  <a href="https://huggingface.co/datasets/XGQ12345/GiTBench_sim_train_lerobotv3">Hugging Face</a> ·
  <a href="https://www.modelscope.cn/datasets/xgq12345/GiTBench_sim_train_lerobotv3">ModelScope</a>
</p>

GiTBench is a simulation benchmark for temporal grounding in robot manipulation under perceptual ambiguity. The target object cannot always be identified from the current frame alone: several objects may look similar, while the instruction refers to an event in the episode history, such as the object filled last, the object placed first, or the object just handed over. Solving the task therefore requires using the demonstration and interaction history together with the current visual observation.

This repository packages the benchmark itself: its fixed simulator episodes, task manifest, local assets, evaluation scripts, and common policy interface. It contains nine simulation tasks and is designed to measure whether a policy can recover the temporally grounded reference and execute the corresponding manipulation.

## 🎯 Benchmark scope

The benchmark contains nine tasks:

- Laboratory: `bio2`, `bio4`, `bio5`
- Industrial: `ind2`, `ind3`, `ind5`
- Household: `house1`, `house3`, `house5`

Each task has 60 fixed test episodes, for 540 episodes total. The manifest divides them into `test-ID` (in-distribution), `test-CF` (counterfactual), and `test-OOD` (out-of-distribution) cases.

The repository includes the required benchmark assets and a vendored ManiSkill runtime. A separate simulator repository or external asset directory is not needed for evaluation.

## 🛠️ Install the benchmark environment

Use Python 3.10. Install a PyTorch build that matches your CUDA driver. The command below is an example for PyTorch 2.7 with CUDA 12.8; adjust the versions and index URL for your machine.

```bash
conda create -n gitbench python=3.10 -y
conda activate gitbench
python -m pip install --upgrade pip setuptools wheel

python -m pip install "torch==2.7.*" "torchvision==0.22.*" \
  --index-url https://download.pytorch.org/whl/cu128

# Run these commands from the GITBench repository root.
cd third_party/ManiSkill-3
python -m pip install -e .
cd ../..
python -m pip install -r requirements.txt
python -m pip install -e .
```

The local ManiSkill installation is required because GITBench imports the runtime from `third_party/ManiSkill-3`. The root requirements install the remaining benchmark dependencies. `opencv-python` and data-collection dependencies are not required for evaluation.

## 🤖 Train The  Models

The repository currently supports exactly three trainable models. All three use
the same LeRobot v3 dataset, while their weights and training outputs stay in
separate policy folders:

| Model  | Implementation     | Training entry point       | Evaluation entry point            |
| ------ | ------------------ | -------------------------- | --------------------------------- |
| pi05   | JAX/OpenPI         | `policy/pi05/train.sh`   | `policy/pi05/evaluate_local.sh` |
| RDT    | PyTorch RDT-1B     | `policy/RDT/train.sh`    | `policy/RDT/evaluate.sh`        |
| HAMLET | PyTorch GR00T N1.6 | `policy/HAMLET/train.sh` | `policy/HAMLET/evaluate.sh`     |

```text
data/GiTBench_train/                         # LeRobot v3 dataset
policy/pi05/model/pi05_base/                 # pi05 base parameters/assets
policy/pi05/checkpoints/                     # pi05 fine-tuned checkpoints
policy/RDT/checkpoints/rdt-1b/               # RDT-1B base checkpoint
policy/RDT/google/t5-v1_1-xxl/               # RDT language encoder
policy/RDT/google/siglip-so400m-patch14-384/ # RDT vision encoder
policy/RDT/checkpoints/rdt-gitbench/         # RDT fine-tuned checkpoints
policy/HAMLET/models/GR00T-N1.6-3B/          # HAMLET/GR00T base model
policy/HAMLET/runs/gitbench/                 # HAMLET fine-tuned checkpoints
```

The commands below assume they are run from the repository root. The dataset
contains 1,440 episodes, 671,299 frames, 35 task instructions, and three RGB
views. It is expected at `data/GiTBench_train`.

### 1. Download The Dataset

Download the dataset from either source and place its contents under exactly
`data/GiTBench_train/`:

| Source       | URL                                                                                                                                                 | Target directory         |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Hugging Face | [https://huggingface.co/datasets/XGQ12345/GiTBench_sim_train_lerobotv3](https://huggingface.co/datasets/XGQ12345/GiTBench_sim_train_lerobotv3)       | `data/GiTBench_train/` |
| ModelScope   | [https://www.modelscope.cn/datasets/xgq12345/GiTBench_sim_train_lerobotv3](https://www.modelscope.cn/datasets/xgq12345/GiTBench_sim_train_lerobotv3) | `data/GiTBench_train/` |

Check the result before training:

```bash
test -f data/GiTBench_train/meta/info.json
test -f data/GiTBench_train/meta/tasks.parquet
test -f data/GiTBench_train/meta/episodes/chunk-000/file-000.parquet
```

### 2. pi05 (JAX/OpenPI)

This is the JAX implementation of pi05. Download the OpenPI base checkpoint
from [https://console.cloud.google.com/storage/browser/openpi-assets/checkpoints/pi05_base](https://console.cloud.google.com/storage/browser/openpi-assets/checkpoints/pi05_base)
and place it at `policy/pi05/model/pi05_base/`. The directory must contain
`params/manifest.ocdbt` and the `assets/` subdirectory.

Install the policy dependencies in an environment containing JAX, Flax,
LeRobot, and OpenPI. The launcher can be pointed at any interpreter with
`OPENPI_PYTHON`:

```bash
conda create -n gitbench-pi05 python=3.11 -y
conda activate gitbench-pi05
python -m pip install -r policy/pi05/requirements.txt
python -m pip install -e third_party/ManiSkill-3
```

Fine-tune with LoRA (the default) or full weights:

```bash
OPENPI_PYTHON="$CONDA_PREFIX/bin/python" \
bash policy/pi05/train.sh lora pi05_gitbench_lora

OPENPI_PYTHON="$CONDA_PREFIX/bin/python" \
bash policy/pi05/train.sh full pi05_gitbench_full
```

The script computes normalization statistics from `data/GiTBench_train` before
training. With the commands above, LoRA checkpoints are written under
`policy/pi05/checkpoints/pi05_gitbench_lora/pi05_gitbench_lora/`; full
fine-tuning uses the analogous `pi05_gitbench` directory. The second argument
to `train.sh` controls the final experiment-directory name. Evaluate a
checkpoint locally with the GITBench evaluator:

```bash
PI05_EVAL_PYTHON="$CONDA_PREFIX/bin/python" \
bash policy/pi05/evaluate_local.sh \
  policy/pi05/checkpoints/pi05_gitbench_lora/pi05_gitbench_lora test
```

To run only selected tasks, append task keys such as `bio2 bio4`. The default
configuration is `pi05_gitbench_lora`; use
`PI05_CONFIG=pi05_gitbench` for a full-finetune checkpoint when needed.

### 3. RDT

RDT requires three local assets:

| Asset                   | URL                                                                                                                         | Target directory                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| RDT-1B base checkpoint  | [https://huggingface.co/robotics-diffusion-transformer/rdt-1b](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) | `policy/RDT/checkpoints/rdt-1b/`               |
| T5-XXL language encoder | [https://huggingface.co/google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl)                                       | `policy/RDT/google/t5-v1_1-xxl/`               |
| SigLIP vision encoder   | [https://huggingface.co/google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)           | `policy/RDT/google/siglip-so400m-patch14-384/` |

Install RDT dependencies and use the RDT environment for both training and
evaluation:

```bash
conda create -n gitbench-rdt python=3.10 -y
conda activate gitbench-rdt
python -m pip install -r policy/RDT/requirements.txt
```

Fine-tune RDT on the LeRobot dataset. The launcher first encodes the 35 task
instructions into `policy/RDT/outs/gitbench_lang_embeds/`, then starts training:

```bash
RDT_PYTHON="$CONDA_PREFIX/bin/python" \
bash policy/RDT/train.sh
```

The default base checkpoint is `policy/RDT/checkpoints/rdt-1b`; fine-tuned
outputs go to `policy/RDT/checkpoints/rdt-gitbench/`. Resume a run with:

```bash
RDT_PYTHON="$CONDA_PREFIX/bin/python" \
RESUME_FROM_CHECKPOINT=latest \
bash policy/RDT/train.sh
```

Evaluate a specific RDT checkpoint. The evaluation script creates any missing
test-instruction embeddings under `policy/RDT/outs/gitbench_lang_embeds/`:

```bash
RDT_PYTHON="$CONDA_PREFIX/bin/python" \
RDT_CHECKPOINT="$PWD/policy/RDT/checkpoints/rdt-gitbench/checkpoint-50000" \
bash policy/RDT/evaluate.sh --episodes 1 --no-video
```

Use `bash policy/RDT/evaluate.sh --dry-run` to inspect the resolved checkpoint
and output paths without launching the simulator.

### 4. HAMLET / GR00T

HAMLET uses NVIDIA's GR00T N1.6 base model. Download it from
[https://huggingface.co/nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) and place it at
`policy/HAMLET/models/GR00T-N1.6-3B/`.

Install the HAMLET/GR00T environment according to
`policy/HAMLET/pyproject.toml`, then run the GITBench-specific launcher:

```bash
conda activate hamlet
cd policy/HAMLET
uv sync
uv pip install -e .
cd ../..

HAMLET_PYTHON="$CONDA_PREFIX/bin/python" \
bash policy/HAMLET/train.sh
```

The launcher reads `data/GiTBench_train`, uses
`policy/HAMLET/gr00t/configs/data/gitbench_config.py`, and writes checkpoints
under `policy/HAMLET/runs/gitbench/hamlet_n1d6/`. Evaluate the resulting
checkpoint with:

```bash
HAMLET_PYTHON="$CONDA_PREFIX/bin/python" \
HAMLET_MODEL_PATH="$PWD/policy/HAMLET/runs/gitbench/hamlet_n1d6/checkpoint-60000" \
bash policy/HAMLET/evaluate.sh
```

Use `HAMLET_MODEL_PATH=... bash policy/HAMLET/evaluate.sh --dry-run` first to
verify the policy path and evaluator command. Set `NUM_GPUS`, `GLOBAL_BATCH_SIZE`,
`MAX_STEPS`, or `OUTPUT_DIR` before `train.sh` to change the training run.

## 💨 Run a smoke evaluation

The reference `Your_Policy` returns zero actions and checks the complete evaluation pipeline. It runs one episode for each configured task and writes results under `eval_results/`.

```bash
bash policy/Your_Policy/evaluate.sh
```

Check the resolved command without starting the simulator:

```bash
bash policy/Your_Policy/evaluate.sh --dry-run
```

## 🚀 Run evaluations directly

Use the built-in `zero` policy for a minimal direct run:

```bash
python scripts/eval_policy.py \
  --tasks bio2 --split test --episodes 1 \
  --policy zero --render-mode none
```

Run all nine tasks and all 540 test episodes without videos:

```bash
python scripts/eval_policy.py \
  --tasks bio2 bio4 bio5 ind2 ind3 ind5 house1 house3 house5 \
  --split test --policy zero --render-mode none \
  --no-stop-on-success
```

The `test` split selects the complete 60-episode set for each task. Use `--split test-ID`, `--split test-CF`, or `--split test-OOD` to evaluate one subset. Omit `--episodes` to run every episode in the selected split.

Common options:

- `--tasks`: one or more task keys;
- `--split`: `test`, `test-ID`, `test-CF`, or `test-OOD`;
- `--episodes`: per-task episode limit;
- `--start-episode`: offset within the selected split;
- `--max-steps`: override the task step limit;
- `--stop-on-success` / `--no-stop-on-success`: stop at success or continue to the step limit;
- `--resume` / `--no-resume`: reuse or ignore an existing log;
- `--save-video --video-dir eval_results/videos`: record videos;
- `--out` and `--summary-out`: choose log and summary paths.

The policy launcher reads `policy/Your_Policy/eval_config.yaml`. Set `episodes: null` there to run every episode in the configured split. Its standard output directory is a timestamped subdirectory of `eval_results/` containing `log.json`, `summary.json`, the resolved configuration, and optional videos.

## 📊 Results and scoring

Each episode is classified as:

- `success`: the task goal is completed;
- `failure`: the goal is not completed, an incorrect interaction occurs, or the policy raises an exception;
- `error`: an evaluation infrastructure error, excluded from the formal success rate.

The primary metric is episode-level success rate. If success and failure occur on the same step, failure takes precedence.

Process completion is a supplementary `[0, 1]` metric based on weighted milestones such as contact, grasping, movement, and reaching a goal. It does not change the binary result. See `PROCESS_COMPLETION.md` for the milestone rules, dependencies, log fields, and aggregate metrics.

## 🧩 Custom policies

Add a package under `policy/` with a `make_policy(action_space)` factory. The returned object must implement `act(obs, info=None)` and may implement `reset(...)`. Use `policy/Your_Policy` as the minimal template:

```text
policy/<policy_name>/
├── __init__.py       # exposes make_policy(action_space)
├── eval_config.yaml  # tasks, split, rendering, output, and resume settings
└── evaluate.sh       # evaluation entry point
```

For a standard launcher, copy the reference policy, update the `policy:` value in `eval_config.yaml`, and implement the policy factory. The three model integrations above have model-specific environments and dependencies; they are optional and are not needed for the built-in benchmark or reference policy.

## 📖 Citation

If you find this benchmark useful, please cite:

```bibtex
@article{groundedintime,
  title   = {Grounded in Time: Benchmarking Temporal Grounding under
             Perceptual Ambiguity in Robotic Manipulation},
  author  = {Yi Wang and Yang Yang and Guangqi Xu and Sumin Lin and Ning Kang and
             Pengxiang Lu and Xiaotong Chen and Zeyu Xue and Chenguang Yang and Zhenyu Lu},
  journal = {arXiv preprint arXiv:<ARXIV_ID>},
  year    = {2026}
}
```

## 🙏 Acknowledgments

We thank the ManiSkill-3 team for their open-source simulation platform and the work that makes this benchmark's manipulation environments possible.

## 📬 Contact

For questions, issues, or collaboration, please open an issue on the [GitHub repository](https://github.com/xgq2005/GiTBench).
