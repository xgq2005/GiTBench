# GITBench RDT

This directory contains RDT-1B at `checkpoints/rdt-1b`, local T5/SigLIP encoders
under `google/`, and simulator fine-tuned models under `checkpoints/rdt-mem-sim`.
SwanLab defaults to project `GITBench`, experiment `rdt`.

## Runtime Environment

RDT has no upstream WebSocket inference mode. Training, T5 embedding generation,
statistics, and end-to-end benchmark evaluation run in process. The launcher uses
the RDT environment for the model stack and appends the ManiSkill environment's
Gym packages without replacing RDT's PyTorch build.

Install only original RDT dependencies that are absent from GITBench:

```bash
conda activate rdt
cd /data/wangyi/xgq/GITBenchv4
python -m pip install -r policy/RDT/requirements.txt
```

The requirements omit benchmark-provided PyTorch, torchvision, NumPy, h5py,
Pillow, OpenCV, imageio, PyYAML, tqdm, packaging, and the simulator stack. They
contain RDT-specific Accelerate, DeepSpeed, Diffusers, Transformers, timm,
Hugging Face, SwanLab, and related packages only. Local model weights are already
present, so no model download is required.

## Train

`train.sh` reads `data/GiTBench_train` directly in LeRobot v3 format, first
creates or reuses cached T5 embeddings, then trains RDT with
`--precomp_lang_embed`; T5-XXL is not loaded in the train loop.

```bash
conda activate rdt
cd /data/wangyi/xgq/GITBenchv4/policy/RDT
bash train.sh
```

The default data is `data/GiTBench_train`. Caches are
under `outs/`. Set `RESUME_FROM_CHECKPOINT=latest`, `CUDA_VISIBLE_DEVICES`,
`NUM_PROCESSES`, `TRAIN_BATCH_SIZE`, or `MAX_TRAIN_STEPS` to adjust a run.

The training launcher uses LeRobot only; state/action, videos, episode metadata,
and `task_index` language are loaded from the LeRobot v3 files.

## Evaluate

RDT evaluation is direct in-process inference, not WebSocket. At startup,
`evaluate.sh` reads public test instructions from the benchmark manifest, encodes
only missing T5 embeddings, caches them under `outs/gitbench_lang_embeds`, and then
starts rollout. No separate embedding step is required. The newest numbered
checkpoint containing `pytorch_model.bin` under `checkpoints/rdt-mem-sim` is
selected automatically.

```bash
conda activate rdt
cd /data/wangyi/xgq/GITBenchv4/policy/RDT
CUDA_VISIBLE_DEVICES=0 bash evaluate.sh
```

Set `RDT_CHECKPOINT` to evaluate a specific checkpoint, or
`RDT_LANG_EMBED_DIR` to use a different embedding cache directory.

Use `bash evaluate.sh --dry-run` to validate the evaluation command without
encoding instructions or launching the simulator.
