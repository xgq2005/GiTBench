# Policy Folder

Put one policy package per model under this folder. Each package should expose a factory:

```python
def make_policy(action_space):
    return policy
```

The returned object must provide:

```python
action = policy.act(obs, info)
policy.reset(task=spec, episode_id=episode_id, obs=obs, info=info)  # optional
```

`policy.reset(...)` runs after the environment reset and before the first
`act(...)`. `info["demonstration"]` contains the scene-only history frames and
the first post-demonstration interaction frame:

```python
demo = info["demonstration"]
frames = demo["frames"]  # uint8 [T, H, W, 3]
assert frames[demo["interaction_frame_index"]].shape == frames[-1].shape
```

`demo["demo_frame_count"]` excludes the final interaction frame. The
demonstration uses the same priority-selected sensor view as the policy RGB
input (`base_camera`, then `human_cam`, then `sensor_cam`). It contains no
robot actions, joint trajectories, planners, or evaluator-only targets.

Policies that still define `reset(task=None, episode_id=None)` remain
compatible with the evaluator.

## Configured Evaluation

Each policy should keep its evaluation entry point and configuration in its own
folder:

```text
policy/<policy_name>/
├── __init__.py          # exposes make_policy(action_space)
├── eval_config.yaml     # tasks, rendering, videos, output, resume
└── evaluate.sh          # standard shell entry point
```

`policy/Your_Policy/` is the quick-start reference template. Copy that folder
for a new model, update `policy:` in `eval_config.yaml`, then implement model
loading and `predict_action` in `__init__.py`.

`Your_Policy` differs from a normal model only in its default action output: it
returns zero actions. It uses the same environment, demonstration, evaluator,
step budget, video recorder, log format, and configuration launcher as every
other policy. Its `eval_config.yaml` sets `stop_on_success: false` so a smoke
run continues to the configured `max_steps`; normal policies can keep the
default `true` to stop as soon as the benchmark success rule is met.

Run the reference policy:

```bash
bash policy/Your_Policy/evaluate.sh
```

Check the resolved command and configuration without starting a simulator:

```bash
bash policy/Your_Policy/evaluate.sh --dry-run
```

The important `eval_config.yaml` fields are:

```yaml
policy: Your_Policy:make_policy
tasks: [bio2, bio4]
split: test
episodes: null       # null means every episode in the split
start_episode: 0
max_steps: null      # null means use the task catalog default
stop_on_success: true # false runs every interaction step through max_steps
success_mode: ever     # ever latches first success; final scores final state
terminate_on_success: false
video:
  enabled: true
  fps: 10
resume: true
output:
  root: eval_results
  run_name: Your_Policy
  timestamp: true
  run_dir: null      # set an existing run directory to continue it
```

The launcher creates one timestamped run directory and saves the exact
resolved configuration as `eval_config.resolved.yaml` beside `log.json` and
`summary.json`. Set `output.run_dir` to a previous result directory with
`resume: true` to continue an interrupted evaluation.

## HAMLET / GR00T

`policy/HAMLET/` adapts a trained GR00T N1.6 + HAMLET checkpoint to the GITBench
protocol. It maps the two PiperX qpos observations and the front/left/right RGB
cameras into the `NEW_EMBODIMENT` modality used by the RMBench HAMLET recipe,
executes the first `HAMLET_EXEC_CHUNK` actions from each predicted chunk, and
primes the rolling history cache from the reset-time demonstration frames.

Install the dependencies from `policy/HAMLET/pyproject.toml` (or activate the
matching HAMLET environment), then run. On the benchmark machine the launcher
automatically uses `/data/wangyi/conda/envs/hamlet/bin/python`; set
`PYTHON_BIN` or `HAMLET_PYTHON` to override it.

```bash
HAMLET_MODEL_PATH=/path/to/checkpoint-60000 \
HAMLET_DEVICE=cuda \
bash policy/HAMLET/evaluate.sh --dry-run

HAMLET_MODEL_PATH=/path/to/checkpoint-60000 \
bash policy/HAMLET/evaluate.sh
```

The adapter also accepts `HAMLET_EXEC_CHUNK`, `HAMLET_DEMO_PRIME`,
`HAMLET_MEMORY_WINDOW`, `HAMLET_MEMORY_STRIDE`, `HAMLET_STRICT`, and
`HAMLET_DEVICE`. The checkpoint path and these settings are included in the
evaluation resume fingerprint. The bundled workspace checkpoint, when present,
is used if `HAMLET_MODEL_PATH` is omitted.
