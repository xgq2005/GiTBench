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

`policy/testpolicy/` is the reference template. Copy that folder for a new
model, update `policy:` in `eval_config.yaml`, then implement the policy
factory in `__init__.py`.

`testpolicy` differs from a normal model only in its action output: it always
returns a zero action. It uses the same environment, demonstration, evaluator,
step budget, video recorder, log format, and configuration launcher as every
other policy. Its `eval_config.yaml` sets `stop_on_success: false` so a smoke
run continues to the configured `max_steps`; normal policies can keep the
default `true` to stop as soon as the benchmark success rule is met.

Run the test policy:

```bash
bash policy/testpolicy/evaluate.sh
```

Check the resolved command and configuration without starting a simulator:

```bash
bash policy/testpolicy/evaluate.sh --dry-run
```

The important `eval_config.yaml` fields are:

```yaml
policy: testpolicy:make_policy
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
  fps: 30
resume: true
output:
  root: eval_results
  run_name: testpolicy
  timestamp: true
  run_dir: null      # set an existing run directory to continue it
```

The launcher creates one timestamped run directory and saves the exact
resolved configuration as `eval_config.resolved.yaml` beside `log.json` and
`summary.json`. Set `output.run_dir` to a previous result directory with
`resume: true` to continue an interrupted evaluation.
