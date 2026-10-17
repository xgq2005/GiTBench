# Your_Policy

This is the smallest ready-to-copy GITBench policy package. It runs out of the
box with zero actions, so the evaluator can be checked before loading a model.

## Add a model

Edit `__init__.py`:

1. Load the checkpoint in `YourPolicy.__init__`.
2. Replace `predict_action` with one model inference call.
3. Return an action matching `self.action_space`. Current GITBench tasks use a
   dictionary containing `piper_x-0` and `piper_x-1`, with seven values per arm.

The evaluator calls `reset(...)` once per episode and `act(obs, info)` on every
control step. `info` contains the instruction and may contain demonstration
frames. The template already extracts the instruction during reset.

## Evaluate

```bash
bash policy/Your_Policy/evaluate.sh --dry-run
bash policy/Your_Policy/evaluate.sh
```

Edit `eval_config.yaml` to select tasks, episodes, rendering, videos, and output
paths. Results are written under `eval_results/Your_Policy_*` by default.

`deploy_policy.py` is optional. It demonstrates a separate cached model object
for integrations that need one; the standard GITBench launcher does not require
it.
