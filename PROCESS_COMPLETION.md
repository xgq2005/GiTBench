# GITBench Process-Completion Scoring

This document describes GITBench's process-completion metric. It reports how many task stages a policy completed before failure. It supplements the binary success/failure result and does not replace the benchmark's success determination.

## 1. Definition

Each episode is evaluated with weighted milestones. Once a milestone is achieved, it is latched, and the process score is the sum of the weights of achieved milestones:

```text
process_completion(t) = Σ weight(m), where m was achieved before step t
```

Every milestone weight must be positive, and the weights must sum to `1.0`. The process-completion range is therefore `[0.0, 1.0]`. For example:

```text
grasp 0.20 + move 0.20 + place in goal area 0.60 = 1.00
```

The score can stay the same or increase. It is not reduced if an object later leaves a goal area. It represents whether a stage was ever completed rather than the object's instantaneous pose at every step.

## 2. Evaluation Time and Baseline

Process evaluation starts when the policy begins formal interaction:

1. Reset the environment.
2. Apply the episode layout.
3. Replay the public demonstration/history.
4. Record object-position baselines after replay finishes.
5. After each action from `policy.act()` is executed by `env.step()`, evaluate the milestones.

Actions completed during the demonstration do not directly contribute process credit. The baseline for an object leaving its initial position is also recorded after demonstration replay.

## 3. Supported Milestone Rules

The current version supports these rules:

| Rule | Meaning | Common parameters |
| --- | --- | --- |
| `object_touched` | At least one gripper finger makes effective contact with the target object | `object_ref`, `min_force`, `stable_steps` |
| `object_grasped` | The robot stably holds the target object | `object_ref`, `stable_steps` |
| `object_displaced_from_reset` | The object moves from the policy-start baseline by at least a threshold | `object_ref`, `threshold`, `metric` |
| `object_lifted_from_reset` | The object rises from the baseline by at least a height threshold | `object_ref`, `height` |
| `object_in_rectangle` | The object enters a rectangular region | `object_ref`, `center`, `half_extents`, `metric` |
| `object_near_point` | The object is no farther from a target point than the threshold | `object_ref`, `target`, `threshold`, `metric` |
| `object_in_circle` | The object is within the specified radius of a circle center | `object_ref`, `center`, `radius`, `metric` |
| `object_near_initial_position` | The object returns near its episode-initial position | `object_ref`, `threshold`, `metric` |
| `lid_open` | The specified lid is open | `object_ref` |
| `plug_inserted` | The plug satisfies the target socket's alignment and insertion-depth conditions | `object_ref`, `socket_ref`, `lateral_threshold`, `min_insert_depth` |

Additional behavior:

- `metric: "xy"` measures planar XY distance; `metric: "xyz"` measures three-dimensional distance.
- `stable_steps` is the number of consecutive policy-action steps for which a rule must hold. The count resets when the condition is not met.
- `object_touched` requires only one gripper finger to reach `min_force` (default `0.05`). It is weaker than `object_grasped` and does not treat a single-finger contact as a grasp.
- A milestone is latched once achieved and is not scored again.

## 4. Default Stages

When an episode does not explicitly provide `evaluator.progress`, the system generates default milestones from the final success rule.

### 4.1 Multi-object placement: `objects_in_pad`

Each target object receives a contact milestone and a placement milestone. By default, `0.10` of each object's share is assigned to correct contact and `0.90` to entering the goal region. With four target objects, each object receives `0.025` for contact and `0.225` for placement.

```text
C2_touched  0.025   + C2_in_goal  0.225
C3_touched  0.025   + C3_in_goal  0.225
C4_touched  0.025   + C4_in_goal  0.225
C5_touched  0.025   + C5_in_goal  0.225
```

An object earns its placement share when it enters its target rectangle.

### 4.2 Opening a lid: `lid_open`

The default stages are contact with the lid and opening the lid:

```text
target_lid_touched  0.15
target_lid_open     0.85
```

The process-completion score reaches `1.0` when the lid satisfies the opening condition.

### 4.3 Restoring a plug: `plug_inserted`

The default stages are:

```text
target_plug_touched    0.10
target_plug_grasped    0.15
target_plug_moved      0.10
target_plug_inserted   0.65
```

The insertion stage requires `target_plug_moved`; the insertion share is not awarded until the plug has first moved.

### 4.4 Returning to the initial position: `object_near_initial_position`

The default stages are:

```text
target_touched               0.10
target_left_initial_position 0.25
target_returned              0.65
```

This prevents an object that never moved, or that was already near the target after reset, from being treated as successfully returned.

### 4.5 General movement to a goal: rectangle, circle, or point

For `object_in_rectangle`, `object_in_circle`, and `object_near_point`, the default stages are:

```text
target_touched  0.10
target_grasped  0.15
target_moved    0.15
target_in_goal  0.60
```

Grasping requires contact, movement requires a grasp, and entering the goal requires movement. Contact alone receives a small share; a collision or push does not directly receive the movement share.

## 5. Dependencies and Stability

Use `requires` to specify prerequisite stages. Configuration loading checks that milestone IDs are non-empty and unique, every rule is supported, weights are finite and positive and sum to `1.0`, referenced dependency IDs exist, the dependency graph has no cycles, `stable_steps` is a positive integer, and `object_touched.min_force` is a non-negative finite number.

```json
{
  "version": "gitbench_progress_v1",
  "aggregation": "weighted_sum",
  "milestones": [
    {
      "id": "grasp",
      "weight": 0.3,
      "rule": "object_grasped",
      "params": {"object_ref": "P3", "stable_steps": 2}
    },
    {
      "id": "goal",
      "weight": 0.7,
      "rule": "object_in_circle",
      "params": {"object_ref": "P3", "center": [-0.5, 0.0], "radius": 0.08, "metric": "xy"},
      "requires": ["grasp"]
    }
  ]
}
```

## 6. Failure and Success Handling

Each action step executes the action, checks benchmark success, checks private failure rules, gives failure precedence when both occur on the same step, and updates milestones only on non-failure steps.

- The failure step does not add process credit; the score remains at its last pre-failure value.
- If the benchmark succeeds without failure, unfinished milestones are forcibly marked achieved and completion is set to `1.0`. These milestones include `forced_by_success: true` for auditing.
- Infrastructure errors (environment reset, simulation, or evaluator exceptions) have no process score and record `null`.
- A policy-load failure has no process score. A policy failure during reset or action generation can include process completion only when interaction was established and a process record exists.

Process completion does not change the binary `success`, `failure`, and `error` categories.

## 7. Log Fields

Every episode that completes reset normally includes the following in `log.json`:

```json
"progress": {
  "version": "gitbench_progress_v1",
  "aggregation": "weighted_sum",
  "process_completion": 0.4,
  "milestones": [
    {"id": "target_grasped", "weight": 0.2, "achieved": true, "first_step": 18},
    {"id": "target_moved", "weight": 0.2, "achieved": true, "first_step": 42}
  ],
  "trace": [
    {"step": 18, "completion": 0.2, "new_milestones": ["target_grasped"]},
    {"step": 42, "completion": 0.4, "new_milestones": ["target_moved"]}
  ]
}
```

`process_completion` is the final score. `milestones` records each milestone and its first achievement step. `first_step` is counted from the policy's first step and is `null` when not achieved. `trace` records each change caused by newly achieved milestones. Progress fields are written only to evaluator logs and are not exposed in policy-visible `info`.

## 8. Aggregate Metrics

`summary.json` and `by_task` provide `mean_process_completion`, `median_process_completion`, `completion_at_failure`, and `milestone_attainment_rates`.

Only `scorable: true` episodes are included. Infrastructure errors and unscorable tasks are excluded. Failures without a valid process record, such as policy-load failures, are excluded from `completion_at_failure`. If no qualifying samples exist, the corresponding value is `null` and milestone attainment rates are `{}`.

## 9. Version and Implementation

- Process-rule version: `gitbench_progress_v1`.
- Rule definitions and validation: [gitbench/progress.py](gitbench/progress.py).
- Environment state tracking: [gitbench/env.py](gitbench/env.py).
- Episode configuration normalization: [gitbench/episodes.py](gitbench/episodes.py).
- Evaluation aggregation: [scripts/eval_policy.py](scripts/eval_policy.py).
