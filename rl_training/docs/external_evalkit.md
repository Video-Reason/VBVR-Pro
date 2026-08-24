# External VBVR EvalKit

VBVR-RL does not vendor VBVR-EvalKit. Rule-based training and offline scoring
load a separately obtained checkout, require its source fingerprint, and
record that fingerprint in provenance. This keeps third-party source and
licensing separate while making the metric definition explicit.

## Compatibility Notice

The public upstream project is
[`Video-Reason/VBVR-EvalKit`](https://github.com/Video-Reason/VBVR-EvalKit).
The recorded `main_v2` workflows in this repository were built against a
compatible fork/revision whose interface and annotations differ from the
upstream default branch. Do not assume that cloning upstream `main` reproduces
the recorded scores.

Use one of these approaches:

- obtain the exact compatible checkout used by the selected config or
  launcher; or
- intentionally validate a different evaluator, compute a new fingerprint,
  and write results to a new namespace that identifies the changed contract.

The evaluator's own license and dependencies apply independently of this
repository's MIT license.

## Expected Checkout Interface

The rule integration expects the checkout to contain:

```text
<evalkit>/
  run_evaluation.py
  requirements.txt
  annotations/
    ... task annotation files ...
  vbvr_bench/
    __init__.py
    evaluators/
      __init__.py
      *.py
```

The selected entrypoint must accept the arguments used by
[`src/eval/vbvr_run_evaluation_parallel.py`](../src/eval/vbvr_run_evaluation_parallel.py).
The `vbvr_bench.evaluators` module must expose `is_out_of_domain` for the
restructuring utility.

## Install Outside the Source Tree

Place the checkout under the ignored artifact tree rather than `third_party/`:

```bash
git clone <compatible-evalkit-repository> \
  storage/evalkits/vbvr-evalkit-compatible
git -C storage/evalkits/vbvr-evalkit-compatible checkout <revision>
```

The end-to-end VBVR-Pro launcher can install a missing checkout when both
`EVALKIT_REPO` and `EVALKIT_REV` are supplied. It serializes installation with
a lock, checks out the requested revision, and verifies the source digest
before scoring. Supplying an already validated `EVALKIT_DIR` is preferable for
offline or multi-machine runs.

Every scoring machine must see the same immutable contents at the configured
path. Do not modify an active checkout while reward workers are running.

## Compute the Source Fingerprint

The fingerprint includes the evaluator entrypoint, evaluator modules, task
annotations, and requirements used by the scoring contract:

```bash
.venv/bin/python -c '
import sys
from pathlib import Path
from src.eval.vbvr_run_evaluation_parallel import evalkit_source_sha256
print(evalkit_source_sha256(Path(sys.argv[1])))
' storage/evalkits/vbvr-evalkit-compatible
```

Record the resulting 64-character digest next to the checkout revision. A
digest mismatch is a hard error in both online reward and offline evaluation.
Recompute it after any source, annotation, or requirements change.

## Configure Online Rule Reward

Set both required fields in the RL YAML:

```yaml
grpo_reward_fn: vbvr_rule
vbvr_reward_evalkit_dir: storage/evalkits/vbvr-evalkit-compatible
vbvr_reward_evalkit_source_sha256: <computed-digest>
```

Common operational fields are:

```yaml
vbvr_reward_device: cpu
vbvr_reward_fps: 16
vbvr_reward_prepared_width: 1024
vbvr_reward_prepared_height: 1024
vbvr_reward_max_duration_seconds: 5.0
vbvr_reward_cpu_workers: 1
vbvr_reward_cpu_threads_per_worker: 1
vbvr_reward_use_process_pool: true
vbvr_reward_tmp_dir: storage/tmp/<run>/vbvr_rule
```

Use a small worker count per GPU rank; total scorer concurrency is the worker
count multiplied by the number of reward-producing ranks. Input and metadata
paths are resolved before spawned workers change their working directory.

Validate dependencies before loading weights:

```bash
.venv/bin/python -m src.cli.validate_grpo_runtime \
  --config configs/<rule-reward-config>.yaml
```

## Configure Offline Scoring

The parallel adapter requires the checkout and digest explicitly:

```bash
.venv/bin/python -m src.eval.vbvr_run_evaluation_parallel \
  --evalkit_dir storage/evalkits/vbvr-evalkit-compatible \
  --expected_evalkit_source_sha256 <computed-digest> \
  --model_path storage/eval_out/<run>/prepared \
  --gt_base storage/datasets/vbvr-pro-eval-500 \
  --output_dir storage/eval_out/<run>/scores \
  --num_workers 8 \
  --threads_per_worker 8
```

Use the full launcher in [VBVR-Pro Evaluation](vbvr_pro_eval.md) for normal
runs. It additionally locks the split manifest, generated media, preparation
parameters, model tree, and runtime fingerprint.

## EasyOCR Assets

Some tasks require EasyOCR. Keep its model files in an ignored shared
directory, for example:

```text
storage/evalkits/easyocr-shared/model/
```

Set `EASYOCR_SOURCE_MODELS` for offline launchers. For training, use
`vbvr_reward_easyocr_module_path` only when the compatible evaluator requires
an explicit module or asset path. All scorer workers must have read access.
Do not commit downloaded OCR weights.

## Runtime Contract

Evaluator source is only one part of the metric. The locked Python environment
also pins EasyOCR, OpenCV, NumPy, SciPy, scikit-image, and related media
packages. Check it with:

```bash
.venv/bin/python -m src.eval.vbvr_runtime --json
```

The report includes a behavioral OpenCV probe and a stable runtime digest.
The evaluation launcher stores this report in score provenance. If the check
fails, repair the environment and restart every scorer process; updating an
environment cannot replace modules already imported by running workers.

## Failure Modes

| Symptom | Action |
| --- | --- |
| Missing explicit evaluator path | Set `vbvr_reward_evalkit_dir` or `--evalkit_dir`; no repository fallback exists. |
| Source digest mismatch | Restore the intended revision or create a new scorer contract and output namespace. |
| `vbvr_bench` import fails | Verify checkout layout and revision; do not add it permanently to the repository. |
| OCR model download occurs during a job | Prestage the EasyOCR assets and point every worker at the same directory. |
| All rewards are zero | Inspect per-sample errors, task support, resolved GT paths, prepared media, and fail-open warnings. |
| Scores differ across environments | Compare evaluator digest, runtime digest, manifest, media-preparation provenance, and EasyOCR assets. |

## Release Rule

Never add an implicit evaluator fallback to repository source. Every rule
objective and reported rule score must identify both the evaluator source
digest and the locked runtime contract.
