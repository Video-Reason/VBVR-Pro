# VBVR-Pro Evaluation Launchers

This directory exposes a small, release-supported interface for VBVR-Pro
evaluation. Model paths, checkpoints, samplers, and output namespaces are
arguments; they are not encoded in experiment-specific script names.

## Files

| File | Purpose |
| --- | --- |
| `run.fish` | Evaluate one model with UniPC, Euler, or Flow-CPS |
| `sweep.fish` | Expand one model into several sampler cells, optionally sharded across machines |
| `reproduce.fish` | Reproduce the pinned 12-cell matrix for the published Rule-RL and Qwen-Judge-RL models |
| `summarize.fish` | Verify and summarize complete rule-score cells |
| `vlm_judge.fish` | Optionally judge generated-video cells with Qwen |
| `lib/rule_pipeline.fish` | Internal conversion, generation, preparation, scoring, and provenance implementation |

Do not invoke or copy `lib/rule_pipeline.fish` directly. Add stable options to
`run.fish` when the public contract needs to grow. A new checkpoint, sampler
coefficient, resolution, or output location does not need a new wrapper.

## Published Hugging Face Reproduction

The release reproduction launcher pins the
[Rule-RL](https://huggingface.co/pufanyi/VBVR-Pro-Wan2.2-TI2V-5B-Rule-RL) and
[Qwen-Judge-RL](https://huggingface.co/pufanyi/VBVR-Pro-Wan2.2-TI2V-5B-Qwen-Judge-RL)
model commits, the reviewed custom pipeline digest, all six paper samplers, and
the 512×512×81, 30-step, CFG 1.0, 16 FPS, seed-zero generation contract. It
materializes each snapshot beneath `storage/models/hf-releases`, validates it,
and then reuses the same generation, preparation, EvalKit scoring, provenance,
and summary stages as `run.fish`:

```fish
fish scripts/eval/vbvr_pro/reproduce.fish \
  --output-base storage/eval_out/published-hf \
  -- \
  --gt-base storage/datasets/vbvr-pro-eval-500 \
  --evalkit-dir storage/evalkits/<compatible-checkout> \
  --easyocr-model-dir storage/evalkits/easyocr-shared/model \
  --num-gpus 8
```

Inspect all 12 cells without downloading or evaluating:

```fish
fish scripts/eval/vbvr_pro/reproduce.fish \
  --output-base storage/eval_out/published-hf \
  --dry-run
```

| Model | CPS 0.1 | CPS 0.3 | CPS 0.7 | CPS 0.9 | Euler | UniPC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-RL | 0.509 | 0.526 | 0.548 | 0.539 | 0.522 | 0.522 |
| Qwen-Judge-RL | 0.482 | 0.493 | 0.508 | 0.509 | 0.488 | 0.497 |

Use `--models rule` or `--models qwen` for one release. Multi-machine runs use
the same `--world-size`/`--rank` cell assignment as `sweep.fish`; after all
ranks finish, run the command once with `--summarize-only`.

## One Rule-Evaluation Cell

Evaluate a DCP checkpoint with UniPC:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/unipc \
  --sampler unipc
```

Evaluate an existing Diffusers model with Euler:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --model storage/models/<model> \
  --output-root storage/eval_out/<model>/euler \
  --sampler euler
```

Flow-CPS requires an explicit coefficient:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --model storage/models/<model> \
  --output-root storage/eval_out/<model>/cps-noise-0.7 \
  --sampler cps \
  --cps-noise 0.7
```

Add `--dry-run` to resolve the model, sampler, evaluator, media contract, and
output namespace without reading model or dataset artifacts. Run
`fish scripts/eval/vbvr_pro/run.fish --help` for every supported option.

## Sampler Sweep

`sweep.fish` owns the sampler and output-root arguments. Place all options for
`run.fish` after `--`:

```fish
fish scripts/eval/vbvr_pro/sweep.fish \
  --output-base storage/eval_out/<model> \
  --samplers unipc,euler,cps:0.3,cps:0.7 \
  -- \
  --model storage/models/<model> \
  --steps 30 \
  --guidance-scale 1.0
```

Cells run sequentially on each machine. To distribute them, pass identical
arguments everywhere and set `--world-size` and `--rank`, or provide the
equivalent `WORLD_SIZE` and `RANK` environment variables. Assignment is a
deterministic round robin over the declared sampler order. Inspect it first
with `--assignment-only`.

For a checkpoint sweep, invoke the same generic command once per checkpoint
with a distinct conversion path and output base. Conversion output can be
reused safely by all sampler cells because the pipeline locks and fingerprints
it.

## Summaries and Optional VLM Judging

Summarize one run or a parent containing sampler cells:

```fish
fish scripts/eval/vbvr_pro/summarize.fish \
  --root storage/eval_out/<model> \
  --expected-samples 500
```

The command verifies score provenance and refuses to mix evaluator source
fingerprints. It writes an Excel workbook, JSON summary, and `final_scores.txt`
under `<root>/reports` unless `--output-dir` is supplied.

The optional Qwen judge consumes generated-video cells without modifying rule
scores:

```fish
fish scripts/eval/vbvr_pro/vlm_judge.fish score \
  --input-root storage/eval_out/<model> \
  --output-root storage/eval_out/<model>-vlm-judge
```

See [`docs/vbvr_pro_eval.md`](../../../docs/vbvr_pro_eval.md) for artifact
requirements, provenance semantics, resume rules, and publication checks.
