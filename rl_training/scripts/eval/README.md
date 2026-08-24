# Evaluation Scripts

The release-supported benchmark path is `vbvr_pro/`. It keeps conversion,
generation, media preparation, rule scoring, and provenance in one auditable
workflow. The evaluator itself remains an external dependency.

## Directory Layout

| Directory | Purpose |
| --- | --- |
| `vbvr_pro/` | Stable VBVR-Pro run, sampler-sweep, summary, and optional VLM-judge launchers |
| `maze/` | Synthetic maze evaluation utilities |
| `lmms/` | Optional lmms-eval/FastVideo integration |

## One VBVR-Pro Cell

Inspect a DCP checkpoint with UniPC before starting expensive work:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/unipc \
  --sampler unipc \
  --dry-run
```

The launcher requires a separately obtained evaluator. It verifies its exact
Git revision when available, a complete source-contract digest, runtime
dependencies, OCR assets, all media, and every stage's provenance. There is no
vendored evaluator fallback.

## Sampler Sweep

Use arguments rather than adding checkpoint- or sampler-specific scripts:

```fish
fish scripts/eval/vbvr_pro/sweep.fish \
  --output-base storage/eval_out/<model> \
  --samplers unipc,euler,cps:0.3,cps:0.7 \
  -- \
  --model storage/models/<model> \
  --steps 30 \
  --guidance-scale 1.0
```

The sweep creates one output cell per sampler. It can deterministically shard
cells across machines with `--world-size` and `--rank`.

Summarize complete, provenance-bound cells with:

```fish
fish scripts/eval/vbvr_pro/summarize.fish \
  --root storage/eval_out/<model>
```

## Optional VLM Judge

The Qwen judge reads generated-video cells and writes a separate resumable
result root:

```fish
fish scripts/eval/vbvr_pro/vlm_judge.fish score \
  --input-root storage/eval_out/<model> \
  --output-root storage/eval_out/<model>-vlm-judge
```

It does not replace rule scoring and must not share output files with it. See
[`docs/vlm_judge_reward.md`](../../docs/vlm_judge_reward.md).

Before rule evaluation, verify the shared scorer runtime:

```bash
.venv/bin/python -m src.eval.vbvr_runtime
```

See [`vbvr_pro/README.md`](vbvr_pro/README.md) for launcher examples and
[`docs/vbvr_pro_eval.md`](../../docs/vbvr_pro_eval.md) for artifacts, stages,
resume behavior, and publication checks.
