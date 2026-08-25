# Evaluation

The supported release workflow evaluates Wan2.2 checkpoints on an exact
VBVR-Pro manifest. It converts the training checkpoint when necessary,
generates videos, validates every expected path, prepares media for the rule
contract, scores through an external pinned evaluator, and records provenance
for every stage.

Use [`scripts/eval/vbvr_pro/run.fish`](../scripts/eval/vbvr_pro/run.fish) for
one complete rule-based evaluation cell and
[`sweep.fish`](../scripts/eval/vbvr_pro/sweep.fish) for sampler comparisons.
Lower-level commands are documented here for inspection and debugging, not as
a substitute for provenance checks.

## Inputs

A complete run needs:

- a DCP checkpoint plus its base Diffusers model, or a preconverted Diffusers
  model with immutable conversion/import metadata;
- a flattened VBVR-Pro evaluation tree;
- the exact split manifest used to select and order samples;
- a separately obtained compatible EvalKit checkout and source fingerprint;
- EasyOCR model assets for OCR-dependent tasks;
- the locked project environment and FFmpeg/ffprobe.

Model weights, the evaluation set, evaluator source, OCR weights, and generated
outputs are not bundled with this repository.

## End-to-End Pipeline

```text
DCP checkpoint + base model
            |
            v
  converted Diffusers model
            |
split manifest + GT tree ---> eval_samples.json
            |                       |
            +-----------------------+
                                    v
                           generated MP4 tree
                                    |
                                    v
                     frame-preserving preparation
                                    |
                                    v
                     pinned external rule scorer
                                    |
                                    v
                    result JSON + provenance files
```

Each stage validates its complete inputs and outputs. Existing files are
resumed only when they match the current provenance contract.

## Inspect a Run Without Loading Weights

Pass the artifact paths and `--dry-run`:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/unipc \
  --gt-base storage/datasets/vbvr-pro-eval-500 \
  --evalkit-dir storage/evalkits/<compatible-checkout> \
  --sampler unipc \
  --dry-run
```

The dry run prints model, output, sampler, and evaluator selections. For a
real run, the configured evaluator revision and digest must match the checkout.
See [External EvalKit](external_evalkit.md).

## Build the Exact Evaluation JSON

The builder walks:

```text
<gt-base>/
  In-Domain_50/<task>/<00000>/
  Out-of-Domain_50/<task>/<00000>/
```

and emits output names that already match EvalKit's domain/task/sample layout:

```bash
.venv/bin/python -m src.eval.build_vbvr_eval_json \
  --gt_base storage/datasets/vbvr-pro-eval-500 \
  --split_manifest storage/datasets/vbvr-pro-eval-500/split_manifest.json \
  --output storage/eval_out/<run>/eval_samples.json \
  --layout domain \
  --expected_samples 500
```

Manifest validation compares every flattened sample's `metadata.json` task ID
with the selected bench entry. Missing, extra, reordered, or mislabeled
samples fail before generation.

For duration-matched generation, add `--generation_fps` and
`--temporal_alignment`. The builder derives a per-sample `num_frames` value
that follows Wan's `alignment * k + 1` temporal rule.

## Generate Videos

For an existing Diffusers model:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=8 \
  -m src.cli.eval_i2v \
  --eval_json storage/eval_out/<run>/eval_samples.json \
  --model_path storage/models/converted/<model> \
  --output_dir storage/eval_out/<run>/generated \
  --height 256 \
  --width 256 \
  --num_frames 161 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --fps 16 \
  --seed 0
```

Each rank loads a pipeline and processes a disjoint deterministic slice. A
video is written through a temporary path and atomically promoted only after
encoding. Existing outputs are reused only when frame count, dimensions, and
FPS validate.

Generation entrypoints are:

| Mode | Module | Extra option |
| --- | --- | --- |
| UniPC ODE | `src.cli.eval_i2v` | none |
| Euler ODE | `src.cli.eval_i2v_euler` | none |
| Flow-CPS | `src.cli.eval_i2v_cps` | `--noise_level` |
| Reviewed Hugging Face pipeline | `src.cli.eval_i2v_hf_pipeline` | `--sampler`, optional `--cps_eta`, and required pipeline digest |

Keep model, checkpoint, EMA choice, sampler, sigma grid, inference steps, CFG,
seed, resolution, frame count, and FPS fixed when comparing checkpoints.

Validate an existing generation tree without loading a model:

```bash
.venv/bin/python -m src.cli.eval_i2v \
  --eval_json storage/eval_out/<run>/eval_samples.json \
  --output_dir storage/eval_out/<run>/generated \
  --height 256 --width 256 --num_frames 161 --fps 16 \
  --validate_only
```

## Load a DCP Checkpoint

`src.cli.eval_i2v` can load DCP directly with `--checkpoint`, but the public
VBVR-Pro launcher first converts to a standalone Diffusers directory. That
conversion is easier to fingerprint, reuse across samplers, and validate
before expensive generation.

```bash
.venv/bin/python -m src.cli.convert_dcp_to_diffusers \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --base_model storage/models/Wan2.2-TI2V-5B-Diffusers \
  --output storage/models/converted/<run>-checkpoint-100 \
  --merge_lora
```

Use `--use_ema` only when the intended evaluation contract calls for EMA.
Conversion provenance must record checkpoint and base-model fingerprints,
dtype, EMA choice, LoRA merge choice, and complete output-tree fingerprint.
See [Checkpoints](checkpoints.md).

## Prepare Media for Rule Scoring

Generated videos are not passed directly to the evaluator. Preparation:

- preserves every source frame;
- scales down without cropping;
- pads to the target canvas;
- removes audio and metadata;
- increases FPS only when necessary to fit the maximum duration;
- validates dimensions, frame count, FPS, and duration after encoding.

```bash
.venv/bin/python -m src.cli.prepare_vbvr_eval_videos \
  --input-dir storage/eval_out/<run>/generated \
  --output-dir storage/eval_out/<run>/prepared \
  --width 1024 \
  --height 1024 \
  --max-duration 5 \
  --workers 8 \
  --expected-videos 500 \
  --crf 12
```

This operation does not temporally subsample the rollout. If the source has too
many frames for the duration limit, it raises the encoded frame rate.

## Rule-Based Scoring

Run the scorer on CPU with CUDA hidden from EasyOCR workers:

```bash
CUDA_VISIBLE_DEVICES= \
EASYOCR_MODULE_PATH=storage/evalkits/easyocr-shared \
.venv/bin/python -m src.eval.vbvr_run_evaluation_parallel \
  --model_path storage/eval_out/<run>/prepared \
  --gt_base storage/datasets/vbvr-pro-eval-500 \
  --output_dir storage/eval_out/<run>/scores \
  --evalkit_dir storage/evalkits/<compatible-checkout> \
  --expected_evalkit_source_sha256 <64-hex-digest> \
  --expected_videos 500 \
  --device cpu \
  --num_workers 8 \
  --threads_per_worker 8
```

The adapter validates the pinned dependency runtime, fingerprints the complete
evaluator contract, checks the discovered video count, and writes per-sample
errors rather than silently dropping samples. A result is complete only when
it contains the expected sample count and no scorer errors.

## Result Contract

The score JSON contains:

- model/prepared-tree identity;
- evaluator source and scorer runtime fingerprints;
- one record per sample with task, domain, score, and optional error;
- aggregate means for `In_Domain`, `Out_of_Domain`, and `overall`.

The end-to-end launcher additionally writes:

```text
<output-root>/
  eval_samples.json
  generation-provenance.json
  preparation-provenance.json
  score-provenance.json
  generated_*/
  eval_*/
  scores/*_vbvr_results.json
```

Do not report an aggregate without retaining its score JSON and provenance
manifests.

## VLM Judge Evaluation

`src.cli.eval_vbvr_vlm_outputs` scores existing generated-video matrices with
the same task-specific prompt contract used by `vbvr_vlm` training reward. It
uses an independent append-only result root, supports deterministic
multi-machine assignment, and audits completeness before aggregation.

The VLM path is optional and does not replace rule scoring. Set up the judge
service and use the command recipes in
[Qwen VLM Reward](vlm_judge_reward.md).

## Reproducible Comparison Checklist

Before comparing two cells, confirm equality of:

- split manifest and expected sample count;
- base model, checkpoint loading, EMA, and LoRA merge semantics;
- sampler implementation, inference steps, sigma schedule, CFG, and seed;
- generated resolution, frame count or duration policy, and FPS;
- preparation canvas, duration limit, CRF, and frame-preserving behavior;
- evaluator source digest, scorer runtime digest, and OCR assets.

If any item changes, label the cell as a different evaluation contract rather
than attributing the difference only to the checkpoint.

## Completion Criteria

A release-quality evaluation is complete only when:

1. conversion or preconverted-model provenance validates;
2. the eval JSON matches the exact split manifest;
3. the generated tree has exactly the expected valid MP4 paths;
4. the prepared tree preserves every frame and passes media validation;
5. the scorer source and runtime fingerprints match the intended contract;
6. the score JSON has the expected sample count and zero sample errors;
7. every provenance manifest is in the `complete` state;
8. all processes exit successfully.

See [VBVR-Pro Evaluation](vbvr_pro_eval.md) for the full launcher environment,
resume behavior, sampler variants, and checkpoint sweeps.
