# VBVR-Pro Rule Evaluation

This is the detailed operator reference for the manifest-locked, rule-based
VBVR-Pro evaluation pipeline. The stable one-cell public launcher is:

```text
scripts/eval/vbvr_pro/run.fish
```

It is designed to fail closed when model, data, media, evaluator, or runtime
provenance changes. Use [Evaluation](evaluation.md) for the conceptual overview
and [External EvalKit](external_evalkit.md) for evaluator installation.

## Release Scope

The repository includes:

- DCP conversion and Diffusers validation;
- exact-manifest eval JSON construction;
- ODE and Flow-CPS generation entrypoints;
- frame-preserving video preparation;
- a parallel adapter for external EvalKit;
- stage provenance, completion checks, a sampler sweep, and generic result
  summaries;
- immutable Hugging Face snapshot materialization and a pinned reproduction
  matrix for the published Rule-RL and Qwen-Judge-RL checkpoints.

It does not include model weights, VBVR-Pro evaluation data, evaluator source,
EasyOCR weights, or generated results.

## Required Artifact Layout

A typical local setup is:

```text
storage/
  models/
    Wan2.2-TI2V-5B-Diffusers/
  checkpoints/
    <training-run>/checkpoint-100/
  datasets/
    vbvr-pro-eval-500/
      split_manifest.json
      In-Domain_50/<task>/<00000>/
      Out-of-Domain_50/<task>/<00000>/
  evalkits/
    <compatible-checkout>/
    easyocr-shared/model/
      craft_mlt_25k.pth
      english_g2.pth
  eval_out/
```

Each flattened sample directory must contain at least:

```text
first_frame.png
prompt.txt
ground_truth.mp4
metadata.json
```

The metadata task ID must match the selected entry in `split_manifest.json`.

## Evaluator Contract

The launcher defaults encode the recorded compatible `main_v2` contract:

```text
revision: e140038f2aee76ca518f464755fa8bc19b783ba5
source SHA-256: 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
```

These values do not imply that the corresponding revision is on the public
upstream default branch. Supply a checkout that actually matches both values,
or deliberately override both and create a new output namespace. Never keep an
old result label while changing evaluator source.

The Python environment is also part of the scorer contract. Before a run:

```bash
.venv/bin/python -m src.eval.vbvr_runtime
```

The launcher repeats this check and records the full runtime report.

## Dry Run

Inspect the resolved top-level selections without cloning evaluator source,
loading weights, or touching outputs:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/unipc \
  --sampler unipc \
  --dry-run
```

Always dry-run after changing model, sampler, media, evaluator, or output
arguments. A dry run does not prove that paths exist or fingerprints match;
those checks occur in the real pipeline.

## Reproduce the Published Hugging Face Matrix

`scripts/eval/vbvr_pro/reproduce.fish` is the single parameterized reproduction
entrypoint for the published
[Rule-RL](https://huggingface.co/pufanyi/VBVR-Pro-Wan2.2-TI2V-5B-Rule-RL) and
[Qwen-Judge-RL](https://huggingface.co/pufanyi/VBVR-Pro-Wan2.2-TI2V-5B-Qwen-Judge-RL)
TI2V-5B releases. It pins:

```text
Rule-RL model revision:       003373efcbc356e263f4c8d10b3dbb8f5cd7c6d0
Qwen-Judge-RL model revision: 1282a14cf5379f97ff77326373285533a9e2387d
pipeline.py SHA-256:          968acf1b214bce097f4d034bf26923dbf496ac319c1adb6560c16089f2ab0e50
samplers:                     cps:0.1,cps:0.3,cps:0.7,cps:0.9,euler,unipc
media:                        512x512x81 at 16 FPS
sampling:                     30 steps, CFG 1.0, base seed 0
samples:                      exact 500-item split manifest
```

Run both releases:

```fish
fish scripts/eval/vbvr_pro/reproduce.fish \
  --output-base storage/eval_out/published-hf \
  -- \
  --gt-base storage/datasets/vbvr-pro-eval-500 \
  --evalkit-dir storage/evalkits/<compatible-checkout> \
  --easyocr-model-dir storage/evalkits/easyocr-shared/model \
  --num-gpus 8
```

The launcher downloads only immutable revisions, verifies the reviewed custom
pipeline before loading weights, writes `conversion_metadata.json`, evaluates
six sampler cells per model, and summarizes each complete matrix. The benchmark
data, compatible EvalKit checkout, and EasyOCR weights remain external
requirements. Use `--dry-run` to inspect all 12 cells without downloading.

For multi-machine generation, pass the same arguments on every machine with
`--world-size N --rank R`. Once all ranks finish, invoke the launcher once with
`--summarize-only`. Exact output bytes can vary with GPU and runtime kernels;
retain every provenance file when comparing reproduced aggregates with the
three-decimal paper values.

## Public CLI Contract

### Model and data

| Option | Meaning |
| --- | --- |
| `--checkpoint PATH` | DCP checkpoint; mutually exclusive with `--model` |
| `--model PATH` | Preconverted Diffusers model; mutually exclusive with `--checkpoint` |
| `--base-model PATH` | Base Diffusers model used to interpret a DCP checkpoint |
| `--converted-model PATH` | Stable conversion output, required with `--checkpoint` |
| `--conversion-provenance PATH` | Immutable import record for a preconverted model |
| `--gt-base PATH` | Flattened VBVR-Pro evaluation tree |
| `--split-manifest PATH` | Exact bench selection; default `<gt-base>/split_manifest.json` |
| `--expected-videos N` | Exact required sample and output count |

### Evaluator and OCR

| Option | Meaning |
| --- | --- |
| `--evalkit-dir PATH` | External checkout path |
| `--evalkit-repo URL` | Optional Git source used only when the checkout is absent |
| `--evalkit-revision REV` | Exact Git revision required by the run |
| `--evalkit-source-sha256 HASH` | Complete source-contract fingerprint |
| `--easyocr-model-dir PATH` | Directory containing pre-populated OCR weights |
| `--easyocr-root PATH` | Runtime EasyOCR root exposed to scorer workers |
| `--score-workers N` | CPU scorer process count |
| `--score-threads N` | Native threads allowed inside each scorer process |

The launcher creates `EVALKIT_DIR/easyocr_models` as a symlink to the staged
OCR model directory. It refuses to replace a real directory at that path.

### Generation

| Option | Meaning |
| --- | --- |
| `--generation-backend NAME` | `native` or the reviewed `hf-pipeline` backend |
| `--hf-pipeline-sha256 HASH` | Required custom-pipeline digest for `hf-pipeline` |
| `--sampler NAME` | `unipc`, `euler`, or `cps` |
| `--cps-noise FLOAT` | Required Flow-CPS coefficient in `[0, 1]` |
| `--num-gpus N` | Local data-parallel generation process count |
| `--cuda-devices LIST` | Visible CUDA device list for generation/conversion |
| `--height N`, `--width N` | Generated spatial size |
| `--num-frames N`, `--fps N` | Fixed generated media contract |
| `--steps N` | Denoising step count |
| `--guidance-scale FLOAT` | Classifier-free guidance scale |
| `--seed N` | Deterministic base seed |
| `--match-gt-duration` | Derive each ODE sample's frame count from GT duration |
| `--temporal-alignment N` | Alignment used for duration-matched frame lengths |

### Preparation and outputs

| Option | Meaning |
| --- | --- |
| `--output-root PATH` | Unique root for the complete evaluation cell; required |
| `--prepared-height N`, `--prepared-width N` | Rule scorer canvas |
| `--max-duration SECONDS` | Maximum prepared-video duration |
| `--prep-crf N` | H.264 preparation quality setting |
| `--prep-workers N` | Parallel FFmpeg preparation workers |

Artifact defaults live beneath ignored `storage/`. An output root must identify
the model, sampler, generation shape, manifest, and scorer contract well enough
to avoid accidental reuse. Run `run.fish --help` for defaults and the complete
option list.

## Run a Complete Cell

Example ODE evaluation:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/unipc \
  --gt-base storage/datasets/vbvr-pro-eval-500 \
  --evalkit-dir storage/evalkits/<compatible-checkout> \
  --evalkit-revision <revision> \
  --evalkit-source-sha256 <64-hex-digest> \
  --sampler unipc \
  --num-gpus 8
```

Flow-CPS changes the sampler and therefore requires a different output root:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/cps-noise-0.7 \
  --sampler cps \
  --cps-noise 0.7
```

Do not compare an ODE cell and a CPS cell unless inference steps, sigma grid,
CFG, seed, resolution, frame policy, preparation, and scoring contracts are
also controlled and reported.

## Stage 1: Evaluator and Runtime Preflight

Before touching model outputs, the launcher:

1. obtains a missing evaluator checkout only when `EVALKIT_REPO` is set;
2. serializes concurrent installation with a directory lock;
3. checks out the exact revision in detached state;
4. computes and compares the complete source fingerprint;
5. stages the two required EasyOCR weights and its compatibility symlink;
6. validates the pinned Python/scientific-media runtime.

If an evaluator directory exists but is incomplete, the launcher stops instead
of overwriting it. If the directory is a Git checkout, its `HEAD` must match
`EVALKIT_REV` as well as the content fingerprint.

## Stage 2: Checkpoint Conversion

For DCP input, one process acquires a conversion lock and runs:

```text
src.cli.convert_dcp_to_diffusers
```

The public launcher converts raw, non-EMA weights, merges LoRA when present,
uses bf16 safe serialization, and validates the resulting Diffusers tree. It
fingerprints the checkpoint, base model, converter source, conversion options,
and complete output tree.

Concurrent jobs targeting the same converted model wait for the lock and then
reuse a complete validated result. An incomplete directory or incompatible
provenance fails instead of being silently repaired from different inputs.

For conversion without evaluation preflight, use
`src.cli.convert_dcp_to_diffusers` directly as documented in
[Checkpoints](checkpoints.md).

### Preconverted models

Pass `--model` only for a complete Diffusers directory. It must carry its own
immutable `conversion_metadata.json`, or pass `--conversion-provenance` with an
equivalent import record. The launcher validates structure, fingerprints the
tree, and checks that it remains stable during the validation interval.

`src.cli.materialize_hf_diffusers_model` creates this local tree and import
record from a full Hugging Face commit SHA. It also validates every referenced
component and safetensors header and refuses a different `pipeline.py` digest.

## Stage 3: Manifest Construction

`src.eval.build_vbvr_eval_json` verifies each flattened sample against the
split manifest and writes names in the evaluator's domain layout:

```text
In-Domain_50/<task>/<index>
Out-of-Domain_50/<task>/<index>
```

The item count must equal `--expected-videos`. The generation JSON contains
absolute first-frame paths and prompts, avoiding dependence on a later working
directory.

With `--match-gt-duration`, the builder probes each ground-truth video and
selects the closest legal `alignment * k + 1` generation length at the selected
FPS. This mode currently requires ODE generation.

## Stage 4: Video Generation

The launcher first validates any existing generated tree without loading
weights. If both media and provenance are complete, it skips generation.
Otherwise it resumes missing/invalid videos through the selected module:

- `src.cli.eval_i2v` for UniPC ODE;
- `src.cli.eval_i2v_euler` for Euler ODE;
- `src.cli.eval_i2v_cps` for Flow-CPS;
- `src.cli.eval_i2v_hf_pipeline` for a reviewed custom pipeline materialized
  from Hugging Face; sampler choice remains a per-cell provenance field.

Generation is local multi-GPU data parallel. Every expected path is derived
from the eval JSON; extra, missing, duplicate, corrupt, wrong-size,
wrong-frame-count, or wrong-FPS outputs fail exact-set validation.

Use `--force-regenerate` only when you intentionally want to replace a tree
whose provenance differs. Prefer a fresh output root for a changed contract.

## Stage 5: Frame-Preserving Preparation

`src.cli.prepare_vbvr_eval_videos` mirrors the generated tree. For each MP4 it:

1. probes exact dimensions, frames, FPS, and duration;
2. scales to fit the configured canvas without cropping;
3. pads the remaining area;
4. retains every frame;
5. raises FPS when needed to satisfy `--max-duration`;
6. writes H.264 through a temporary file and validates it before promotion.

The prepared output must contain exactly the expected video count. Preparation input,
parameters, implementation source, and full output tree are fingerprinted.

Use `--force-reprepare` only for an intentional rewrite. Regenerating videos
automatically invalidates downstream preparation.

## Stage 6: Rule Scoring

The launcher hides CUDA and sets native thread limits before starting CPU
scorer workers. The adapter:

- validates the evaluator source and dependency runtime again;
- changes worker directories only after all relevant paths are resolved;
- requires EvalKit to discover exactly the expected number of videos;
- captures one result per sample;
- writes domain and overall aggregates;
- preserves sample errors instead of dropping them.

The result file name is derived from the prepared directory. A complete score
file is reused only when its provenance and exact sample set still validate.
Otherwise the launcher deletes the stale or partial score state and starts
scoring cleanly. It promotes score provenance only when sample count is exact
and no record contains an error.

## Output and Provenance Layout

```text
<output-root>/
  eval_samples.json
  generation-provenance.json
  preparation-provenance.json
  score-provenance.json
  generated_<shape>/
    In-Domain_50/<task>/<index>.mp4
    Out-of-Domain_50/<task>/<index>.mp4
  prepared_<shape-and-duration>/
    ... mirrored MP4 tree ...
  scores/
    <prepared-name>_vbvr_results.json
```

Conversion provenance normally sits beside `CONVERTED_MODEL`, allowing
multiple evaluation cells to reuse one immutable conversion safely.

Each stage manifest records a state such as `in_progress_resume`,
`in_progress_rewrite`, or `complete`; scalar parameters; input files/trees;
implementation files; and output-tree fingerprints. A `complete` manifest is
valid only while all fingerprints still match.

## Resume Rules

Safe automatic resume applies to:

- a complete matching converted model;
- individual generated videos that pass exact media validation under matching
  generation inputs;
- individual prepared videos under matching preparation inputs;
- a complete, error-free score result with matching scorer provenance.

A complete matching score is reused. Any incomplete or mismatched score starts
from a clean result because merging arbitrary partial evaluator outputs can hide
contract changes or missing errors.

If an output tree exists without compatible provenance, the launcher stops and
asks for an explicit force option or a fresh namespace. A fresh namespace is
the recommended response to any semantic change.

## Checkpoint and Sampler Sweeps

Use one parameterized sweep instead of creating checkpoint-by-sampler wrapper
scripts:

```fish
fish scripts/eval/vbvr_pro/sweep.fish \
  --output-base storage/eval_out/<run>/checkpoint-100 \
  --samplers unipc,euler,cps:0.3,cps:0.7 \
  -- \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --steps 30 \
  --guidance-scale 1.0
```

The sweep assigns a separate output root to every sampler and reuses the same
locked, provenance-bound conversion. It runs cells sequentially on one machine.
For several machines, pass identical arguments with `--world-size N --rank R`;
`--assignment-only` prints the deterministic round-robin assignment without
starting work.

Invoke the same sweep once per checkpoint with a distinct conversion path and
output base. Keep manifest, media, and scorer settings fixed when comparing
cells. Summarize only complete results that share one evaluator fingerprint:

```fish
fish scripts/eval/vbvr_pro/summarize.fish \
  --root storage/eval_out/<run>/checkpoint-100 \
  --expected-samples 500 \
  --expected-evalkit-source-sha256 <64-hex-digest>
```

The summary command discovers the result filename from score provenance and
writes `vbvr_pro_summary.xlsx`, `vbvr_pro_summary.json`, and `final_scores.txt`.

## Training Reward Parity

`vbvr_rule` training reward uses the same external evaluator source pin,
dependency runtime, frame-preserving preparation dimensions, duration limit,
and task-specific scorer interface. The relevant YAML fields are:

```yaml
vbvr_reward_evalkit_dir: storage/evalkits/<compatible-checkout>
vbvr_reward_evalkit_source_sha256: <64-hex-digest>
vbvr_reward_fps: 16
vbvr_reward_prepared_width: 1024
vbvr_reward_prepared_height: 1024
vbvr_reward_max_duration_seconds: 5.0
vbvr_reward_prepare_crf: 12
```

Parity still depends on generated frame count, FPS, task metadata, and
evaluator revision. Keep those aligned when using offline ground-truth checks
to validate the online reward.

## Failure Recovery

### Evaluator mismatch

Restore the intended checkout. If the evaluator change is deliberate, compute
a new digest, change both config and output namespace, and report it as a new
metric contract.

### Incomplete converted model

Do not write into it concurrently by hand. Remove or relocate only the exact
incomplete artifact after confirming no job owns the conversion lock, then
rerun conversion. Never replace a complete model behind matching provenance.

### Generated set validation failure

Read the reported paths and media mismatch. Rerunning normally repairs missing
or invalid files under matching provenance. Use `--force-regenerate` only when
the entire prior generation contract is intentionally superseded.

### Preparation failure

Verify FFmpeg/ffprobe availability and inspect the source MP4. The preparer
reports exact dimension, frame, FPS, or duration violations and does not
promote an invalid temporary output.

### Partial or errored score JSON

Inspect per-sample `error` fields and evaluator logs. Fix evaluator assets,
metadata paths, or task-specific dependencies, then rerun the score stage
through the launcher. Do not manually remove errored records and recompute the
mean.

### All scores are zero

Confirm EvalKit found the intended prepared paths and ground truth. Check task
support, metadata identity, OCR assets, worker exceptions, and the source/runtime
fingerprints before interpreting the result as model behavior.

## Completion Audit

Before publishing a number, verify:

- [ ] the repository revision is recorded;
- [ ] conversion provenance is complete and the model tree is stable;
- [ ] the eval JSON matches the intended split manifest and exact sample count;
- [ ] generated and prepared trees have the exact expected path sets;
- [ ] every generated/prepared MP4 passes media validation;
- [ ] evaluator revision and source digest match the declared contract;
- [ ] scorer runtime digest and OCR assets are recorded;
- [ ] result JSON contains all samples and zero error records;
- [ ] In-Domain, Out-of-Domain, and overall aggregates come from that file;
- [ ] generation, preparation, and score provenance are complete;
- [ ] all launcher processes exited successfully.

Retain the launcher arguments, score JSON, and provenance manifests with any
reported aggregate. Generated videos may be stored separately, but their
complete tree fingerprint must remain auditable.
