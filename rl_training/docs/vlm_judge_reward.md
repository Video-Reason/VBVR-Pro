# Qwen VLM Judge Reward

`vbvr_vlm` scores generated VBVR-Pro rollouts with a separately hosted
OpenAI-compatible multimodal service. The training process decodes a rollout,
encodes it as an in-memory H.264 MP4, submits the input frame plus video, and
receives a normalized task-specific score.

The judge is optional. Its model weights are external, while its dependency
stack lives in a separate uv-managed environment resolved by
`requirements/vllm.lock`.

## Supported Contract

The reference service uses:

| Component | Recorded value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B` |
| Model revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Served name | `qwen3.6-27b` |
| vLLM | `0.26.0` |
| API | OpenAI-compatible `/v1/chat/completions` |
| Prompt mode | `task_specific` |
| Prompt-source SHA-256 | `4d3159232590bd4b99266c9e82df445a3a54ada50a7af30051cf505057574202` |

Changing the model revision, serving runtime, prompt source, media sampling,
or parser creates a different reward/evaluation contract. Record those values
with every result.

## Request Semantics

The default `task_specific` request sends:

- the task's pinned evaluator-derived rubric;
- the input first frame;
- one in-memory MP4 containing the complete decoded generated rollout.

It does not send the ground-truth final frame. That omission is intentional and
part of the task-specific prompt contract.

The MP4 is encoded at `vlm_reward_video_fps`. vLLM's OpenCV media loader then
selects exactly `vlm_reward_video_num_frames` frames uniformly from the full
video. The Hugging Face processor receives `do_sample_frames: false`, so there
is no second temporal sampling pass.

Each of the 100 tasks has its own rubric fields and line-oriented output
schema. The client:

1. derives a per-task constrained-decoding regex;
2. validates every field, score, reason, and 100-point weight sum;
3. normalizes `total_score` to `[0, 1]`;
4. sends a repair request for an invalid response up to the configured retry
   count.

Regex can constrain syntax but cannot prove arithmetic across separate lines,
so semantic validation remains mandatory.

## Isolated vLLM Environment

Create the ignored service environment:

```fish
fish scripts/dev/setup_host_vllm.fish
```

The service uses `storage/host_vllm/.venv`, which explicitly excludes the
training stack. Its vLLM, Ray, NumPy, Torch, and CUDA-wheel contract is recorded
in `requirements/vllm.in` and the compiled `requirements/vllm.lock`; setup also
writes an ignored `storage/host_vllm/requirements.freeze.txt` inspection
snapshot. Regenerate the lock deliberately when updating this runtime:

```bash
uv pip compile --no-config requirements/vllm.in \
  --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu \
  --torch-backend cu126 \
  --generate-hashes \
  --output-file requirements/vllm.lock
```

## Download the Pinned Model

After creating the isolated environment:

```fish
fish scripts/download/qwen36_27b_hf_mirror.fish
```

The helper downloads the exact recorded revision into
`storage/models/Qwen3.6-27B` and asks the Hugging Face CLI to verify all remote
and LFS files. To use the standard Hugging Face endpoint instead, run the
isolated `hf` command directly:

```bash
storage/host_vllm/.venv/bin/hf download Qwen/Qwen3.6-27B \
  --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --local-dir storage/models/Qwen3.6-27B
```

Review the model license and access requirements. Keep weights under
`storage/`; do not commit or redistribute them through this repository.

## Start a Standalone Service

The reference launcher hosts one OpenAI-compatible endpoint:

```fish
fish scripts/serve/qwen36_27b_vllm.fish
```

Defaults include:

```text
endpoint: http://127.0.0.1:18080/v1
tensor parallel: 8
data parallel: 1
GPU memory utilization: 0.50
maximum model length: 32768
maximum concurrent sequences: 32
```

Override the topology to match available devices:

```bash
WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE=2 \
WAN_TRAINER_VLM_DATA_PARALLEL_SIZE=4 \
WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL=4 \
WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION=0.50 \
fish scripts/serve/qwen36_27b_vllm.fish
```

For the built-in `mp` backend, local data-parallel size must equal global
data-parallel size. Multi-machine vLLM deployments require an externally
managed Ray or vLLM head/headless setup; they are not created by this helper.

The launcher removes inherited proxies for loopback requests and places
Triton, Inductor, CUDA, and FlashInfer workspaces under node-local `/tmp` by
default. Those caches are runtime artifacts and can be overridden with the
corresponding `WAN_TRAINER_VLM_*` variables.

## Probe the Service

Wait for startup and exercise both vision input and the exact dynamic task
schema:

```bash
.venv/bin/python -m src.cli.probe_vlm_service \
  --base-url http://127.0.0.1:18080/v1 \
  --model qwen3.6-27b \
  --wait-seconds 900 \
  --multimodal-smoke \
  --task-prompt-smoke
```

A model-list response alone is insufficient: the multimodal smoke proves image
decoding, and the task-prompt smoke proves video encoding, per-task constrained
output, and semantic parsing.

## Configure Training Reward

Minimal YAML:

```yaml
grpo_reward_fn: vbvr_vlm
vlm_reward_base_url: http://127.0.0.1:18080/v1
vlm_reward_model: qwen3.6-27b
vlm_reward_api_key: EMPTY
vlm_reward_prompt_mode: task_specific
vlm_reward_video_fps: 16
vlm_reward_video_num_frames: 32
vlm_reward_include_gt_first_frame: true
vlm_reward_image_max_edge: 512
vlm_reward_concurrency: 2
vlm_reward_max_pending_jobs: 0
vlm_reward_validate_service: true
vlm_reward_fail_on_error: false
vlm_reward_error_score: 0.0
```

Environment variables override endpoint credentials without modifying the
config:

```bash
WAN_TRAINER_VLM_BASE_URL=http://127.0.0.1:18080/v1
WAN_TRAINER_VLM_MODEL=qwen3.6-27b
WAN_TRAINER_VLM_API_KEY=<token>
```

`vlm_reward_image_max_edge` is downscale-only. Frames at or below the bound
retain their native resolution. `vlm_reward_max_pending_jobs: 0` selects the
larger of decode batch size and twice the request concurrency.

Only tensor-parallel rank zero decodes and submits rewards; the result is
broadcast to its tensor-parallel peers. Other data-parallel ranks run their own
request pools.

## Strict and Fail-Open Modes

Use strict mode for a bounded validation:

```yaml
vlm_reward_fail_on_error: true
vlm_reward_validate_service: true
vlm_reward_log_first_n: 2
```

Production reference configs use fail-open behavior so one exhausted HTTP or
schema failure does not tear down every distributed rank:

```yaml
vlm_reward_fail_on_error: false
vlm_reward_error_score: 0.0
```

Fail-open is operational protection, not permission to ignore judge health.
Repeated fallback scores can flatten or bias group advantages. Monitor error
counts, retry reasons, response validation, latency, and reward distributions.

## Custom Prompt Mode

`vlm_reward_prompt_mode: custom` keeps the generic evaluator interface and
allows either an inline system prompt or a file:

```yaml
vlm_reward_prompt_mode: custom
vlm_reward_system_prompt_path: storage/prompts/custom_vlm_judge.txt
vlm_reward_use_structured_output: true
```

Custom mode is a separate reward contract. It may include a ground-truth final
reference and may use a fixed JSON schema. Do not compare it directly with the
task-specific 100-rubric score without labeling the distinction.

## Co-Hosted Training

The wrapper can start one node-local service, probe it, launch standard
multi-machine GRPO, and stop the complete service process group on exit:

```bash
MASTER_ADDR=<rank-zero-host> \
MASTER_PORT=29500 \
WORLD_SIZE=<machine-count> \
RANK=<machine-rank> \
WAN_TRAINER_VLM_DATA_PARALLEL_SIZE=4 \
WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE=2 \
fish scripts/train/grpo_vlm_eval_multinode.fish --nproc 8 -- \
  --config configs/train_rl_5b_vlm.yaml \
  --output_dir storage/checkpoints/<unique-topology-run> \
  --wandb_run_name <unique-topology-run>
```

The checked-in eight-GPU-per-machine VLM config uses four TP2 Qwen replicas per
machine. Other models or GPU counts may need a different DP/TP split. Give each
training topology its own checkpoint and W&B namespace. The service uses the
same visible GPUs as training with a configured memory utilization budget.
That value is a vLLM allocation target, not a hard partition of physical
memory. Run a bounded target-shape smoke before a long co-hosted job and leave
headroom for CUDA contexts, media decoding, transient activations, and
allocator fragmentation.

Set `WAN_TRAINER_VLM_START_SERVICE=0` to use an independently managed endpoint.
Then provide a reachable `WAN_TRAINER_VLM_BASE_URL` on every training machine
and secure it appropriately.

## Offline Judge for Existing Videos

Offline judging uses the same task-specific prompts and media contract as
training. It reads completed VBVR-Pro evaluation cells and writes an independent
append-only result root.

Single-machine example:

```bash
.venv/bin/python -m src.cli.eval_vbvr_vlm_outputs score \
  --input-root storage/eval_out/<rule-or-generation-matrix> \
  --output-root storage/eval_out/<matrix>-vlm-judge \
  --base-url http://127.0.0.1:18080/v1 \
  --model qwen3.6-27b \
  --world-size 1 \
  --rank 0 \
  --concurrency 16 \
  --expected-samples-per-cell 500
```

Audit assignments without issuing requests:

```bash
.venv/bin/python -m src.cli.eval_vbvr_vlm_outputs score \
  --input-root storage/eval_out/<matrix> \
  --output-root storage/eval_out/<matrix>-vlm-judge \
  --world-size 1 --rank 0 \
  --assignment-only
```

Strictly summarize already complete cells:

```bash
.venv/bin/python -m src.cli.eval_vbvr_vlm_outputs summarize \
  --input-root storage/eval_out/<matrix> \
  --output-root storage/eval_out/<matrix>-vlm-judge
```

The convenience wrapper can start and stop a local judge service automatically:

```fish
fish scripts/eval/vbvr_pro/vlm_judge.fish \
  score \
  --input-root storage/eval_out/<matrix> \
  --output-root storage/eval_out/<matrix>-vlm-judge
```

## Distributed Offline Assignment

For multiple evaluation machines, run the same command everywhere with
machine-count `WORLD_SIZE` and zero-based `RANK`. Complete cells are sorted and
assigned deterministically in round-robin order. Each machine owns complete
cells rather than arbitrary sample fragments, enabling independent resume.

Rank zero can wait for all cells and publish the aggregate. Every cell is
considered complete only when source fingerprints, judge contract, expected
sample count, append-only results, zero-error policy, and final aggregates
validate.

Use a separate output root when any judge setting changes. Never merge JSONL
from different model revisions, prompts, frame-sampling settings, or service
contracts.

## Capacity Tuning

The main tuning controls are:

- tensor parallelism, which reduces per-device model weight load;
- data parallelism, which creates independent serving replicas;
- `WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION`;
- `WAN_TRAINER_VLM_MAX_NUM_SEQS`;
- client-side `vlm_reward_concurrency` or offline `--concurrency`;
- `vlm_reward_max_pending_jobs`;
- source resolution and uniformly sampled video frame count.

Increase one dimension at a time while measuring memory, request throughput,
tail latency, retry rate, and training GPU headroom. More client concurrency
does not help after the service is saturated and can increase timeouts.

## Security

The default service binds to loopback and uses the placeholder key `EMPTY`.
For a remotely reachable endpoint:

- bind only to an intended interface;
- set a real API key;
- restrict network access;
- avoid logging request payloads because they contain input images and videos;
- do not place keys in YAML, Git, shell history, or W&B config;
- keep proxy bypass scoped to trusted service addresses.

## Troubleshooting

### Service starts but preflight fails

Read the vLLM service log. Verify the served name, model revision, multimodal
limits, video support, and reasoning parser. Run the simple multimodal smoke
before the task-prompt smoke to isolate media loading from schema parsing.

### CUDA out of memory during startup

Reduce GPU memory utilization or data-parallel replicas, increase tensor
parallelism where supported, reduce maximum sequences/model length, or host the
service on dedicated devices.

### Training times out waiting for rewards

Check service health and latency first. Then reduce per-rank concurrency or
pending jobs, increase request timeout only when requests are making progress,
and verify that every rank resolves the intended endpoint without a proxy.

### Schema repair loops

Confirm the exact prompt source and served model revision. Inspect the first
logged responses, task-specific weight sum, missing rubric fields, and reason
format. Do not weaken semantic validation merely to accept malformed outputs.

### Offline resume refuses an existing cell

The source tree or judge contract differs from the recorded manifest, or the
cell is incomplete. Use the assignment audit to identify the mismatch. Repair
the exact source or choose a new output root; do not hand-edit completion
metadata.

## Publication Record

For a VLM-reward or offline-judge result, retain:

- model repository and immutable revision;
- vLLM version and frozen service environment;
- prompt source digest and prompt mode;
- FPS, source frames, sampled frames, image bound, and JPEG quality;
- endpoint topology, request concurrency, timeout, retry, and error policy;
- raw per-sample outputs, validation errors, and normalized scores;
- source-video tree fingerprints and complete aggregate manifests.
