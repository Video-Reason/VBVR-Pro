"""Checkpoint primitives for FSDP2 + DCP training.

Disk layout per checkpoint directory (`R` = shard rank within the DCP group):

    checkpoint-N/
      ├─ .metadata + __*_*.distcp           # DCP: train_state (model state +
      │                                     #      step counters + RNG) + ema
      ├─ optimizer_<name>_rank{R}.pt        # one .pt per shard rank, outside DCP
      ├─ dataloader_rank{R}.pt              # per shard rank
      └─ lora/                              # only when lora_rank > 0
          ├─ transformer/{adapter_model.safetensors, adapter_config.json}
          └─ transformer_2/{adapter_model.safetensors, adapter_config.json}

For expert-parallel runs the same layout sits inside `high/` (transformer) and
`low/` (transformer_2) subdirectories.

Resume requires the **same layout** (flat ↔ EP transitions raise). For
inference, ``load_dcp_into_pipeline`` handles both full-FT and LoRA
checkpoints — LoRA is auto-detected from the ``lora/<transformer>/``
sidecars and applied to the pipeline before weight load.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


def _is_fsdp_wrapped(model: torch.nn.Module) -> bool:
    """Detect FSDP2 (fully_shard) wrapping via the DTensor ._local_tensor attr."""
    return any(hasattr(p, "_local_tensor") for p in model.parameters())


# ---------------------------------------------------------------------------
# TrainState — model weights + step counters + RNG.  Optimizers are NOT here;
# they live in `optimizer_<name>_rank{R}.pt` files alongside the DCP shards.
# ---------------------------------------------------------------------------


MODEL_KEYS = ("text_encoder", "transformer", "transformer_2")


class TrainState(Stateful):
    """DCP `Stateful` wrapper for model weights and lightweight scalar state.

    ``set_save_filter`` restricts which of the model-state keys
    (``text_encoder``, ``transformer``, ``transformer_2``) are included in
    ``state_dict()``.  This is used by the unified ``high/`` and ``low/``
    layout so that each subdir holds exactly one transformer.  ``None`` (the
    default) means "include every model that is non-None".
    """

    def __init__(
        self,
        text_encoder: torch.nn.Module | None,
        transformer: torch.nn.Module | None,
        transformer_2: torch.nn.Module | None,
        step: int = 0,
        epoch: int = 0,
        batch_idx: int = 0,
    ):
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.step = step
        self.epoch = epoch
        self.batch_idx = batch_idx
        self._save_filter: frozenset[str] | None = None

    def set_save_filter(self, keys: set[str] | frozenset[str] | None) -> None:
        self._save_filter = None if keys is None else frozenset(keys)

    def _included(self, key: str) -> bool:
        return self._save_filter is None or key in self._save_filter

    @staticmethod
    def _model_sd(model: torch.nn.Module | None):
        if model is None:
            return None
        if _is_fsdp_wrapped(model):
            return get_model_state_dict(model)
        return model.state_dict()

    def state_dict(self):
        sd: dict = {"step": self.step, "epoch": self.epoch, "batch_idx": self.batch_idx}
        if self.text_encoder is not None and self._included("text_encoder"):
            sd["text_encoder"] = self._model_sd(self.text_encoder)
        if self.transformer is not None and self._included("transformer"):
            sd["transformer"] = self._model_sd(self.transformer)
        if self.transformer_2 is not None and self._included("transformer_2"):
            sd["transformer_2"] = self._model_sd(self.transformer_2)
        sd["rng_cpu"] = torch.random.get_rng_state()
        sd["rng_cuda"] = torch.cuda.get_rng_state()
        return sd

    @staticmethod
    def _apply(model: torch.nn.Module | None, sd: dict, key: str) -> None:
        if model is None or key not in sd:
            return
        if _is_fsdp_wrapped(model):
            set_model_state_dict(model, model_state_dict=sd[key])
        else:
            model.load_state_dict(sd[key])

    def load_state_dict(self, state_dict):
        self._apply(self.text_encoder, state_dict, "text_encoder")
        self._apply(self.transformer, state_dict, "transformer")
        self._apply(self.transformer_2, state_dict, "transformer_2")
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.batch_idx = state_dict.get("batch_idx", 0)
        if "rng_cpu" in state_dict:
            torch.random.set_rng_state(state_dict["rng_cpu"])
        if "rng_cuda" in state_dict:
            torch.cuda.set_rng_state(state_dict["rng_cuda"])


# ---------------------------------------------------------------------------
# Optimizer shards — one .pt per shard rank, written outside DCP so the DCP
# save plan stays small (resume reads only the matching rank's shard).
# ---------------------------------------------------------------------------


def save_optimizer_shard(
    path: Path,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> bool:
    if model is None or optimizer is None:
        return False
    state = get_optimizer_state_dict(model, optimizer) if _is_fsdp_wrapped(model) else optimizer.state_dict()
    torch.save(state, path)
    return True


def load_optimizer_shard(
    path: Path,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> bool:
    if model is None or optimizer is None or not path.exists():
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Tolerate the legacy wrapper dict {"format_version", "process_group_size",
    # "optimizer_state"} written by older trainer versions.
    if isinstance(state, dict) and "optimizer_state" in state and "param_groups" not in state:
        state = state["optimizer_state"]
    if _is_fsdp_wrapped(model):
        set_optimizer_state_dict(model, optimizer, state)
    else:
        optimizer.load_state_dict(state)
    return True


# ---------------------------------------------------------------------------
# LoRA export — gather DTensors → CPU → PEFT-format safetensors.
# ---------------------------------------------------------------------------


def gather_full_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collective. Returns the unsharded state_dict on rank 0; ``{}`` on others.

    MUST be called on every rank in the FSDP group — internally runs an
    all-gather and offloads to CPU. Safe to call on a non-FSDP module too
    (just returns a plain CPU state_dict).
    """
    if _is_fsdp_wrapped(model):
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return get_model_state_dict(model, options=opts)
    # Non-FSDP: still move to CPU for parity with the FSDP path.
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def write_peft_lora_adapter(
    out_dir: Path,
    model: torch.nn.Module,
    full_state_dict: dict[str, torch.Tensor],
    adapter_name: str = "default",
) -> None:
    """Write ``adapter_model.safetensors`` + ``adapter_config.json`` for one PEFT adapter.

    Caller's responsibility:
      * invoke on rank 0 only,
      * pass a state_dict that has been gathered to CPU plain tensors,
      * ensure ``model`` carries the matching ``peft_config[adapter_name]``.
    """
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_sd = get_peft_model_state_dict(model, state_dict=full_state_dict, adapter_name=adapter_name)
    adapter_sd = {k: v.detach().contiguous().cpu() for k, v in adapter_sd.items()}
    save_file(adapter_sd, str(out_dir / "adapter_model.safetensors"))

    cfg = model.peft_config[adapter_name]
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg.__dict__)
    # PEFT stores ``target_modules`` as a set internally; serialize sets as
    # sorted lists so ``LoraConfig.from_pretrained`` round-trips them as lists
    # rather than stringifying via ``default=str``.
    for k, v in list(cfg_dict.items()):
        if isinstance(v, set):
            cfg_dict[k] = sorted(v)
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)


# ---------------------------------------------------------------------------
# Inference: load DCP into a non-FSDP diffusers pipeline (single process).
# Handles both full fine-tune and LoRA checkpoints — LoRA is auto-detected
# from the ``lora/<transformer>/adapter_config.json`` sidecars written by
# ``write_peft_lora_adapter`` during training.
# ---------------------------------------------------------------------------


def load_dcp_into_pipeline(pipe, checkpoint_path: str, use_ema: bool = False) -> None:
    """Load a DCP checkpoint into a diffusers pipeline (single-process, no FSDP).

    Auto-detects LoRA by probing for ``adapter_config.json`` under
    ``high/lora/transformer/`` and ``low/lora/transformer_2/``; when present,
    wraps the corresponding pipeline module with the stored ``LoraConfig``
    before loading so the DCP keys line up. Reuses the training-time init
    primitives (``read_dcp_to_flat_dict`` + ``extract_init_weights`` +
    ``remap_for_current_model``) so plain↔LoRA mismatches surface with the
    same diagnostics as training.
    """
    apply_lora_adapters_from_checkpoint(pipe, checkpoint_path)

    for name, dcp_path in _detect_layout(checkpoint_path).items():
        model = getattr(pipe, name, None)
        if model is None:
            continue
        flat = read_dcp_to_flat_dict(dcp_path)
        try:
            weights, source_tag = _extract_pipeline_weights(flat, name, model, use_ema=use_ema)
        except RuntimeError as e:
            logger.debug("No {} data in {}: {}", name, dcp_path, e)
            del flat
            gc.collect()
            continue
        remapped = remap_for_current_model(weights, model)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        logger.info(
            "Loaded {} {} weights from {} (missing={}, unexpected={})",
            name,
            source_tag,
            dcp_path,
            len(missing),
            len(unexpected),
        )
        del flat, weights, remapped
        gc.collect()


def _extract_pipeline_weights(
    flat: dict[str, torch.Tensor],
    model_key: str,
    model: torch.nn.Module,
    *,
    use_ema: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    """Select checkpoint weights for inference export/eval.

    For LoRA training, EMA tracks only trainable ``lora_*`` tensors.  The
    frozen base layers still live in ``train_state`` and may themselves come
    from a previous SFT checkpoint.  Loading EMA alone would therefore apply
    the LoRA delta to the original base model.  For LoRA-wrapped models, use
    raw train_state as the complete base and overlay EMA LoRA tensors.
    """
    if not use_ema:
        return extract_init_weights(flat, model_key, prefer="raw")

    try:
        ema_weights, _ = extract_init_weights(flat, model_key, prefer="ema")
    except RuntimeError:
        logger.warning(
            "No EMA shadow for {} in checkpoint; falling back to raw train_state weights.",
            model_key,
        )
        return extract_init_weights(flat, model_key, prefer="raw")

    if not getattr(model, "peft_config", None):
        return ema_weights, "EMA"

    try:
        raw_weights, _ = extract_init_weights(flat, model_key, prefer="raw")
    except RuntimeError:
        logger.warning(
            "LoRA EMA shadow for {} has no raw train_state base; loading EMA tensors only.",
            model_key,
        )
        return ema_weights, "EMA"

    raw_weights.update(ema_weights)
    return raw_weights, "raw+EMA"


def _detect_layout(checkpoint_path: str) -> dict[str, str]:
    """Return ``{module_name: dcp_dir}`` for flat or expert-parallel layouts."""
    p = Path(checkpoint_path)
    high, low = p / "high", p / "low"
    if (high / ".metadata").exists() or (low / ".metadata").exists():
        layout: dict[str, str] = {}
        if (high / ".metadata").exists():
            layout["transformer"] = str(high)
        if (low / ".metadata").exists():
            layout["transformer_2"] = str(low)
        return layout
    return {"transformer": checkpoint_path, "transformer_2": checkpoint_path}


def apply_lora_adapters_from_checkpoint(pipe, checkpoint_path: str) -> bool:
    """Wrap the pipeline's transformers with LoRA adapters stored in a checkpoint.

    Looks for PEFT adapter sidecars written by ``write_peft_lora_adapter``:

        {ckpt}/high/lora/transformer/     → pipe.transformer
        {ckpt}/low/lora/transformer_2/    → pipe.transformer_2

    For each present adapter, instantiates a ``LoraConfig`` from the sidecar
    and calls ``model.add_adapter(cfg)`` so the model's ``state_dict`` keys
    match the wrapped layout saved in the DCP. If the target module already
    carries a PEFT adapter (e.g. a prior checkpoint in a multi-checkpoint
    eval loop), skips re-wrapping — the subsequent DCP load will refresh
    the LoRA weights in place.

    Returns ``True`` if the checkpoint contains any LoRA adapter sidecars.
    """
    from peft import LoraConfig

    p = Path(checkpoint_path)
    targets = [
        ("transformer", [p / "high" / "lora" / "transformer", p / "lora" / "transformer"]),
        ("transformer_2", [p / "low" / "lora" / "transformer_2", p / "lora" / "transformer_2"]),
    ]
    found = False
    for name, adapter_dirs in targets:
        adapter_dir = next((d for d in adapter_dirs if (d / "adapter_config.json").exists()), None)
        if adapter_dir is None:
            continue
        found = True
        model = getattr(pipe, name, None)
        if model is None:
            continue
        if getattr(model, "peft_config", None):
            logger.info("{} already has a PEFT adapter; skipping add_adapter", name)
            continue
        cfg = LoraConfig.from_pretrained(str(adapter_dir))
        _repair_stringified_collections(cfg)
        model.add_adapter(cfg)
        logger.info("Applied LoRA adapter to {} from {}", name, adapter_dir)
    return found


def _repair_stringified_collections(cfg) -> None:
    """Backfill for legacy adapter_config.json that stored sets via ``default=str``.

    Older ``write_peft_lora_adapter`` calls dumped ``target_modules`` as e.g.
    ``"{'to_q', 'to_k'}"`` because ``json.dump(default=str)`` stringified
    the PEFT-internal set. ``LoraConfig.from_pretrained`` then surfaces that
    as a bare ``str``, which ``add_adapter`` would interpret as a single
    regex pattern and silently match nothing. Parse it back to a list.
    """
    import ast

    for field in ("target_modules", "exclude_modules", "modules_to_save"):
        value = getattr(cfg, field, None)
        if isinstance(value, str) and value.startswith(("{", "[")):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (set, frozenset)):
                parsed = sorted(parsed)
            elif isinstance(parsed, tuple):
                parsed = list(parsed)
            if isinstance(parsed, list):
                setattr(cfg, field, parsed)
                logger.info(
                    "Repaired stringified {} in adapter_config: {!r} -> {!r}",
                    field,
                    value,
                    parsed,
                )


# ---------------------------------------------------------------------------
# Init-from-checkpoint — read a DCP checkpoint as weight-only initialisation
# for a fresh run.  Handles plain → LoRA remap (full-FT → LoRA workflow).
# ---------------------------------------------------------------------------


def read_dcp_to_flat_dict(path: Path | str) -> dict[str, torch.Tensor]:
    """Single-process read of an entire DCP checkpoint into a CPU flat dict.

    Uses the same internal primitive as
    ``torch.distributed.checkpoint.format_utils.dcp_to_torch_save`` (which
    reads via ``_EmptyStateDictLoadPlanner`` + ``no_dist=True``), so this is
    safe to call inside an already-initialised distributed context — it does
    not participate in any collective.  Callers typically invoke it only on
    rank 0 to avoid redundant I/O.
    """
    # Private API, but the same one torchtitan and PyTorch's own format_utils
    # use.  Stable across 2.4+ (no public replacement yet).
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    sd: dict = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(str(path)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return _flatten_nested(sd)


def _flatten_nested(d, prefix: str = "") -> dict[str, torch.Tensor]:
    """``{"a": {"b": t}} → {"a.b": t}``. Skips non-tensor leaves (scalars, etc.)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_nested(v, key))
        elif isinstance(v, torch.Tensor):
            out[key] = v
    return out


def extract_init_weights(
    flat: dict[str, torch.Tensor],
    model_key: str,
    prefer: str = "auto",
) -> tuple[dict[str, torch.Tensor], str]:
    """Pull a single model's init weights out of a flat DCP dict.

    ``prefer`` selects which shadow to use:
      * ``"auto"`` (default): EMA if present, else raw — used by training
        init-from-checkpoint (EMA is a higher-quality init when available).
      * ``"ema"``: EMA only; raises if absent.
      * ``"raw"``: raw ``train_state`` only; raises if absent.

    Keys in the returned dict are local to the model (e.g.
    ``blocks.0.attn1.to_q.weight``).

    Returns ``(weights, source_tag)`` where ``source_tag`` is ``"EMA"`` or
    ``"raw"``.  Raises ``RuntimeError`` if the requested source is absent.
    """
    assert prefer in ("auto", "ema", "raw"), f"prefer must be auto/ema/raw, got {prefer!r}"
    ema_prefix = f"ema.shadow.{model_key}."
    raw_prefix = f"train_state.{model_key}."

    if prefer in ("auto", "ema"):
        ema_keys = [k for k in flat if k.startswith(ema_prefix)]
        if ema_keys:
            return {k[len(ema_prefix) :]: flat[k] for k in ema_keys}, "EMA"
        if prefer == "ema":
            raise RuntimeError(f"Checkpoint has no EMA data for {model_key} (prefix {ema_prefix!r}).")

    if prefer in ("auto", "raw"):
        raw_keys = [k for k in flat if k.startswith(raw_prefix)]
        if raw_keys:
            return {k[len(raw_prefix) :]: flat[k] for k in raw_keys}, "raw"

    raise RuntimeError(
        f"Checkpoint has no data for {model_key} (looked for prefixes {ema_prefix!r} and {raw_prefix!r})."
    )


def remap_for_current_model(
    weights: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Remap source keys into the current model's parameter-name namespace.

    Four cases, decided from source-vs-model PEFT wrapping:

    - **Same architecture** (both plain, or both LoRA-wrapped): identity.
    - **Plain source → LoRA model** (full-FT → LoRA workflow): insert
      ``.base_layer.`` before the trailing ``.weight`` / ``.bias`` for keys
      that have a LoRA base_layer counterpart in the model.  Non-target
      modules (e.g. ``patch_embedding.weight``) pass through unchanged.
      ``lora_A/B`` keys are absent from the result; caller loads with
      ``strict=False`` so they keep PEFT's default init.
    - **LoRA source → plain model**: raises — requires an offline merge.
    - **LoRA source → LoRA model, different rank/targets**: raises.

    Shape mismatches on same-name keys also raise (caller should then either
    change ``model_path`` or convert offline).
    """
    model_keys = {k: v.shape for k, v in model.state_dict().items()}
    source_has_lora = _has_peft_segments(weights)
    model_has_lora = _has_peft_segments(model_keys)

    if source_has_lora and not model_has_lora:
        raise _lora_to_plain_error()
    if source_has_lora and model_has_lora:
        return _remap_lora_to_lora(weights, model_keys)
    if not source_has_lora and model_has_lora:
        return _remap_plain_to_lora(weights, model_keys)
    # Both plain.
    return _remap_same_arch(weights, model_keys)


def _has_peft_segments(keys) -> bool:
    """Match PEFT path segments (``base_layer`` or ``lora_*``) anywhere in the key."""
    for k in keys:
        for part in k.split("."):
            if part == "base_layer" or part.startswith("lora_"):
                return True
    return False


def _remap_same_arch(weights: dict[str, torch.Tensor], model_keys: dict[str, torch.Size]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in weights.items():
        if k not in model_keys:
            logger.warning("Source key {!r} has no counterpart in model; dropped.", k)
            continue
        if tuple(v.shape) != tuple(model_keys[k]):
            raise ValueError(f"Shape mismatch on {k}: source {tuple(v.shape)} vs model {tuple(model_keys[k])}.")
        out[k] = v
    return out


def _remap_plain_to_lora(
    weights: dict[str, torch.Tensor], model_keys: dict[str, torch.Size]
) -> dict[str, torch.Tensor]:
    out = {}
    for src_key, tensor in weights.items():
        # Non-target module — key exists as-is in the LoRA-wrapped model.
        if src_key in model_keys:
            if tuple(tensor.shape) != tuple(model_keys[src_key]):
                raise ValueError(
                    f"Shape mismatch on {src_key}: source {tuple(tensor.shape)} vs model {tuple(model_keys[src_key])}."
                )
            out[src_key] = tensor
            continue
        # LoRA-wrapped target — insert `.base_layer` before the suffix.
        for suffix in (".weight", ".bias"):
            if src_key.endswith(suffix):
                base_key = src_key[: -len(suffix)] + ".base_layer" + suffix
                if base_key in model_keys:
                    if tuple(tensor.shape) != tuple(model_keys[base_key]):
                        raise ValueError(
                            f"Shape mismatch on {base_key}: source {tuple(tensor.shape)} "
                            f"vs model {tuple(model_keys[base_key])}."
                        )
                    out[base_key] = tensor
                    break
        else:
            logger.warning(
                "Source key {!r} has no plain or .base_layer. counterpart in model; dropped.",
                src_key,
            )
    return out


def _remap_lora_to_lora(weights: dict[str, torch.Tensor], model_keys: dict[str, torch.Size]) -> dict[str, torch.Tensor]:
    out = {}
    mismatches: list[str] = []
    for k, v in weights.items():
        if k not in model_keys:
            mismatches.append(f"  - key missing in model: {k}")
            continue
        if tuple(v.shape) != tuple(model_keys[k]):
            mismatches.append(f"  - {k}: source {tuple(v.shape)} vs model {tuple(model_keys[k])}")
            continue
        out[k] = v
    if mismatches:
        # Most common cause: LoRA rank / target_modules changed between runs.
        head = "\n".join(mismatches[:10])
        more = f"\n  ...and {len(mismatches) - 10} more" if len(mismatches) > 10 else ""
        raise ValueError(
            "LoRA checkpoint does not match current LoRA architecture.\n"
            "Likely cause: lora_rank or target_modules differ between runs.\n"
            f"First mismatches:\n{head}{more}\n"
            "Fix: merge the source LoRA into its base (see the LoRA→plain "
            "error message below for the recipe), then re-LoRA fresh from "
            "the merged model_path."
        )
    return out


def _lora_to_plain_error() -> ValueError:
    return ValueError(
        "Cannot init a full-FT model from a LoRA-trained checkpoint — the LoRA "
        "delta must be merged into base weights first. Recipe:\n"
        "  1. .venv/bin/python -m src.cli.convert_dcp_to_lora \\\n"
        "         --config <original_train_config.yaml> \\\n"
        "         --checkpoint <ckpt_path> \\\n"
        "         --output <adapter_out>\n"
        "  2. Use PEFT merge_and_unload to fold the adapter into base weights:\n"
        "         from diffusers.models import WanTransformer3DModel\n"
        "         from peft import PeftModel\n"
        "         base = WanTransformer3DModel.from_pretrained('<orig>/transformer')\n"
        "         merged = PeftModel.from_pretrained(base, '<adapter_out>/transformer').merge_and_unload()\n"
        "         merged.save_pretrained('<merged_model>/transformer')\n"
        "  3. Point cfg.model_path at <merged_model> and start a fresh run "
        "     (no resume_from)."
    )
