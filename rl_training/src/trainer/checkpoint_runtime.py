"""Checkpoint save/load orchestration — unified ``high/`` + ``low/`` layout.

Every checkpoint on disk has the same shape regardless of whether the trainer
was flat or expert-parallel when it wrote it::

    checkpoint-N/
      ├─ high/                          # transformer (+ shared scalars)
      │   ├─ .metadata + *.distcp       # DCP: train_state (filter=text_encoder+transformer)
      │   │                             #      + ema  (filter=text_encoder+transformer)
      │   ├─ optimizer_transformer_rank{R}.pt
      │   ├─ optimizer_text_encoder_rank{R}.pt   # if trained, duplicated in low/
      │   ├─ dataloader_rank{R}.pt               # duplicated in low/
      │   └─ lora/transformer/{adapter_model.safetensors, adapter_config.json}
      └─ low/                           # transformer_2 (+ shared scalars — duplicated)
          └─ ...                        # symmetric to high/

Shared scalars (``step``, ``epoch``, ``batch_idx``, RNG, and ``text_encoder``
weights when trained) are duplicated across both subdirs.  Duplicates are
tiny relative to transformer weights.  The benefit: **a flat trainer and an
EP trainer write and read exactly the same layout**, so cross-layout resume
just works.

Legacy flat checkpoints (top-level ``.metadata`` with no ``high``/``low``
subdirs) are still loadable as a fallback; they are never *written* again.

Host trainer contract — the mixin reads:

    self.cfg, self.model, self.ema, self.train_state, self.dataloader,
    self.rank, self.mesh,
    self.expert_parallel, self.expert_group, self.dp_rank, self._dp_pg,
    self._dcp_pg,
    self._reset_on_load, self.optimizer_te, self.optimizer_1, self.optimizer_2,
    self.total_steps,

and calls:

    self._barrier()
    self._build_optimizers(cfg)
    self._compute_total_steps()
    self._checkpoint_rank() -> int
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcp_state_dict
from loguru import logger
from torch.distributed._state_dict_utils import _distribute_tensors, _TensorInfo
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict, _stateful_to_state_dict
from torch.distributed.tensor import DTensor

from src.trainer.checkpoint import (
    MODEL_KEYS,
    extract_init_weights,
    gather_full_state_dict,
    load_optimizer_shard,
    read_dcp_to_flat_dict,
    remap_for_current_model,
    save_optimizer_shard,
    write_peft_lora_adapter,
)


@dataclass
class _SubdirEntry:
    """One ``high/`` or ``low/`` subdir that this rank should save or load."""

    subdir: str  # "high" | "low"
    transformer_key: str  # "transformer" | "transformer_2"
    transformer_model: torch.nn.Module | None
    transformer_optimizer: torch.optim.Optimizer | None
    shard_rank: int
    dcp_kwargs: dict  # e.g. {"process_group": dp_pg} for EP
    dcp_group_size: int
    dcp_coordinator_rank: int


def _set_model_state_dict_from_group_root(
    model: torch.nn.Module,
    model_state_dict: dict[str, Any],
    *,
    object_pg: dist.ProcessGroup | None,
    tensor_pg: dist.ProcessGroup | None,
    options: StateDictOptions,
):
    """Like ``set_model_state_dict(..., broadcast_from_rank0=True)`` with group-local rank 0.

    PyTorch's helper hard-codes global rank 0 as the full-state source.  That is
    wrong for expert-parallel weight-only init: the low expert's source rank is
    local rank 0 of the low expert group, not global rank 0.
    """

    model_state_dict = dcp_state_dict._unflatten_model_state_dict(model, model_state_dict)
    with dcp_state_dict._gc_context():
        info = dcp_state_dict._verify_options(model, (), optim_only=False, options=options)
        dcp_state_dict._verify_state_dict(model_state_dict, {}, info)
        return _load_model_state_dict_from_group_root(
            model,
            model_state_dict,
            info,
            object_pg=object_pg,
            tensor_pg=tensor_pg,
        )


@torch.no_grad()
def _load_model_state_dict_from_group_root(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    info,
    *,
    object_pg: dist.ProcessGroup | None,
    tensor_pg: dist.ProcessGroup | None,
):
    if not info.handle_model:
        return None

    local_state_dict: dict[str, Any] = {}
    is_root = _group_rank_is_zero(object_pg)
    for key, value in dcp_state_dict._iterate_valid_model_state(model, info.dsd_fqn_modifiers):
        fqns = dcp_state_dict._get_fqns(model, key, info.dsd_fqn_modifiers)
        fqns_with_prefix = dcp_state_dict._get_fqns(
            model,
            key,
            info.dsd_fqn_modifiers,
            skip_ddp_prefix=False,
            skip_compiler_prefix=False,
        )

        for fqn, fqn_with_prefix in zip(fqns, fqns_with_prefix, strict=False):
            if is_root and fqn != fqn_with_prefix:
                load_value = state_dict.pop(fqn, None)
                if load_value is None:
                    if info.strict:
                        raise RuntimeError(f"Missing key: {fqn}.")
                else:
                    state_dict[fqn_with_prefix] = load_value
            local_state_dict[fqn_with_prefix] = value

    assign = False
    devices = set()
    for value in local_state_dict.values():
        if torch.is_tensor(value) and value.dim() > 0:
            devices.add(value.device)
    if torch.device("meta") in devices:
        devices.remove(torch.device("meta"))
        assign = True
    if len(devices) == 0:
        devices.add(dist.distributed_c10d._get_pg_default_device())
    elif len(devices) > 1:
        raise ValueError("Multiple devices found")

    _broadcast_state_dict_from_group_root(
        state_dict,
        local_state_dict,
        device=devices.pop(),
        object_pg=object_pg,
        tensor_pg=tensor_pg,
        strict=info.strict,
        cpu_offload=info.cpu_offload,
    )
    state_dict.update(local_state_dict)

    with info.fsdp_context():
        return dcp_state_dict._state_dict_fn(model, "load_state_dict")(
            state_dict=state_dict,
            strict=info.strict,
            assign=assign,
        )


def _group_rank_is_zero(pg: dist.ProcessGroup | None) -> bool:
    if pg is None:
        return dist.get_rank() == 0
    return dist.get_rank(group=pg) == 0


def _broadcast_object_list_from_group_root(objects: list[Any], pg: dist.ProcessGroup | None) -> None:
    if pg is None:
        dist.broadcast_object_list(objects, src=0)
    else:
        dist.broadcast_object_list(objects, group=pg, group_src=0)


def _broadcast_tensor_from_group_root(tensor: torch.Tensor, pg: dist.ProcessGroup | None) -> None:
    if pg is None:
        dist.broadcast(tensor, src=0)
    else:
        dist.broadcast(tensor, group=pg, group_src=0)


def _broadcast_state_dict_from_group_root(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    *,
    device: torch.device,
    object_pg: dist.ProcessGroup | None,
    tensor_pg: dist.ProcessGroup | None,
    strict: bool,
    cpu_offload: bool,
) -> None:
    ret: dict[str, Any] = {}
    is_root = _group_rank_is_zero(object_pg)
    if is_root:
        for key, value in full_state_dict.items():
            if not torch.is_tensor(value):
                ret[key] = value
            elif value.dim() == 0:
                ret[key] = value.cpu()
            else:
                ret[key] = _TensorInfo(value.size(), value.dtype)

    broadcast_list: list[Any] = [ret]
    _broadcast_object_list_from_group_root(broadcast_list, object_pg)
    ret = broadcast_list[0]

    local_state_dict_keys = set(local_state_dict.keys())
    global_keys = set()
    for key, value in ret.items():
        global_keys.add(key)
        if not isinstance(value, _TensorInfo):
            if key in local_state_dict:
                local_state_dict[key] = value
            continue

        if is_root:
            ret[key] = full_state_dict[key]

        _broadcast_state_tensor_from_group_root(ret, local_state_dict, key, device, tensor_pg)
        if cpu_offload and torch.is_tensor(local_state_dict.get(key)):
            local_state_dict[key] = local_state_dict[key].cpu()

    if strict:
        for key in local_state_dict_keys - global_keys:
            local_state_dict.pop(key)


def _broadcast_state_tensor_from_group_root(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    key: str,
    device: torch.device,
    pg: dist.ProcessGroup | None,
) -> None:
    if pg is None:
        pg = dist.distributed_c10d._get_default_group()
    pg_device = device if device.type in {pg_device.type for pg_device in pg._device_types} else pg._device_types[0]

    if _group_rank_is_zero(pg):
        full_state = full_state_dict[key]
        if not isinstance(full_state, torch.Tensor):
            raise AssertionError("full_state must be a torch.Tensor")
        full_tensor = full_state.detach().to(pg_device)
    else:
        tensor_info = full_state_dict[key]
        full_tensor = torch.empty(
            size=tensor_info.size,
            device=pg_device,
            dtype=tensor_info.dtype,
        )

    if (local_state := local_state_dict.get(key)) is not None:
        local_state_dict[key] = (local_state, full_tensor) if isinstance(local_state, DTensor) else full_tensor

    _broadcast_tensor_from_group_root(full_tensor, pg)

    if pg_device != device and (local_state := local_state_dict.get(key)) is not None:
        local_state_dict[key] = (
            (local_state[0], full_tensor.to(device))
            if isinstance(local_state, tuple) and isinstance(local_state[0], DTensor)
            else full_tensor.to(device)
        )

    _distribute_tensors(local_state_dict, [key], device, pg)


class CheckpointRuntimeMixin:
    def _expert_parallel_dcp_kwargs(self) -> dict:
        """Return DCP kwargs for this expert-parallel group.

        DCP uses Python object collectives while planning.  Keep the training
        DP group on NCCL, but prefer a dedicated Gloo group for checkpoint
        metadata to avoid NCCL object-collective failures during save/load.
        """

        if not self.expert_parallel:
            return {}
        pg = getattr(self, "_dcp_pg", None) or self._dp_pg
        return {"process_group": pg}

    def _flat_dcp_kwargs(self) -> dict:
        """Return DCP kwargs for non-EP trainers.

        In split RL, only train ranks participate in checkpoint collectives,
        so DCP must use the train-only process group instead of the default
        world group that also contains rollout actors.
        """
        if getattr(self, "rl_split_enabled", False):
            pg = getattr(self, "_dcp_pg", None) or getattr(self, "_train_gloo_pg", None)
            if pg is not None:
                return {"process_group": pg}
        return {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_latest_checkpoint(self) -> str | None:
        out = Path(self.cfg.output_dir)
        if not out.exists():
            return None
        candidates: list[tuple[int, Path]] = []
        for d in out.iterdir():
            if not d.is_dir() or not _is_checkpoint_dir(d) or not _has_valid_step_suffix(d.name):
                continue
            candidates.append((d.stat().st_mtime_ns, d))
        if not candidates:
            return None
        candidates.sort()
        latest = candidates[-1][1]
        logger.info("Auto-resume: found checkpoint {}", latest)
        return str(latest)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path):
        self._assert_finite_checkpoint_state(path)
        path.mkdir(parents=True, exist_ok=True)
        entries = list(self._save_plan())
        if not entries:
            raise RuntimeError("Nothing to save — no transformer is attached to this trainer.")
        for entry in entries:
            self._save_subdir(path, entry)
        self._barrier()
        if entries[0].shard_rank == 0:
            logger.info(
                "Saved checkpoint to {} (subdirs: {})",
                path,
                ", ".join(e.subdir for e in entries),
            )

    def _assert_finite_checkpoint_state(self, path: Path) -> None:
        """Fail all ranks before DCP save if any local parameter or EMA shard is non-finite."""
        problem = self._find_nonfinite_checkpoint_tensor()
        if problem is not None:
            logger.error("Refusing to save {}: {}", path, problem)

        bad = torch.tensor([problem is not None], dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=self._checkpoint_pg)
        if bad.item():
            raise FloatingPointError(
                f"Non-finite parameter or EMA tensor detected before saving {path}. "
                "See rank logs for the first offending tensor."
            )

    def _find_nonfinite_checkpoint_tensor(self) -> str | None:
        for module_name in ("text_encoder", "transformer", "transformer_2"):
            module = getattr(self.model, module_name, None)
            if module is None:
                continue
            for param_name, param in module.named_parameters():
                problem = _nonfinite_tensor_summary(param.detach(), f"{module_name}.{param_name}")
                if problem is not None:
                    return problem

        ema = getattr(self, "ema", None)
        if ema is not None:
            for name, tensor in ema.shadow.items():
                problem = _nonfinite_tensor_summary(tensor, f"ema.shadow.{name}")
                if problem is not None:
                    return problem
        return None

    def _save_plan(self):
        if self.expert_parallel:
            sub = "high" if self.expert_group == 0 else "low"
            key = "transformer" if sub == "high" else "transformer_2"
            model = self.model.transformer if sub == "high" else self.model.transformer_2
            optimizer = self.optimizer_1 if sub == "high" else self.optimizer_2
            yield _SubdirEntry(
                subdir=sub,
                transformer_key=key,
                transformer_model=model,
                transformer_optimizer=optimizer,
                shard_rank=self.dp_rank,
                dcp_kwargs=self._expert_parallel_dcp_kwargs(),
                dcp_group_size=self.dp_size,
                dcp_coordinator_rank=self._resolve_dcp_coordinator_rank(self.dp_size),
            )
            return
        # Flat: save each attached transformer to its own subdir, all ranks
        # participate in each DCP collective on the default world pg.
        for sub, key, model, optimizer in [
            ("high", "transformer", self.model.transformer, self.optimizer_1),
            ("low", "transformer_2", self.model.transformer_2, self.optimizer_2),
        ]:
            if model is None:
                continue
            yield _SubdirEntry(
                subdir=sub,
                transformer_key=key,
                transformer_model=model,
                transformer_optimizer=optimizer,
                shard_rank=self.rank,
                dcp_kwargs=self._flat_dcp_kwargs(),
                dcp_group_size=self.world_size,
                dcp_coordinator_rank=self._resolve_dcp_coordinator_rank(self.world_size),
            )

    def _resolve_dcp_coordinator_rank(self, group_size: int) -> int:
        configured = self.cfg.checkpoint_dcp_coordinator_rank
        if configured is None:
            return 0
        rank = int(configured)
        if rank < 0:
            rank += group_size
        if not 0 <= rank < group_size:
            raise ValueError(
                f"checkpoint_dcp_coordinator_rank={configured} resolves to {rank}, outside DCP group size {group_size}",
            )
        return rank

    def _dcp_save(self, state: dict, checkpoint_id: Path, entry: _SubdirEntry) -> None:
        writer = dcp.FileSystemWriter(
            str(checkpoint_id),
            thread_count=self.cfg.checkpoint_dcp_thread_count,
            sync_files=self.cfg.checkpoint_dcp_sync_files,
        )
        coordinator_global_rank = entry.dcp_coordinator_rank
        process_group = entry.dcp_kwargs.get("process_group")
        if process_group is not None:
            coordinator_global_rank = dist.get_global_rank(process_group, entry.dcp_coordinator_rank)
        if getattr(self, "log_enabled", self.rank == 0):
            logger.info(
                "Checkpoint DCP save {}: coordinator rank {} (local {}/{}), writer_threads={}, sync_files={}",
                checkpoint_id,
                coordinator_global_rank,
                entry.dcp_coordinator_rank,
                entry.dcp_group_size,
                self.cfg.checkpoint_dcp_thread_count,
                self.cfg.checkpoint_dcp_sync_files,
            )
        _save_state_dict(
            state_dict=_stateful_to_state_dict(state),
            storage_writer=writer,
            process_group=process_group,
            coordinator_rank=entry.dcp_coordinator_rank,
        )

    def _save_subdir(self, root: Path, entry: _SubdirEntry) -> None:
        sub_path = root / entry.subdir
        sub_path.mkdir(parents=True, exist_ok=True)
        filter_keys = self._subdir_filter_keys(entry.transformer_key)

        # 1. DCP — train_state + EMA, restricted to this subdir's models.
        self.train_state.set_save_filter(filter_keys)
        if self.ema is not None:
            self.ema.set_save_filter(filter_keys)
        try:
            state: dict = {"train_state": self.train_state}
            if self.ema is not None:
                state["ema"] = self.ema
            self._dcp_save(state, sub_path, entry)
        finally:
            self.train_state.set_save_filter(None)
            if self.ema is not None:
                self.ema.set_save_filter(None)

        # 2. Optimizer shards — transformer for this subdir, plus text_encoder
        #    when trained (duplicated across both subdirs).
        if not self._hsdp_skip_optimizer_shard():
            save_optimizer_shard(
                sub_path / f"optimizer_{entry.transformer_key}_rank{entry.shard_rank}.pt",
                entry.transformer_model,
                entry.transformer_optimizer,
            )
            if self.cfg.train_text_encoder and self.model.text_encoder is not None:
                save_optimizer_shard(
                    sub_path / f"optimizer_text_encoder_rank{entry.shard_rank}.pt",
                    self.model.text_encoder,
                    self.optimizer_te,
                )

        # 3. Dataloader — per shard rank (duplicated across subdirs).
        torch.save(self.dataloader.state_dict(), sub_path / f"dataloader_rank{entry.shard_rank}.pt")

        # 4. LoRA adapter — gather is collective, writer is rank 0.
        if self.model.lora_config is not None and entry.transformer_model is not None:
            full_sd = gather_full_state_dict(entry.transformer_model)
            if entry.shard_rank == 0:
                write_peft_lora_adapter(
                    sub_path / "lora" / entry.transformer_key,
                    entry.transformer_model,
                    full_sd,
                )

    def _subdir_filter_keys(self, transformer_key: str) -> frozenset[str]:
        """Keys this subdir is responsible for (transformer + text_encoder if trained)."""
        keys = {transformer_key}
        if self.cfg.train_text_encoder and self.model.text_encoder is not None:
            keys.add("text_encoder")
        return frozenset(keys)

    def _hsdp_skip_optimizer_shard(self) -> bool:
        """Under HSDP, only the first replica writes optimizer shards."""
        return self.mesh is not None and self.mesh.ndim == 2 and self.mesh.get_local_rank("replicate") > 0

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str):
        ckpt = Path(path)
        entries = list(self._load_plan(ckpt))
        if not entries:
            raise ValueError(
                f"Checkpoint at {path} is missing both high/.metadata and low/.metadata "
                f"(and no legacy top-level .metadata). Cannot resume."
            )

        mode = "init" if self._reset_on_load else "resume"
        logger.info(
            "{} from {} ({}) ...",
            "Initializing" if mode == "init" else "Resuming",
            ckpt,
            "legacy flat root" if entries[0].subdir == "" else "subdirs: " + ", ".join(e.subdir for e in entries),
        )
        for entry in entries:
            self._load_subdir(entry)

        if self._reset_on_load:
            # Weight-only init: EMA tracks the newly-initialised model weights.
            if self.ema is not None:
                self.ema.reinitialize()
                logger.info("EMA reinitialized from loaded model weights.")
            self._reset_training_state()
            logger.info(
                "reset_dataloader=True: reset step/epoch/optimizer, total_steps={}",
                self.total_steps,
            )

        logger.info(
            "Resumed at step={} epoch={} batch_idx={}",
            self.train_state.step,
            self.train_state.epoch,
            self.train_state.batch_idx,
        )

    def _load_plan(self, ckpt: Path):
        """Build the list of subdirs (or a single legacy-root entry) to load from."""
        high_ok = (ckpt / "high" / ".metadata").exists()
        low_ok = (ckpt / "low" / ".metadata").exists()

        if not high_ok and not low_ok:
            # Legacy flat layout: single DCP at the root holding every model.
            if not (ckpt / ".metadata").exists():
                return
            yield _LoadEntry(
                subdir="",
                path=ckpt,
                filter_keys=None,  # load everything that matches what's in TrainState
                shard_rank=self._checkpoint_rank(),
                dcp_kwargs=self._expert_parallel_dcp_kwargs() if self.expert_parallel else self._flat_dcp_kwargs(),
                transformer_key=None,
            )
            return

        # Unified layout: iterate high/ then low/.  EP trainers only load their
        # own group's subdir; flat trainers load both when both are present.
        for sub, key in [("high", "transformer"), ("low", "transformer_2")]:
            sub_path = ckpt / sub
            if not (sub_path / ".metadata").exists():
                continue
            if self.expert_parallel:
                want = "high" if self.expert_group == 0 else "low"
                if sub != want:
                    continue
            if getattr(self.model, key) is None:
                # Trainer doesn't actually own this transformer; skip.
                logger.warning("Skipping {} (trainer has no {} attached).", sub_path, key)
                continue
            yield _LoadEntry(
                subdir=sub,
                path=sub_path,
                filter_keys=self._subdir_filter_keys(key),
                shard_rank=self.dp_rank if self.expert_parallel else self.rank,
                dcp_kwargs=self._expert_parallel_dcp_kwargs() if self.expert_parallel else self._flat_dcp_kwargs(),
                transformer_key=key,
            )

    def _load_subdir(self, entry: _LoadEntry) -> None:
        if self._reset_on_load:
            self._load_for_init(entry)
        else:
            self._load_for_resume(entry)

    # ------------------------------------------------------------------
    # Resume — full state (model + optimizer + dataloader + EMA + counters).
    # ------------------------------------------------------------------

    def _load_for_resume(self, entry: _LoadEntry) -> None:
        # 1. DCP — train_state + EMA.
        self.train_state.set_save_filter(entry.filter_keys)
        if self.ema is not None:
            self.ema.set_save_filter(entry.filter_keys)
        try:
            state: dict = {"train_state": self.train_state}
            if self.ema is not None:
                state["ema"] = self.ema
            try:
                dcp.load(state, checkpoint_id=str(entry.path), **entry.dcp_kwargs)
            except Exception as e:
                if "ema" in state:
                    logger.warning(
                        "Failed to load EMA from {} ({}); retrying without it.",
                        entry.path,
                        type(e).__name__,
                    )
                    dcp.load({"train_state": self.train_state}, checkpoint_id=str(entry.path), **entry.dcp_kwargs)
                    self.ema.reinitialize()
                    logger.warning("EMA reinitialized from loaded model weights.")
                else:
                    raise
        finally:
            self.train_state.set_save_filter(None)
            if self.ema is not None:
                self.ema.set_save_filter(None)

        # 2. Optimizer shards + dataloader.
        restored = self._load_optimizer_shards_for(entry)
        if restored:
            logger.info("Restored optimizer shards from {}: {}", entry.path, ", ".join(restored))
        elif entry.transformer_key is not None:
            logger.warning(
                "No optimizer shards found in {} for {}; optimizer state starts fresh.",
                entry.path,
                entry.transformer_key,
            )
        dl_state_path = entry.path / f"dataloader_rank{entry.shard_rank}.pt"
        if dl_state_path.exists():
            self.dataloader.load_state_dict(torch.load(dl_state_path, weights_only=False))
            logger.info("Restored dataloader state from {}", dl_state_path)

    # ------------------------------------------------------------------
    # Init — weight-only load (prefer EMA shadow; auto plain → LoRA remap).
    # ------------------------------------------------------------------

    def _load_for_init(self, entry: _LoadEntry) -> None:
        """Weight-only init from a previous checkpoint.

        The local rank 0 for this DCP group materialises the entire DCP into a
        CPU flat dict once (~28 GB for a 14B bf16 model — one-shot, acceptable
        for startup), extracts and remaps weights per model, then all ranks in
        the same FSDP group reshard from that full state.

        EMA is *not* loaded here; it will be reinitialised once at the end of
        ``_load_checkpoint`` to track the newly-initialised model weights.
        Optimizer and dataloader state are not loaded either — they are reset
        via ``_reset_training_state()`` also at the end of ``_load_checkpoint``.
        """
        # Which models to init from this subdir.
        if entry.transformer_key is not None:
            model_specs: list[tuple[str, torch.nn.Module | None]] = [
                (entry.transformer_key, getattr(self.model, entry.transformer_key)),
            ]
        else:
            # Legacy flat root — a single DCP with all models.
            model_specs = [
                ("transformer", self.model.transformer),
                ("transformer_2", self.model.transformer_2),
            ]
        # text_encoder (when trained) is duplicated in both subdirs, so loading
        # it twice across high/low is idempotent — same CPU tensors broadcast.
        if self.cfg.train_text_encoder and self.model.text_encoder is not None:
            model_specs.append(("text_encoder", self.model.text_encoder))

        init_root = entry.shard_rank == 0
        flat: dict[str, torch.Tensor] | None = None
        if init_root:
            flat = read_dcp_to_flat_dict(entry.path)

        try:
            for model_key, model in model_specs:
                if model is None:
                    continue
                self._broadcast_init_weights(flat, model_key, model, entry.path, init_root=init_root)
        finally:
            # Release the full CPU dict before the next subdir's pass.
            del flat

    def _broadcast_init_weights(
        self,
        flat: dict[str, torch.Tensor] | None,
        model_key: str,
        model: torch.nn.Module,
        source_path: Path,
        *,
        init_root: bool,
    ) -> None:
        """Extract + remap on the local DCP root, then broadcast-reshard."""
        remapped: dict[str, torch.Tensor] = {}
        source_tag: str | None = None

        if init_root and flat is not None:
            try:
                weights, source_tag = extract_init_weights(flat, model_key)
                remapped = remap_for_current_model(weights, model)
            except RuntimeError as e:
                # Model simply isn't in this subdir's DCP (e.g. transformer_2
                # not in a high/ subdir) — skip it silently.
                logger.debug("No {} data in {}: {}", model_key, source_path, e)
                remapped, source_tag = {}, None

        # Collective: every rank participates, even with an empty dict.
        options = StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=False,  # tolerate lora_A/B absent from plain source
        )
        if self.expert_parallel:
            object_pg = self._expert_parallel_dcp_kwargs().get("process_group")
            tensor_pg = getattr(self, "_dp_pg", None) or object_pg
            _set_model_state_dict_from_group_root(
                model,
                remapped,
                object_pg=object_pg or tensor_pg,
                tensor_pg=tensor_pg,
                options=options,
            )
        elif getattr(self, "rl_split_enabled", False):
            object_pg = self._flat_dcp_kwargs().get("process_group")
            tensor_pg = getattr(self, "_dp_pg", None) or object_pg
            _set_model_state_dict_from_group_root(
                model,
                remapped,
                object_pg=object_pg or tensor_pg,
                tensor_pg=tensor_pg,
                options=options,
            )
        else:
            set_model_state_dict(model, model_state_dict=remapped, options=options)

        if init_root and source_tag is not None:
            logger.info("Initialized {} from {} shadows at {}", model_key, source_tag, source_path)

    def _load_optimizer_shards_for(self, entry: _LoadEntry) -> list[str]:
        """Load any optimizer shards present in this subdir (new layout).

        Shard names are ``optimizer_<name>_rank{R}.pt``.  We look for
        ``transformer`` / ``transformer_2`` / ``text_encoder`` variants; absent
        files are skipped silently.  Legacy checkpoints (where optimizers were
        stored inside DCP) have no shard files here, so this returns ``[]``
        and the trainer runs with freshly initialised optimizer state.
        """
        restored: list[str] = []
        candidates: list[tuple[str, torch.nn.Module | None, torch.optim.Optimizer | None]] = []
        if entry.transformer_key == "transformer":
            candidates.append(("transformer", self.model.transformer, self.optimizer_1))
        elif entry.transformer_key == "transformer_2":
            candidates.append(("transformer_2", self.model.transformer_2, self.optimizer_2))
        else:
            # Legacy root: try all three names.
            candidates.extend(
                [
                    ("transformer", self.model.transformer, self.optimizer_1),
                    ("transformer_2", self.model.transformer_2, self.optimizer_2),
                ]
            )
        if self.cfg.train_text_encoder and self.model.text_encoder is not None:
            candidates.append(("text_encoder", self.model.text_encoder, self.optimizer_te))

        for name, model, optimizer in candidates:
            path = entry.path / f"optimizer_{name}_rank{entry.shard_rank}.pt"
            if load_optimizer_shard(path, model, optimizer):
                restored.append(name)
        return restored

    # ------------------------------------------------------------------
    # Reset helper
    # ------------------------------------------------------------------

    def _reset_training_state(self):
        self.train_state.step = 0
        self.train_state.epoch = 0
        self.train_state.batch_idx = 0
        (
            self.params,
            self.optimizers,
            self.optimizer_te,
            self.optimizer_1,
            self.optimizer_2,
            self.fallback_te,
            self.fallback_1,
            self.fallback_2,
        ) = self._build_optimizers(self.cfg)
        self.total_steps = self._compute_total_steps()


@dataclass
class _LoadEntry:
    subdir: str  # "high" | "low" | "" (legacy root)
    path: Path
    filter_keys: frozenset[str] | None  # which models to load (None = no filter)
    shard_rank: int
    dcp_kwargs: dict
    transformer_key: str | None  # transformer | transformer_2 | None (legacy)


def _is_checkpoint_dir(d: Path) -> bool:
    return (d / ".metadata").exists() or (d / "high" / ".metadata").exists() or (d / "low" / ".metadata").exists()


def _has_valid_step_suffix(name: str) -> bool:
    if name.startswith("checkpoint-epoch"):
        suffix = name.removeprefix("checkpoint-epoch")
    elif name.startswith("checkpoint-"):
        suffix = name.removeprefix("checkpoint-")
    else:
        return False
    try:
        int(suffix)
        return True
    except ValueError:
        return False


def _nonfinite_tensor_summary(tensor: torch.Tensor, name: str) -> str | None:
    local = getattr(tensor, "_local_tensor", tensor)
    local = local.detach()
    if not torch.is_floating_point(local):
        return None
    finite = torch.isfinite(local)
    if finite.all().item():
        return None
    nan_count = torch.isnan(local).sum().item()
    inf_count = torch.isinf(local).sum().item()
    return f"{name} has non-finite values on local shard (nan={nan_count}, inf={inf_count})"


# Re-export so call sites that import MODEL_KEYS from checkpoint_runtime still work.
__all__ = ["CheckpointRuntimeMixin", "MODEL_KEYS"]
