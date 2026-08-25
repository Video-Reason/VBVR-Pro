"""Exponential Moving Average for FSDP2 training with DCP support."""

import torch
from torch.distributed.checkpoint.stateful import Stateful


class EMA(Stateful):
    """Exponential Moving Average of model parameters.

    Works with FSDP2: each rank maintains EMA of its local parameter shards
    as DTensors, enabling DCP save/load with automatic resharding support.

    ``set_save_filter`` restricts which ``shadow.<model_name>.*`` groups are
    included in ``state_dict()``.  The unified save layout uses this so that
    ``high/`` only carries ``transformer`` (+ ``text_encoder``) shadow, and
    ``low/`` only carries ``transformer_2`` (+ ``text_encoder``) shadow.
    ``None`` (the default) means "include everything".
    """

    def __init__(self, models: dict[str, torch.nn.Module], decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for model_name, model in models.items():
            for pname, p in model.named_parameters():
                if p.requires_grad:
                    s = p.data.clone()
                    self.shadow[f"{model_name}.{pname}"] = s
                    self._pairs.append((p, s))
        self._save_filter: frozenset[str] | None = None

    def set_save_filter(self, model_names: set[str] | frozenset[str] | None) -> None:
        self._save_filter = None if model_names is None else frozenset(model_names)

    @torch.no_grad()
    def update(self):
        for p, s in self._pairs:
            s.lerp_(p.data, 1 - self.decay)

    def reinitialize(self):
        """Reset shadow parameters to current model weights."""
        for p, s in self._pairs:
            s.copy_(p.data)

    def state_dict(self) -> dict:
        if self._save_filter is None:
            return {"shadow": self.shadow, "decay": self.decay}
        wanted = self._save_filter
        shadow = {k: v for k, v in self.shadow.items() if k.split(".", 1)[0] in wanted}
        return {"shadow": shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict) -> None:
        loaded_shadow = state_dict["shadow"]
        for key, s in self.shadow.items():
            if key in loaded_shadow:
                s.copy_(loaded_shadow[key])
        if "decay" in state_dict:
            self.decay = state_dict["decay"]
