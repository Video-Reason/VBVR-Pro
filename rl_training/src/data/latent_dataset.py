"""Dataset for precomputed latents stored in safetensors.

Config JSON format:
    [{"data_path": "/path/to/shard.safetensors"}, ...]

Each safetensors file contains tensors keyed as:
    {i}.prompt_embeds, {i}.latents, {i}.condition
with metadata header: count (int), prompts (JSON list of strings).
"""

import bisect
import json
import logging
from pathlib import Path

from safetensors import safe_open
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LatentDataset(Dataset):
    """Safetensors-backed dataset of precomputed latents."""

    def __init__(self, json_path: str):
        config_path = Path(json_path)
        raw = json.loads(config_path.read_text())
        entries = [raw] if isinstance(raw, dict) else raw

        self._shards: list[safe_open] = []
        self._prompts: list[list[str]] = []
        self._cumulative: list[int] = []
        total = 0

        for entry in entries:
            data_path = Path(entry["data_path"])
            if not data_path.is_absolute():
                data_path = config_path.parent / data_path

            f = safe_open(str(data_path), framework="pt")
            metadata = f.metadata()
            n = int(metadata["count"])
            prompts = json.loads(metadata["prompts"])

            self._shards.append(f)
            self._prompts.append(prompts)
            total += n
            self._cumulative.append(total)
            logger.info("Loaded %d precomputed samples from %s", n, data_path)

        self._len = total

    def _locate(self, idx: int) -> tuple[int, int]:
        ti = bisect.bisect_right(self._cumulative, idx)
        local = idx if ti == 0 else idx - self._cumulative[ti - 1]
        return ti, local

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        ti, row = self._locate(idx)
        f = self._shards[ti]
        prompt = self._prompts[ti][row]

        prompt_embeds = f.get_tensor(f"{row}.prompt_embeds")
        latents = f.get_tensor(f"{row}.latents")
        condition = f.get_tensor(f"{row}.condition")

        return {
            "index": idx,
            "prompt": prompt,
            "prompt_embeds": prompt_embeds,
            "video_latents": latents,
            "condition": condition,
        }
