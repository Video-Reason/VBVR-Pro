import pytest
import torch
from safetensors.torch import save as st_save

from src.data.vbvr_latent_dataset import _decode_sample, _RankShardSplitter, _ShardSubsetSplitter


def test_rank_shard_splitter_keeps_normal_rank_stride():
    splitter = _RankShardSplitter(rank=2, world_size=4, num_shards=10)

    assert splitter.rank_shard_count == 2
    assert list(splitter(iter(range(10)))) == [2, 6]


def test_shard_subset_splitter_reads_group_shards():
    splitter = _ShardSubsetSplitter(offset=3, stride=4, num_shards=20)

    assert splitter.rank_shard_count == 5
    assert list(splitter(iter(range(20)))) == [3, 7, 11, 15, 19]


def test_decode_sample_uses_one_target_latent():
    target = torch.randn(4, 3, 2, 2)
    sample = {
        "safetensors": st_save(
            {
                "prompt_embeds": torch.randn(3, 8),
                "condition": torch.randn(4, 3, 2, 2),
                "latents": target,
            }
        )
    }

    decoded = _decode_sample(sample, max_text_len=5)

    assert decoded["prompt_embeds"].shape == (5, 8)
    assert torch.equal(decoded["video_latents"], target)


def test_decode_sample_rejects_numbered_target_latents():
    sample = {
        "safetensors": st_save(
            {
                "prompt_embeds": torch.randn(3, 8),
                "condition": torch.randn(4, 3, 2, 2),
                "latents_0": torch.randn(4, 3, 2, 2),
                "latents_1": torch.randn(4, 3, 2, 2),
            }
        )
    }

    with pytest.raises(KeyError, match="Numbered target latents"):
        _decode_sample(sample)
