from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from src.cli.probe_vlm_service import _percentile, _task_prompt_payload, main
from src.trainer.rewards.vbvr_vlm import task_vlm_judge_output_regex


def test_percentile_interpolates_sorted_values():
    values = [4.0, 1.0, 3.0, 2.0]
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 0.5) == 2.5
    assert _percentile(values, 0.95) == pytest.approx(3.85)
    assert _percentile(values, 1.0) == 4.0


def test_task_prompt_payload_matches_production_video_and_regex_contract(tmp_path):
    payload, task_prompt = _task_prompt_payload(
        model="qwen3.6-27b",
        task_name="G-21_multiple_occlusions_vertical_data-generator",
        frame_count=6,
        image_size=384,
        video_fps=16,
        video_num_frames=32,
    )

    assert payload["model"] == "qwen3.6-27b"
    assert payload["max_tokens"] == 1024
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["structured_outputs"] == {"regex": task_vlm_judge_output_regex(task_prompt)}
    assert payload["media_io_kwargs"] == {"video": {"video_backend": "opencv", "num_frames": 32, "fps": -1}}
    assert payload["mm_processor_kwargs"] == {"do_sample_frames": False}

    content = payload["messages"][0]["content"]
    image_urls = [block["image_url"]["url"] for block in content if block["type"] == "image_url"]
    assert len(image_urls) == 1
    for data_url in image_urls:
        prefix, encoded = data_url.split(",", maxsplit=1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            assert image.size == (384, 384)

    video_urls = [block["video_url"]["url"] for block in content if block["type"] == "video_url"]
    assert len(video_urls) == 1
    prefix, encoded = video_urls[0].split(",", maxsplit=1)
    assert prefix == "data:video/mp4;base64"
    video_path = tmp_path / "probe.mp4"
    video_path.write_bytes(base64.b64decode(encoded))

    import imageio.v3 as iio

    assert iio.imread(video_path).shape == (6, 384, 384, 3)
    assert iio.immeta(video_path)["fps"] == pytest.approx(16.0)


@pytest.mark.parametrize(
    "args",
    [
        ["--benchmark-requests", "-1"],
        ["--benchmark-concurrency", "0"],
        ["--benchmark-warmup", "-1"],
    ],
)
def test_probe_rejects_invalid_benchmark_arguments_before_network(args: list[str]):
    assert main(args) == 2
