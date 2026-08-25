import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.cli.render_vbvr_i2v_steps import (
    _formal_final_frames,
    _OdeStepRecorder,
    _sample_output_dir,
    _trajectory_validation_error,
    _write_cell_manifest,
    parse_args,
)


class _FakeScheduler:
    def __init__(self, *, predict_x0: bool):
        self.config = SimpleNamespace(predict_x0=predict_x0)
        self.sigmas = torch.tensor([0.75, 0.0])
        self.step_index = None
        self.original_calls = 0

    def _init_step_index(self, timestep):
        self.step_index = 0

    def convert_model_output(self, model_output, *, sample):
        return sample.float() - self.sigmas[self.step_index] * model_output.float()

    def step(self, model_output, timestep, sample, return_dict=False):
        self.original_calls += 1
        if self.step_index is None:
            self._init_step_index(timestep)
        self.step_index += 1
        return (sample - model_output,)


class _FakePipe:
    def __init__(self, *, predict_x0: bool):
        self.scheduler = _FakeScheduler(predict_x0=predict_x0)

    def prepare_latents(self):
        latents = torch.full((1, 1, 2, 1, 1), 9.0)
        condition = torch.tensor([[[[[3.0]], [[4.0]]]]])
        mask = torch.tensor([[[[[0.0]], [[1.0]]]]])
        return latents, condition, mask


@pytest.mark.parametrize("predict_x0", [False, True])
def test_ode_recorder_captures_clean_endpoint_and_pins_conditioned_frame(predict_x0: bool):
    pipe = _FakePipe(predict_x0=predict_x0)
    recorder = _OdeStepRecorder(pipe)
    recorder.install()
    try:
        pipe.prepare_latents()
        sample = torch.tensor([[[[[8.0]], [[8.0]]]]])
        velocity = torch.tensor([[[[[2.0]], [[4.0]]]]])
        output = pipe.scheduler.step(velocity, torch.tensor(750.0), sample, return_dict=False)
    finally:
        recorder.restore()

    assert pipe.scheduler.original_calls == 1
    assert torch.equal(output[0], sample - velocity)
    assert recorder.sigmas == [0.75]
    assert recorder.timesteps == [750.0]
    assert torch.equal(recorder.pred_x0[0], torch.tensor([[[[[3.0]], [[5.0]]]]]))


def test_render_cli_requires_exactly_cps_noise_level(tmp_path):
    common = [
        "--eval_json",
        str(tmp_path / "eval.json"),
        "--model_path",
        str(tmp_path / "model"),
        "--output_dir",
        str(tmp_path / "output"),
    ]

    with pytest.raises(SystemExit):
        parse_args([*common, "--sampler", "cps"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--sampler", "unipc", "--noise_level", "0.7"])
    assert parse_args([*common, "--sampler", "cps", "--noise_level", "0.3"]).noise_level == 0.3


def test_render_cli_all_samples_requires_formal_root(tmp_path):
    common = [
        "--eval_json",
        str(tmp_path / "eval.json"),
        "--model_path",
        str(tmp_path / "model"),
        "--output_dir",
        str(tmp_path / "output"),
        "--sampler",
        "euler",
    ]
    with pytest.raises(SystemExit):
        parse_args([*common, "--all_samples"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--all_samples", "--formal_final_root", "formal", "--sample_index", "0"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--limit", "2"])

    args = parse_args([*common, "--all_samples", "--formal_final_root", "formal", "--limit", "2"])
    assert args.all_samples
    assert args.sample_index is None
    assert args.limit == 2


def test_render_cli_validates_sample_sharding(tmp_path):
    common = [
        "--eval_json",
        str(tmp_path / "eval.json"),
        "--model_path",
        str(tmp_path / "model"),
        "--output_dir",
        str(tmp_path / "output"),
        "--sampler",
        "euler",
        "--all_samples",
        "--formal_final_root",
        str(tmp_path / "formal"),
    ]
    args = parse_args([*common, "--sample_shard_count", "2", "--sample_shard_index", "1"])
    assert (args.sample_shard_index, args.sample_shard_count) == (1, 2)

    with pytest.raises(SystemExit):
        parse_args([*common, "--sample_shard_count", "2", "--sample_shard_index", "2"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--eval_json",
                "eval.json",
                "--model_path",
                "model",
                "--output_dir",
                "output",
                "--sampler",
                "euler",
                "--sample_shard_count",
                "2",
            ]
        )


def test_sharded_cell_manifest_is_isolated_from_canonical_manifest(tmp_path):
    args = parse_args(
        [
            "--eval_json",
            str(tmp_path / "eval.json"),
            "--model_path",
            str(tmp_path / "model"),
            "--output_dir",
            str(tmp_path / "output"),
            "--sampler",
            "cps",
            "--noise_level",
            "0.7",
            "--all_samples",
            "--formal_final_root",
            str(tmp_path / "formal"),
            "--sample_shard_count",
            "2",
            "--sample_shard_index",
            "1",
        ]
    )
    _write_cell_manifest(
        args,
        output_root=tmp_path / "output",
        sample_count=5,
        selected_count=2,
        completed_count=1,
        initial_completed_count=0,
        started_at_unix=10.0,
    )

    path = tmp_path / "output/cell_manifest.shard-001-of-002.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["state"] == "in_progress"
    assert payload["global_sample_count"] == 5
    assert payload["completed_selected_count"] == 1
    assert not (tmp_path / "output/cell_manifest.json").exists()


def test_sample_output_dir_preserves_formal_hierarchy(tmp_path):
    assert _sample_output_dir(tmp_path, "In-Domain_50/task/00000") == tmp_path / "In-Domain_50/task/00000"


def test_trajectory_validation_binds_formal_final_by_digest(tmp_path):
    model = tmp_path / "model"
    eval_json = tmp_path / "eval.json"
    output = tmp_path / "trajectory"
    formal = tmp_path / "formal.mp4"
    model.mkdir()
    eval_json.write_text("[]", encoding="utf-8")
    output.mkdir()
    formal.write_bytes(b"formal-final-video")
    digest = hashlib.sha256(formal.read_bytes()).hexdigest()
    args = parse_args(
        [
            "--eval_json",
            str(eval_json),
            "--model_path",
            str(model),
            "--output_dir",
            str(output),
            "--sampler",
            "cps",
            "--noise_level",
            "0.7",
            "--formal_final_video",
            str(formal),
        ]
    )
    for index in range(30):
        (output / f"step_{index:02d}.mp4").write_bytes(formal.read_bytes() if index == 29 else b"step")
    (output / "final_00.mp4").write_bytes(formal.read_bytes())
    (output / "steps_grid.mp4").write_bytes(b"grid")
    (output / "step_contact_sheet.jpg").write_bytes(b"sheet")
    manifest = {
        "sample_index": 0,
        "sample_name": "sample/00000",
        "sampler": "flow_cps",
        "noise_scale": 0.7,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "seed": 0,
        "height": 512,
        "width": 512,
        "num_frames": 81,
        "fps": 16,
        "model_path": str(model.resolve()),
        "step_previews": [{} for _ in range(30)],
        "formal_final_binding": {"source": str(formal.resolve()), "sha256": digest},
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        _trajectory_validation_error(
            args,
            output_dir=output,
            formal_final_video=formal,
            sample_index=0,
            sample_name="sample/00000",
        )
        is None
    )
    (output / "step_29.mp4").write_bytes(b"not-formal")
    assert "digest mismatch" in _trajectory_validation_error(
        args,
        output_dir=output,
        formal_final_video=formal,
        sample_index=0,
        sample_name="sample/00000",
    )


def test_formal_final_frames_supports_decord2_numpy_api(tmp_path, monkeypatch):
    path = tmp_path / "formal.mp4"
    path.write_bytes(b"video")
    expected = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    class FakeFrames:
        def numpy(self):
            return expected

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_batch(self, _indices):
            return FakeFrames()

    monkeypatch.setattr("src.cli.render_vbvr_i2v_steps.eval_i2v._video_validation_error", lambda *_a, **_k: None)
    monkeypatch.setattr("src.cli.render_vbvr_i2v_steps.decord.VideoReader", FakeReader)
    args = SimpleNamespace(
        formal_final_video=str(path),
        width=4,
        height=4,
        num_frames=2,
        fps=16,
    )

    assert np.array_equal(_formal_final_frames(args), expected)
