from __future__ import annotations

import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.trainer.rewards.vbvr_rule as vbvr_rule_module
from src.eval.vbvr_run_evaluation_parallel import evalkit_source_sha256
from src.inference.outputs import uint8_from_decoded
from src.trainer.rewards.vbvr_rule import VBVRRuleReward


def _write_fake_evalkit(path: Path) -> None:
    path.mkdir()
    (path / "vbvr_bench").mkdir()
    (path / "run_evaluation.py").write_text(
        textwrap.dedent(
            """
            import json
            import os

            import cv2


            class NumpyEncoder(json.JSONEncoder):
                pass


            TASK_EVALUATOR_MAP = {"fake-task": object, "error-task": object}


            def collect_videos(model_path):
                return []


            def find_gt_info(task_name, video_idx, gt_base):
                return {}


            def evaluate_single_video(video_path, task_name, gt_info, device):
                if task_name == "error-task":
                    return {"score": 0.0, "error": "intentional scorer failure", "dimensions": {}}

                cap = cv2.VideoCapture(video_path)
                info = {
                    "opened": cap.isOpened(),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "cuda_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                    "cwd_is_evalkit": os.getcwd() == os.path.dirname(__file__),
                    "device": device,
                    "prompt": gt_info.get("prompt"),
                    "gt_path": gt_info.get("gt_path"),
                    "metafile_path": gt_info.get("metafile_path"),
                }
                cap.release()
                valid = (
                    info["opened"]
                    and info["width"] == 64
                    and info["height"] == 64
                    and info["frames"] == 9
                    and abs(info["fps"] - 9.0) < 1e-6
                    and info["cuda_hidden"]
                    and info["cwd_is_evalkit"]
                    and info["device"] == "cpu"
                    and info["prompt"] == "video prompt"
                    and os.path.isdir(info["gt_path"])
                    and isinstance(info["metafile_path"], list)
                    and os.path.isfile(info["metafile_path"][0])
                )
                return {
                    "score": 0.75 if valid else 0.0,
                    "error": None if valid else repr(info),
                    "dimensions": info,
                }


            def init_results(model_name, model_path):
                return {"samples": [], "summary": {}}


            def aggregate_score(results, sample):
                pass


            def finalize_summary(results):
                pass


            def print_results(results):
                pass
            """
        ).lstrip()
    )


def _config(evalkit: Path, tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        vbvr_reward_evalkit_dir=str(evalkit),
        vbvr_reward_evalkit_source_sha256=evalkit_source_sha256(evalkit),
        vbvr_reward_easyocr_module_path=None,
        vbvr_reward_device="cpu",
        vbvr_reward_fps=4,
        vbvr_reward_prepared_width=64,
        vbvr_reward_prepared_height=64,
        vbvr_reward_max_duration_seconds=1.0,
        vbvr_reward_prepare_crf=12,
        vbvr_reward_decode_batch_size=1,
        vbvr_reward_max_pending_jobs=2,
        vbvr_reward_cpu_workers=1,
        vbvr_reward_cpu_threads_per_worker=1,
        vbvr_reward_use_process_pool=True,
        vbvr_reward_task_specific_only=True,
        vbvr_reward_fail_on_error=True,
        vbvr_reward_tmp_dir=str(tmp_path / "reward-tmp"),
        vbvr_reward_keep_tmp=False,
        vbvr_reward_unsupported_score=0.0,
    )


def _trainer() -> SimpleNamespace:
    return SimpleNamespace(rank=0, tensor_parallel_enabled=False, tp_rank=0)


def test_reward_uses_canonical_decoder_to_uint8_conversion():
    decoded = torch.tensor([[[[[-1.0, -0.5, 0.0, 0.5, 1.0]]]]])

    assert np.array_equal(VBVRRuleReward._to_uint8_videos(decoded), uint8_from_decoded(decoded))


def test_reward_requires_explicit_external_evalkit_path():
    with pytest.raises(ValueError, match="explicit external EvalKit checkout"):
        vbvr_rule_module._resolve_evalkit_path(None)


def test_reward_rejects_unpinned_evalkit_source(tmp_path: Path):
    evalkit = tmp_path / "evalkit"
    _write_fake_evalkit(evalkit)
    cfg = _config(evalkit, tmp_path)
    cfg.vbvr_reward_evalkit_source_sha256 = "0" * 64

    with pytest.raises(RuntimeError, match="source fingerprint mismatch"):
        VBVRRuleReward(_trainer(), cfg)


def test_delayed_replay_expands_reward_queue_to_one_local_rollout_step(tmp_path: Path):
    evalkit = tmp_path / "evalkit"
    _write_fake_evalkit(evalkit)
    cfg = _config(evalkit, tmp_path)
    cfg.grpo_delayed_replay = True
    cfg.grpo_shared_prompt_batch = True
    cfg.batch_size = 8
    cfg.grpo_group_size = 32
    trainer = _trainer()
    trainer.world_size = 8
    trainer.dp_size = 8

    reward = VBVRRuleReward(trainer, cfg)
    try:
        assert reward._delayed_min_pending_jobs == 32
        assert reward._max_pending_jobs == 32
    finally:
        reward.close()


def test_reward_requires_evalkit_source_pin(tmp_path: Path):
    evalkit = tmp_path / "evalkit"
    _write_fake_evalkit(evalkit)
    cfg = _config(evalkit, tmp_path)
    cfg.vbvr_reward_evalkit_source_sha256 = None

    with pytest.raises(ValueError, match="requires vbvr_reward_evalkit_source_sha256"):
        VBVRRuleReward(_trainer(), cfg)


def test_reward_rejects_incomplete_easyocr_package_before_starting_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evalkit = tmp_path / "evalkit"
    _write_fake_evalkit(evalkit)
    (evalkit / "vbvr_bench" / "ocr_evaluator.py").write_text("easyocr.Reader\n")
    (evalkit / "easyocr_models").mkdir()
    cfg = _config(evalkit, tmp_path)

    incomplete_package = tmp_path / "site-packages" / "easyocr"
    incomplete_package.mkdir(parents=True)
    spec = SimpleNamespace(
        submodule_search_locations=[str(incomplete_package)],
        origin=str(incomplete_package / "__init__.py"),
    )
    monkeypatch.setattr(vbvr_rule_module.importlib.util, "find_spec", lambda name: spec if name == "easyocr" else None)

    with pytest.raises(FileNotFoundError, match=r"EasyOCR is installed incompletely.*en_char\.txt"):
        VBVRRuleReward(_trainer(), cfg)


@pytest.mark.parametrize("task_name, expected_error", [("fake-task", None), ("error-task", "intentional")])
def test_reward_uses_final_eval_contract_and_surfaces_errors(
    tmp_path: Path,
    task_name: str,
    expected_error: str | None,
):
    evalkit = tmp_path / "evalkit"
    _write_fake_evalkit(evalkit)
    source_dir = tmp_path / "gt-sample"
    source_dir.mkdir()
    gt_video = source_dir / "ground_truth.mp4"
    gt_first = source_dir / "first_frame.png"
    gt_final = source_dir / "final_frame.png"
    metadata = source_dir / "metadata.json"
    for path in (gt_video, gt_first, gt_final):
        path.write_bytes(b"fixture")
    metadata.write_text("{}")

    reward = VBVRRuleReward(_trainer(), _config(evalkit, tmp_path))
    frames = np.zeros((9, 32, 48, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 220
    try:
        kwargs = {
            "gt_video_path": str(gt_video),
            "gt_first_frame_path": str(gt_first),
            "gt_final_frame_path": str(gt_final),
            "gt_metadata_path": str(metadata),
            "gt_source_dir": str(source_dir),
        }
        if expected_error is None:
            (tmp_path / "score-ok").mkdir()
            score = reward._score_in_dir(
                tmp_path / "score-ok",
                task_name,
                "video prompt",
                frames,
                None,
                **kwargs,
            )
            assert score == 0.75
        else:
            (tmp_path / "score-error").mkdir()
            with pytest.raises(RuntimeError, match=expected_error):
                reward._score_in_dir(
                    tmp_path / "score-error",
                    task_name,
                    "video prompt",
                    frames,
                    None,
                    **kwargs,
                )
    finally:
        reward.close()


def test_reward_streams_first_decoded_batch_before_decoding_next(tmp_path: Path):
    evalkit = tmp_path / "evalkit"
    _write_fake_evalkit(evalkit)
    source_dir = tmp_path / "gt-sample"
    source_dir.mkdir()
    gt_video = source_dir / "ground_truth.mp4"
    gt_first = source_dir / "first_frame.png"
    gt_final = source_dir / "final_frame.png"
    metadata = source_dir / "metadata.json"
    for path in (gt_video, gt_first, gt_final):
        path.write_bytes(b"fixture")
    metadata.write_text("{}")

    score_started = threading.Event()
    release_score = threading.Event()

    class StreamingModel:
        def __init__(self):
            self.decode_calls = 0

        def decode_latents(self, latents):
            self.decode_calls += 1
            if self.decode_calls == 2:
                assert score_started.wait(timeout=5), "first decoded batch was not queued before the second decode"
            return torch.zeros(latents.shape[0], 3, 2, 2, 2)

    model = StreamingModel()
    trainer = _trainer()
    trainer.model = model
    cfg = _config(evalkit, tmp_path)
    cfg.vbvr_reward_use_process_pool = False
    reward = VBVRRuleReward(trainer, cfg)

    def fake_score_video_pair(*, sample_id: str, **_kwargs) -> float:
        score_started.set()
        assert release_score.wait(timeout=5)
        return 0.25 if sample_id.endswith("-0") else 0.75

    reward._score_video_pair = fake_score_video_pair
    meta = {
        "sample_task_name": ["fake-task", "fake-task"],
        "sample_prompt": ["video prompt", "video prompt"],
        "sample_id": ["0", "1"],
        "sample_gt_video_path": [str(gt_video), str(gt_video)],
        "sample_gt_first_frame": [str(gt_first), str(gt_first)],
        "sample_gt_final_frame": [str(gt_final), str(gt_final)],
        "sample_metadata_path": [str(metadata), str(metadata)],
        "sample_source_dir": [str(source_dir), str(source_dir)],
    }
    latents = torch.zeros(2, 1)
    try:
        pending = reward.submit(latents, latents, latents, latents, meta=meta)
        assert model.decode_calls == 2
        assert score_started.is_set()
        assert not pending.futures[0][1].done()

        release_score.set()
        assert torch.equal(pending.result(), torch.tensor([0.25, 0.75]))
    finally:
        release_score.set()
        reward.close()


def test_reward_submit_resolves_relative_gt_paths_before_spawned_scorer_chdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    evalkit = tmp_path / "evalkit"
    _write_fake_evalkit(evalkit)
    source_dir = tmp_path / "gt-sample"
    source_dir.mkdir()
    gt_video = source_dir / "ground_truth.mp4"
    gt_first = source_dir / "first_frame.png"
    gt_final = source_dir / "final_frame.png"
    metadata = source_dir / "metadata.json"
    for path in (gt_video, gt_first, gt_final):
        path.write_bytes(b"fixture")
    metadata.write_text("{}")

    class DecodeModel:
        @staticmethod
        def decode_latents(latents):
            return torch.zeros(latents.shape[0], 3, 9, 32, 48)

    trainer = _trainer()
    trainer.model = DecodeModel()
    reward = VBVRRuleReward(trainer, _config(evalkit, tmp_path))
    monkeypatch.chdir(tmp_path)
    meta = {
        "sample_task_name": ["fake-task"],
        "sample_prompt": ["video prompt"],
        "sample_id": ["0"],
        "sample_gt_video_path": [str(gt_video.relative_to(tmp_path))],
        "sample_gt_first_frame": [str(gt_first.relative_to(tmp_path))],
        "sample_gt_final_frame": [str(gt_final.relative_to(tmp_path))],
        "sample_metadata_path": [str(metadata.relative_to(tmp_path))],
        "sample_source_dir": [str(source_dir.relative_to(tmp_path))],
    }
    latents = torch.zeros(1, 1)
    try:
        pending = reward.submit(latents, latents, latents, latents, meta=meta)
        assert torch.equal(pending.result(), torch.tensor([0.75]))
    finally:
        reward.close()
