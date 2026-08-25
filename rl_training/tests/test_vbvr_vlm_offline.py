from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from src.eval.vbvr_vlm_offline import (
    JudgeOutcome,
    OfflineJudgeConfig,
    OfflineTaskVLMJudge,
    _cell_attributes,
    aggregate_complete_cells,
    cell_is_complete,
    load_eval_cell,
    score_eval_cell,
)
from src.eval.vbvr_vlm_protocol import (
    TASK_VLM_JUDGE_OUTPUT_REMINDER,
    load_pinned_eval_prompts,
    task_vlm_judge_output_regex,
)

G21_TASK = "G-21_multiple_occlusions_vertical_data-generator"
G15_TASK = "G-15_grid_avoid_obstacles_data-generator"
EVAL_PROMPTS = load_pinned_eval_prompts()

G21_RESPONSE = """mask_path_vadility_score: 70
occlusion_correctness_score: 80
elements_preservation_score: 90
mask_path_vadility_weight: 40
occlusion_correctness_weight: 35
elements_preservation_weight: 25
total_score: 79.5
reason: Mostly correct."""

G15_RESPONSE = """proximity_score: 80
coverage_score: 70
continuity_factor_score: 60
obstacle_multiplier_score: 90
bg_preservation_score: 100
proximity_weight: 30
coverage_weight: 25
continuity_factor_weight: 20
obstacle_multiplier_weight: 20
bg_preservation_weight: 5
total_score: 80
reason: The path is mostly correct."""


@pytest.mark.parametrize(
    ("name", "sampler"),
    [("unipc", "UniPC ODE"), ("euler", "Euler ODE"), ("cps-noise-0.7", "CPS 0.7")],
)
def test_generic_release_cell_names_have_sampler_labels(name: str, sampler: str) -> None:
    assert _cell_attributes(name)["sampler"] == sampler


def _sample_tree(tmp_path: Path):
    cell_dir = tmp_path / "dancegrpo_vbvr_pro_5b_checkpoint-100-cps-noise-0.7"
    video_root = cell_dir / "generated_512x512x81"
    rows = []
    for domain_dir, domain, task, video_idx in (
        ("In-Domain_50", "In_Domain", G21_TASK, "00000"),
        ("Out-of-Domain_50", "Out_of_Domain", G15_TASK, "00001"),
    ):
        first_frame = tmp_path / "gt" / domain_dir / task / video_idx / "first_frame.png"
        first_frame.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 16), color="white").save(first_frame)
        name = f"{domain_dir}/{task}/{video_idx}"
        video_path = video_root / f"{name}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"native-video-{video_idx}".encode())
        rows.append(
            {
                "name": name,
                "image": str(first_frame),
                "prompt": "sample prompt",
                "task_name": task,
                "video_idx": video_idx,
                "domain": domain,
            }
        )
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "eval_samples.json").write_text(json.dumps(rows))
    provenance = {
        "media_trees": {
            "generated_videos": {
                "path": str(video_root),
                "sha256": "a" * 64,
                "entries": 2,
            }
        },
        "values": {"state": "complete"},
    }
    (cell_dir / "generation-provenance.json").write_text(json.dumps(provenance))
    return load_eval_cell(cell_dir, expected_samples=2)


def test_offline_judge_sends_native_mp4_with_exact_training_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cell = _sample_tree(tmp_path)
    sample = cell.samples[0]
    config = OfflineJudgeConfig(max_retries=0, retry_backoff_seconds=0)
    judge = OfflineTaskVLMJudge(config)
    captured = {}

    def fake_request(method: str, path: str, payload=None):
        captured.update({"method": method, "path": path, "payload": payload})
        return {"choices": [{"message": {"content": G21_RESPONSE}}]}, 1

    monkeypatch.setattr(judge, "_request_json", fake_request)
    outcome = judge.score(sample)

    assert outcome.error is None
    assert outcome.score == pytest.approx(0.795)
    payload = captured["payload"]
    assert payload["structured_outputs"] == {"regex": task_vlm_judge_output_regex(EVAL_PROMPTS[G21_TASK])}
    assert payload["media_io_kwargs"] == {"video": {"video_backend": "opencv", "num_frames": 32, "fps": -1}}
    assert payload["mm_processor_kwargs"] == {"do_sample_frames": False}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": EVAL_PROMPTS[G21_TASK].strip()}
    assert content[3]["text"] == "Generated video (81 source frames encoded at 16 FPS, chronological order):"
    assert content[-1] == {"type": "text", "text": TASK_VLM_JUDGE_OUTPUT_REMINDER}
    assert sum(block["type"] == "image_url" for block in content) == 1
    assert sum(block["type"] == "video_url" for block in content) == 1
    encoded_video = content[4]["video_url"]["url"].split(",", maxsplit=1)[1]
    assert base64.b64decode(encoded_video) == sample.video_path.read_bytes()


def test_offline_judge_repairs_semantically_invalid_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sample = _sample_tree(tmp_path).samples[1]
    invalid = G15_RESPONSE.replace("bg_preservation_weight: 5", "bg_preservation_weight: 15")
    responses = iter((invalid, G15_RESPONSE))
    payloads = []
    judge = OfflineTaskVLMJudge(OfflineJudgeConfig(max_retries=1, retry_backoff_seconds=0))

    def fake_request(method: str, path: str, payload=None):
        del method, path
        payloads.append(payload)
        return {"choices": [{"message": {"content": next(responses)}}]}, 1

    monkeypatch.setattr(judge, "_request_json", fake_request)
    outcome = judge.score(sample)

    assert outcome.error is None
    assert outcome.score == pytest.approx(0.8)
    assert outcome.semantic_attempts == 2
    assert outcome.request_attempts == 2
    assert payloads[1]["messages"][-2] == {"role": "assistant", "content": invalid}
    assert "Validation error:" in payloads[1]["messages"][-1]["content"]


def test_cell_scoring_is_resumable_and_writes_strict_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cell = _sample_tree(tmp_path)
    output_root = tmp_path / "scores"
    config = OfflineJudgeConfig()
    judge = OfflineTaskVLMJudge(config)

    def fake_score(sample):
        return JudgeOutcome(
            score=0.75 if sample.domain == "In_Domain" else 0.25,
            reasoning="ok",
            response="raw",
            error=None,
            request_attempts=1,
            semantic_attempts=1,
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(judge, "score", fake_score)
    summary = score_eval_cell(
        cell,
        output_root=output_root,
        judge=judge,
        concurrency=2,
        rank=0,
        progress_interval_seconds=1,
        fsync_every=1,
    )
    assert summary["state"] == "complete"
    assert summary["summary"]["overall"]["mean_score"] == pytest.approx(0.5)
    assert cell_is_complete(cell, output_root, config)

    def unexpected_score(sample):
        raise AssertionError(f"resume should not re-score {sample.name}")

    monkeypatch.setattr(judge, "score", unexpected_score)
    resumed = score_eval_cell(
        cell,
        output_root=output_root,
        judge=judge,
        concurrency=2,
        rank=0,
        progress_interval_seconds=1,
        fsync_every=1,
    )
    assert resumed["state"] == "complete"
    assert len((output_root / cell.name / "samples.jsonl").read_text().splitlines()) == 2

    aggregate = aggregate_complete_cells([cell], output_root=output_root, config=config)
    assert aggregate["num_cells"] == 1
    assert aggregate["num_sample_judgments"] == 2
    assert aggregate["mean_over_all_judgments"] == pytest.approx(0.5)
    assert (output_root / "summary.csv").is_file()
    assert (output_root / "summary.md").is_file()


def test_error_records_are_retried_and_not_promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cell = load_eval_cell(_sample_tree(tmp_path).source_dir, expected_samples=2, sample_limit=1)
    output_root = tmp_path / "scores"
    config = OfflineJudgeConfig()
    judge = OfflineTaskVLMJudge(config)
    error = JudgeOutcome(0.0, "", None, "RuntimeError: unavailable", 3, 0, 0.1)
    monkeypatch.setattr(judge, "score", lambda sample: error)
    first = score_eval_cell(
        cell,
        output_root=output_root,
        judge=judge,
        concurrency=1,
        rank=0,
        progress_interval_seconds=1,
        fsync_every=1,
    )
    assert first["state"] == "incomplete"
    assert not cell_is_complete(cell, output_root, config)

    success = JudgeOutcome(0.9, "fixed", "raw", None, 1, 1, 0.1)
    monkeypatch.setattr(judge, "score", lambda sample: success)
    second = score_eval_cell(
        cell,
        output_root=output_root,
        judge=judge,
        concurrency=1,
        rank=0,
        progress_interval_seconds=1,
        fsync_every=1,
    )
    assert second["state"] == "complete"
    assert second["summary"]["overall"]["mean_score"] == pytest.approx(0.9)
    assert len((output_root / cell.name / "samples.jsonl").read_text().splitlines()) == 2


def test_judge_contract_never_persists_api_key():
    config = OfflineJudgeConfig(api_key="secret")
    assert "api_key" not in config.contract()
    assert "secret" not in json.dumps(config.contract())
