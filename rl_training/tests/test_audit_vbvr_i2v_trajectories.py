import hashlib
import json

import pytest

from src.cli.audit_vbvr_i2v_trajectories import audit, parse_args


def _make_complete_cell(tmp_path, *, noise_level: float = 0.7):
    model = tmp_path / "model"
    model.mkdir()
    eval_json = tmp_path / "eval.json"
    name = "In-Domain_50/task/00000"
    eval_json.write_text(json.dumps([{"name": name}]), encoding="utf-8")
    formal_root = tmp_path / "formal"
    formal = formal_root / "In-Domain_50/task/00000.mp4"
    formal.parent.mkdir(parents=True)
    formal.write_bytes(b"formal-video")
    digest = hashlib.sha256(formal.read_bytes()).hexdigest()
    output_root = tmp_path / "trajectories"
    sample = output_root / "In-Domain_50/task/00000"
    sample.mkdir(parents=True)
    (sample / "step_00.mp4").write_bytes(b"step-zero")
    (sample / "step_01.mp4").write_bytes(formal.read_bytes())
    (sample / "final_00.mp4").write_bytes(formal.read_bytes())
    (sample / "steps_grid.mp4").write_bytes(b"grid")
    (sample / "step_contact_sheet.jpg").write_bytes(b"sheet")
    manifest = {
        "sample_index": 0,
        "sample_name": name,
        "sampler": "flow_cps",
        "noise_scale": noise_level,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "seed": 0,
        "height": 512,
        "width": 512,
        "num_frames": 81,
        "fps": 16,
        "model_path": str(model.resolve()),
        "step_previews": [
            {
                "display_step": 1,
                "file_index": 0,
                "kind": "predicted_clean_x0",
                "output_sigma": 0.0,
            },
            {"display_step": 2, "file_index": 1, "kind": "final_latent", "output_sigma": 0.0},
        ],
        "formal_final_binding": {"source": str(formal.resolve()), "sha256": digest},
    }
    (sample / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cell = {
        "state": "complete",
        "model_path": str(model.resolve()),
        "eval_json": str(eval_json.resolve()),
        "formal_final_root": str(formal_root.resolve()),
        "sampler": "cps",
        "noise_level": noise_level,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "seed": 0,
        "height": 512,
        "width": 512,
        "num_frames": 81,
        "fps": 16,
        "sample_count": 1,
        "completed_count": 1,
    }
    (output_root / "cell_manifest.json").write_text(json.dumps(cell), encoding="utf-8")
    args = parse_args(
        [
            "--eval_json",
            str(eval_json),
            "--model_path",
            str(model),
            "--output_dir",
            str(output_root),
            "--formal_final_root",
            str(formal_root),
            "--sampler",
            "cps",
            "--noise_level",
            str(noise_level),
            "--num_inference_steps",
            "2",
        ]
    )
    return args, sample


def test_audit_complete_trajectory_cell(tmp_path):
    args, _ = _make_complete_cell(tmp_path)

    assert audit(args)["state"] == "complete"


def test_audit_rejects_wrong_cps_noise_level(tmp_path):
    args, sample = _make_complete_cell(tmp_path)
    manifest_path = sample / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["noise_scale"] = 0.3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="noise level"):
        audit(args)


def test_audit_can_publish_canonical_cell_manifest(tmp_path):
    args, sample = _make_complete_cell(tmp_path)
    cell_manifest = sample.parents[2] / "cell_manifest.json"
    cell_manifest.write_text(json.dumps({"state": "in_progress"}), encoding="utf-8")
    args.write_cell_manifest = True

    assert audit(args)["state"] == "complete"
    payload = json.loads(cell_manifest.read_text(encoding="utf-8"))
    assert payload["state"] == "complete"
    assert payload["completed_count"] == payload["sample_count"] == 1
