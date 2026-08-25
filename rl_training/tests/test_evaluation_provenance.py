from pathlib import Path

from src.eval.evaluation_provenance import (
    build_manifest,
    fingerprint_tree,
    main,
    manifest_matches,
    refresh_manifest_outputs,
    verify_recorded_manifest,
    write_manifest,
)


def test_manifest_detects_value_file_and_tree_changes(tmp_path: Path):
    source_file = tmp_path / "input.json"
    source_tree = tmp_path / "model"
    source_file.write_text("one")
    source_tree.mkdir()
    (source_tree / "weight.bin").write_bytes(b"weights")
    manifest_path = tmp_path / "provenance.json"

    expected = build_manifest(
        stage="generation",
        values={"state": "complete", "seed": "0"},
        files={"eval_json": str(source_file)},
        trees={"model": str(source_tree)},
    )
    write_manifest(manifest_path, expected)
    assert manifest_matches(manifest_path, expected) == (True, "")

    source_file.write_text("two")
    changed_file = build_manifest(
        stage="generation",
        values={"state": "complete", "seed": "0"},
        files={"eval_json": str(source_file)},
        trees={"model": str(source_tree)},
    )
    assert not manifest_matches(manifest_path, changed_file)[0]

    source_file.write_text("one")
    (source_tree / "weight.bin").write_bytes(b"different")
    changed_tree = build_manifest(
        stage="generation",
        values={"state": "complete", "seed": "0"},
        files={"eval_json": str(source_file)},
        trees={"model": str(source_tree)},
    )
    assert not manifest_matches(manifest_path, changed_tree)[0]


def test_tree_fingerprint_is_stable_and_tracks_symlinks(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "target-a").write_text("same")
    (tree / "target-b").write_text("same")
    (tree / "link").symlink_to("target-a")

    first = fingerprint_tree(tree)
    assert fingerprint_tree(tree) == first

    (tree / "link").unlink()
    (tree / "link").symlink_to("target-b")
    assert fingerprint_tree(tree) != first


def test_cli_check_and_atomic_write(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("content")
    manifest = tmp_path / "state/provenance.json"
    common = [
        "--manifest",
        str(manifest),
        "--stage",
        "preparation",
        "--value",
        "state=complete",
        "--file",
        f"source={source}",
    ]

    assert main(["write", *common]) == 0
    assert main(["check", *common]) == 0
    assert not list(manifest.parent.glob(f".{manifest.name}.tmp-*"))

    source.write_text("changed")
    assert main(["check", *common, "--quiet"]) == 1


def test_promote_rechecks_inputs_and_fingerprints_outputs(tmp_path: Path):
    source = tmp_path / "source.txt"
    output_dir = tmp_path / "videos"
    output = output_dir / "sample.mp4"
    source.write_text("input")
    output_dir.mkdir()
    output.write_bytes(b"video-one")
    manifest = tmp_path / "provenance.json"
    common = [
        "--manifest",
        str(manifest),
        "--stage",
        "generation",
        "--file",
        f"source={source}",
    ]

    assert main(["write", *common, "--value", "state=in_progress_resume"]) == 0
    source.write_text("changed")
    assert (
        main(
            [
                "promote",
                *common,
                "--value",
                "state=in_progress_resume",
                "--media-tree",
                f"videos={output_dir}",
                "--quiet",
            ]
        )
        == 1
    )

    source.write_text("input")
    assert (
        main(
            [
                "promote",
                *common,
                "--value",
                "state=in_progress_resume",
                "--media-tree",
                f"videos={output_dir}",
            ]
        )
        == 0
    )
    complete = [
        "check",
        *common,
        "--value",
        "state=complete",
        "--media-tree",
        f"videos={output_dir}",
    ]
    assert main(complete) == 0
    output.write_bytes(b"video-two")
    assert main([*complete, "--quiet"]) == 1


def test_verify_recorded_manifest_recomputes_outputs(tmp_path: Path):
    source = tmp_path / "source.txt"
    output = tmp_path / "result.json"
    source.write_text("input")
    output.write_text('{"score": 1}')
    manifest_path = tmp_path / "provenance.json"
    manifest = build_manifest(
        stage="vbvr-pro-score",
        values={"state": "complete"},
        files={"source": str(source)},
        trees={},
        output_files={"result": str(output)},
    )
    write_manifest(manifest_path, manifest)

    assert verify_recorded_manifest(
        manifest_path,
        expected_stage="vbvr-pro-score",
        require_complete=True,
    ) == (True, "")
    assert not verify_recorded_manifest(manifest_path, expected_stage="other-stage")[0]

    output.write_text('{"score": 0}')
    matches, detail = verify_recorded_manifest(
        manifest_path,
        expected_stage="vbvr-pro-score",
        require_complete=True,
        sections=("output_files",),
    )
    assert not matches
    assert "output_files.result" in detail


def test_refresh_outputs_requires_exact_complete_inputs(tmp_path: Path):
    source = tmp_path / "source.txt"
    output_dir = tmp_path / "model"
    output = output_dir / "weights.bin"
    source.write_text("input")
    output_dir.mkdir()
    output.write_bytes(b"weights-one")
    manifest_path = tmp_path / "provenance.json"

    recorded = build_manifest(
        stage="conversion",
        values={"state": "complete", "dtype": "bf16"},
        files={"source": str(source)},
        trees={},
        output_trees={"model": str(output_dir)},
    )
    write_manifest(manifest_path, recorded)

    output.write_bytes(b"weights-two")
    expected = build_manifest(
        stage="conversion",
        values={"state": "complete", "dtype": "bf16"},
        files={"source": str(source)},
        trees={},
        output_trees={"model": str(output_dir)},
    )
    assert refresh_manifest_outputs(manifest_path, expected) == (True, "")
    assert manifest_matches(manifest_path, expected) == (True, "")

    source.write_text("different")
    changed_inputs = build_manifest(
        stage="conversion",
        values={"state": "complete", "dtype": "bf16"},
        files={"source": str(source)},
        trees={},
        output_trees={"model": str(output_dir)},
    )
    assert not refresh_manifest_outputs(manifest_path, changed_inputs)[0]


def test_cli_refresh_outputs_rejects_non_complete_state(tmp_path: Path):
    source = tmp_path / "source.txt"
    output_dir = tmp_path / "model"
    source.write_text("input")
    output_dir.mkdir()
    (output_dir / "weights.bin").write_bytes(b"weights")
    manifest = tmp_path / "provenance.json"
    common = [
        "--manifest",
        str(manifest),
        "--stage",
        "conversion",
        "--file",
        f"source={source}",
        "--output-tree",
        f"model={output_dir}",
    ]

    assert main(["write", *common, "--value", "state=in_progress_resume"]) == 0
    assert main(["refresh-outputs", *common, "--value", "state=in_progress_resume", "--quiet"]) == 2
