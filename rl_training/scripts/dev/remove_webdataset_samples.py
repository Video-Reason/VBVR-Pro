"""Remove selected WebDataset samples by key from shard tar files.

This script rewrites affected tar files instead of editing them in place.
Use --dry-run first, then pass --apply to replace each tar while keeping a
timestamped .bak backup beside it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def shard_name_for_key(key: str) -> str:
    try:
        shard_id = int(key) // 1000
    except ValueError as exc:
        raise ValueError(f"key must be numeric, got {key!r}") from exc
    return f"shard-{shard_id:06d}.tar"


def parse_target(raw: str) -> tuple[str, str]:
    """Return (shard_name, key).

    Accepted forms:
    - 0017195
    - shard-000017.tar:0017195
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("empty target")
    if ":" in raw:
        shard_name, key = raw.split(":", 1)
        shard_name = Path(shard_name).name
    else:
        key = raw
        shard_name = shard_name_for_key(key)
    if not key.isdigit():
        raise ValueError(f"key must be numeric, got {key!r}")
    return shard_name, key


def load_targets(args: argparse.Namespace) -> dict[str, set[str]]:
    raws: list[str] = list(args.key or [])
    if args.targets_file:
        for line in Path(args.targets_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raws.append(line)
    if not raws:
        raise ValueError("provide at least one --key or --targets-file entry")

    targets: dict[str, set[str]] = defaultdict(set)
    for raw in raws:
        shard_name, key = parse_target(raw)
        targets[shard_name].add(key)
    return targets


def member_key(member_name: str) -> str:
    return Path(member_name).name.split(".", 1)[0]


def read_json_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, Any]:
    f = tf.extractfile(member)
    if f is None:
        return {}
    try:
        return json.loads(f.read().decode("utf-8"))
    except Exception:
        return {}


def inspect_tar(tar_path: Path, keys: set[str]) -> tuple[list[tarfile.TarInfo], list[dict[str, Any]]]:
    found_members: list[tarfile.TarInfo] = []
    metas: list[dict[str, Any]] = []
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            key = member_key(member.name)
            if key not in keys:
                continue
            found_members.append(member)
            if member.name.endswith(".json"):
                meta = read_json_member(tf, member)
                metas.append({"key": key, "member": member.name, "meta": meta})
    return found_members, metas


def rewrite_tar(tar_path: Path, keys: set[str]) -> tuple[Path, int]:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = tar_path.with_name(f".{tar_path.name}.tmp.{os.getpid()}")
    backup_path = tar_path.with_suffix(tar_path.suffix + f".bak.{timestamp}")
    removed = 0

    with tarfile.open(tar_path, "r") as src, tarfile.open(tmp_path, "w") as dst:
        for member in src:
            if member_key(member.name) in keys:
                removed += 1
                continue
            f = src.extractfile(member) if member.isfile() else None
            try:
                dst.addfile(member, f)
            finally:
                if f is not None:
                    f.close()

    os.replace(tar_path, backup_path)
    try:
        os.replace(tmp_path, tar_path)
    except Exception:
        os.replace(backup_path, tar_path)
        raise
    return backup_path, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webdataset-dir", default="data/vbvr/latents/vbvr_384x384x81/webdataset/sft")
    parser.add_argument(
        "--key",
        action="append",
        help="Sample key to remove, e.g. 0017195 or shard-000017.tar:0017195. Repeat for multiple samples.",
    )
    parser.add_argument("--targets-file", help="Text file with one key or shard:key per line.")
    parser.add_argument("--apply", action="store_true", help="Actually rewrite tar files. Without this, dry-run only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.webdataset_dir)
    targets = load_targets(args)
    total_members = 0

    for shard_name, keys in sorted(targets.items()):
        tar_path = root / shard_name
        if not tar_path.exists():
            raise FileNotFoundError(tar_path)

        members, metas = inspect_tar(tar_path, keys)
        found_keys = {member_key(m.name) for m in members}
        missing = sorted(keys - found_keys)
        print(f"\n=== {tar_path} ===")
        print(f"target_keys={sorted(keys)} found_keys={sorted(found_keys)} missing={missing}")
        for item in metas:
            meta = item["meta"]
            prompt = " ".join(str(meta.get("prompt", "")).splitlines())
            print(
                json.dumps(
                    {
                        "key": item["key"],
                        "member": item["member"],
                        "tar": meta.get("tar"),
                        "index_in_tar": meta.get("index_in_tar"),
                        "seq_len": meta.get("seq_len"),
                        "prompt": prompt[:300],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        total_members += len(members)

        if args.apply:
            backup_path, removed = rewrite_tar(tar_path, keys)
            print(f"removed_members={removed} backup={backup_path}")
        else:
            print(f"dry_run_members_to_remove={len(members)}")

    print(f"\nTOTAL_MEMBERS_MATCHED={total_members}")
    if not args.apply:
        print("DRY_RUN_ONLY=1")


if __name__ == "__main__":
    main()
