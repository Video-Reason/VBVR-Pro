"""Helpers for localizing remote media paths before video/image decoding."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

_S3_PREFIX = "s3://"
logger = logging.getLogger(__name__)


def is_remote_path(path: str) -> bool:
    return path.startswith(_S3_PREFIX)


def resolve_media_path(path: str, root: str | Path | None = None) -> str:
    """Resolve local or S3 media paths without mangling S3 URI separators."""
    if path.startswith(_S3_PREFIX) or Path(path).is_absolute():
        return path
    if root:
        root_str = str(root)
        if root_str.startswith(_S3_PREFIX):
            return root_str.rstrip("/") + "/" + path.lstrip("/")
        return str(Path(root_str) / path)
    return path


def localize_media_path(path: str) -> str:
    """Return a local path for media decoders.

    Local paths are returned unchanged. S3 paths are downloaded into a cache
    directory through an operator-provided command. Configuration is
    environment-driven so credentials stay out of training configs:

    - WAN_TRAINER_REMOTE_DOWNLOADER: command invoked with URI and destination
    - WAN_TRAINER_REMOTE_CACHE_DIR: download cache root
    """
    if not is_remote_path(path):
        return path
    return str(_download_s3_to_cache(path))


def _env_seconds(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        seconds = float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.1fs", name, value, default)
        return default
    return max(0.0, seconds)


def _remote_cache_path(uri: str) -> Path:
    cache_dir = Path(os.environ.get("WAN_TRAINER_REMOTE_CACHE_DIR", "storage/remote_cache"))
    parsed = urlparse(uri)
    suffix = Path(parsed.path).suffix or ".bin"
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    return cache_dir / parsed.netloc / digest[:2] / f"{digest}{suffix}"


def _lock_owner(lock_dir: Path) -> str:
    owner_path = lock_dir / "owner.json"
    with suppress(OSError, json.JSONDecodeError):
        owner = json.loads(owner_path.read_text())
        host = owner.get("host", "?")
        pid = owner.get("pid", "?")
        thread = owner.get("thread", "?")
        created = owner.get("created_time")
        created_text = f" created={created:.0f}" if isinstance(created, int | float) else ""
        return f"host={host} pid={pid} thread={thread}{created_text}"
    return "owner=<unknown>"


def _write_lock_owner(lock_dir: Path, uri: str) -> None:
    owner = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "created_time": time.time(),
        "uri_sha256": hashlib.sha256(uri.encode("utf-8")).hexdigest(),
    }
    with suppress(OSError):
        (lock_dir / "owner.json").write_text(json.dumps(owner, sort_keys=True) + "\n")


def _tmp_download_path(local_path: Path) -> Path:
    host = re.sub(r"[^A-Za-z0-9_.-]+", "_", socket.gethostname())
    thread_id = threading.get_ident()
    return local_path.with_name(f".{local_path.name}.{host}.{os.getpid()}.{thread_id}.tmp")


def _acquire_download_lock(lock_dir: Path, local_path: Path, uri: str) -> bool:
    lock_timeout = _env_seconds("WAN_TRAINER_REMOTE_LOCK_TIMEOUT_SECONDS", 120.0)
    stale_seconds = _env_seconds("WAN_TRAINER_REMOTE_LOCK_STALE_SECONDS", 900.0)
    warn_seconds = _env_seconds("WAN_TRAINER_REMOTE_DOWNLOAD_WARN_SECONDS", 60.0)
    started = time.monotonic()
    last_warn = started

    while True:
        try:
            lock_dir.mkdir()
            _write_lock_owner(lock_dir, uri)
            waited = time.monotonic() - started
            if warn_seconds > 0 and waited >= warn_seconds:
                logger.warning("Waited %.1fs for remote cache lock %s (%s)", waited, lock_dir, uri)
            return True
        except FileExistsError:
            if local_path.exists() and local_path.stat().st_size > 0:
                return False

            now = time.monotonic()
            try:
                lock_age = max(0.0, time.time() - lock_dir.stat().st_mtime)
            except FileNotFoundError:
                continue

            if stale_seconds > 0 and lock_age >= stale_seconds:
                owner = _lock_owner(lock_dir)
                try:
                    shutil.rmtree(lock_dir)
                    logger.warning(
                        "Removed stale remote cache lock %s age=%.1fs %s (%s)",
                        lock_dir,
                        lock_age,
                        owner,
                        uri,
                    )
                    continue
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    logger.warning("Could not remove stale remote cache lock %s: %r", lock_dir, exc)

            waited = now - started
            if lock_timeout > 0 and waited >= lock_timeout:
                raise TimeoutError(
                    f"Timed out after {waited:.1f}s waiting for remote cache lock {lock_dir} "
                    f"age={lock_age:.1f}s {_lock_owner(lock_dir)} uri={uri}"
                ) from None
            if warn_seconds > 0 and now - last_warn >= warn_seconds:
                logger.warning(
                    "Still waiting for remote cache lock %s waited=%.1fs age=%.1fs %s (%s)",
                    lock_dir,
                    waited,
                    lock_age,
                    _lock_owner(lock_dir),
                    uri,
                )
                last_warn = now
            time.sleep(0.25)


def _download_s3_to_cache(uri: str) -> Path:
    local_path = _remote_cache_path(uri)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = local_path.with_suffix(local_path.suffix + ".lock")
    lock_acquired = _acquire_download_lock(lock_dir, local_path, uri)
    if not lock_acquired:
        return local_path

    try:
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        tmp_path = _tmp_download_path(local_path)
        if tmp_path.exists():
            tmp_path.unlink()
        started = time.monotonic()
        _run_remote_downloader(uri, tmp_path)
        elapsed = time.monotonic() - started
        warn_seconds = _env_seconds("WAN_TRAINER_REMOTE_DOWNLOAD_WARN_SECONDS", 60.0)
        if warn_seconds > 0 and elapsed >= warn_seconds:
            logger.warning("Downloaded remote media in %.1fs: %s -> %s", elapsed, uri, local_path)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Remote downloader produced an empty file for {uri}")
        tmp_path.replace(local_path)
        return local_path
    finally:
        with suppress(FileNotFoundError):
            shutil.rmtree(lock_dir)


def _run_remote_downloader(uri: str, destination: Path) -> None:
    command_raw = os.environ.get("WAN_TRAINER_REMOTE_DOWNLOADER", "").strip()
    if not command_raw:
        raise RuntimeError(
            "S3 media requires WAN_TRAINER_REMOTE_DOWNLOADER; the command is invoked as "
            "'<command> <s3-uri> <destination>'"
        )
    try:
        command = shlex.split(command_raw)
    except ValueError as exc:
        raise RuntimeError("WAN_TRAINER_REMOTE_DOWNLOADER is not valid shell-style argv") from exc
    if not command:
        raise RuntimeError("WAN_TRAINER_REMOTE_DOWNLOADER must contain an executable")
    try:
        subprocess.run([*command, uri, str(destination)], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Remote downloader executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Remote downloader exited with status {exc.returncode} for {uri}") from exc
