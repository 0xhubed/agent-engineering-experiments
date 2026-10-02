"""Everything needed to say exactly what produced a set of results."""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_VOLATILE = ("created_at", "hostname")


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def build_manifest(config: dict, *, models: dict[str, str], extra: dict | None = None,
                   repo_dir: str | Path = ".") -> dict:
    repo = Path(repo_dir)
    sha = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain") if sha else None
    return {
        "git_sha": sha or "unknown",
        "git_dirty": bool(status),
        "config_hash": _sha256(canonical_json(config)),
        "config": config,
        "models": dict(models),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extra": extra or {},
    }


def manifest_hash(manifest: dict) -> str:
    stable = {k: v for k, v in manifest.items() if k not in _VOLATILE}
    return _sha256(canonical_json(stable))
