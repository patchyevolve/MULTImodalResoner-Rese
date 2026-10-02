"""Run identity captures (12 §6.4) — source snapshot + environment.

At run creation every run records:

  code/source.snapshot.tar.gz   full source-tree snapshot (the authority
                                if the working tree later disappears —
                                12 §6.4 "the experiment is still
                                recoverable")
  code/source.json              source tree hash + snapshot digest + git
                                provenance (commit / dirty / submodules)
  environment/fingerprint.json  python + platform identity (the blessed
                                v1 stand-in until the OCI image digest
                                lands — same mechanism, 12 §6.4)

The gate verifies these captures (steps 3 and 7): a missing capture, a
tampered snapshot, a changed source tree, or a moved environment are
FAILs — code/environment identity is frozen at creation (12 §13.3).

Excluded from the tree (mutable state, never code identity): run
folders, the content store, derived artifacts, VCS metadata, caches,
byte-compiled files, and any dataset whose machine-local path lives
INSIDE the workspace (data is dataset identity — gate step 4 — not
source identity).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import mlforge
from mlforge.errors import ValidationBlock
from mlforge.hashing import content_hash_bytes, file_hash
from mlforge.ingest.transforms import env_fingerprint
from mlforge.validation.gate import GateContext, Provider
from mlforge.validation.report import FAIL, PASS, Check

#: Mutable/non-code directories never part of source identity. These are
#: SYSTEM-GENERATED registries and runtime state — each is verified by its
#: own gate step (datasets → 4, model registry → 6, journal is the
#: authority 12 §16.1). Counting them would make every model publish,
#: `dataset add`, or supervisor tick "source drift" and block resume.
SOURCE_EXCLUDES = frozenset({
    "runs", "store", "artifacts", "environment", "code",
    ".git", "__pycache__", ".pytest_cache", ".mlforge",
    "dist", "build",
    "state", "projects", "models", "datasets", ".mlforge_probe",
})

#: Generated top-level files (same reasoning as the dirs above).
SOURCE_EXCLUDE_FILES = frozenset({"commands.jsonl", "datasets.yaml"})

SNAPSHOT_NAME = "source.snapshot.tar.gz"
SOURCE_META_NAME = "source.json"
ENV_META_NAME = "fingerprint.json"

SOURCE_SCHEMA = "mlforge.source_artifact.v1"
ENV_SCHEMA = "mlforge.environment.v1"


def _configured_dataset_paths(root: Path) -> set[Path]:
    """Machine-local dataset dirs inside the workspace — data identity
    is gate step 4's job; snapshotting it would be double-counting (and
    potentially enormous). Unreadable config ⇒ no exclusions (the gate's
    dataset step reports the config problem separately)."""
    try:
        from mlforge.ingest.config import load_paths
        values = load_paths(root)
    except Exception:
        return set()
    out: set[Path] = set()
    for raw in values.values():
        try:
            out.add(Path(raw).resolve())
        except OSError:
            continue
    return out


def iter_source_files(root: str | Path) -> list[Path]:
    """Every file that constitutes the experiment's source identity."""
    root = Path(root)
    data_dirs = _configured_dataset_paths(root)
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SOURCE_EXCLUDES)
        for name in sorted(filenames):
            if name.endswith((".pyc", ".pyo")) or name in SOURCE_EXCLUDE_FILES:
                continue
            p = Path(dirpath) / name
            if p.is_symlink():
                continue  # an identity we cannot walk deterministically
            try:
                rp = p.resolve()
            except OSError:
                continue
            if any(rp == d or d in rp.parents for d in data_dirs):
                continue
            files.append(p)
    return files


def source_tree_hash(root: str | Path) -> tuple[str, int]:
    """sha256 over (relative path, bytes) of the source tree (12 §6.4)."""
    root = Path(root)
    h = hashlib.sha256()
    count = 0
    for p in iter_source_files(root):
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(p.read_bytes())
        except OSError as exc:
            raise ValidationBlock(
                f"source file unreadable: {p}: {exc}",
                hint="source identity must hash cleanly (12 §6.4) — "
                     "fix permissions or remove the file",
            ) from exc
        count += 1
    return content_hash_bytes(h.digest()), count


def _git_info(root: Path) -> dict[str, Any] | None:
    """git provenance when the tree IS a repo; None otherwise (the
    snapshot is the authority — git is provenance, never a requirement)."""
    def _run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            timeout=5, check=False,
        )

    try:
        head = _run("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode != 0:
        return None
    status = _run("status", "--porcelain")
    subs = _run("submodule", "status")
    return {
        "commit": head.stdout.strip(),
        "dirty": (bool(status.stdout.strip())
                  if status.returncode == 0 else None),
        "submodules": (len([ln for ln in subs.stdout.splitlines() if ln.strip()])
                       if subs.returncode == 0 else None),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)  # atomic — readers never see a half-written capture


def capture_run_identity(root: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Record source + environment identity for a new run (12 §6.4)."""
    root = Path(root)
    run_dir = Path(run_dir)
    tree_hash, file_count = source_tree_hash(root)

    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    snapshot = code_dir / SNAPSHOT_NAME
    with tarfile.open(snapshot, "w:gz") as tar:
        for p in iter_source_files(root):
            tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
    source_meta = {
        "schema": SOURCE_SCHEMA,
        "source_tree_hash": tree_hash,
        "file_count": file_count,
        "snapshot": {
            "file": SNAPSHOT_NAME,
            "sha256": file_hash(snapshot),
            "bytes": snapshot.stat().st_size,
        },
        "git": _git_info(root),
        "mlforge": mlforge.__version__,
        "captured_ts": time.time(),
    }
    _write_json(code_dir / SOURCE_META_NAME, source_meta)

    env_meta = {
        "schema": ENV_SCHEMA,
        "fingerprint": env_fingerprint(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mlforge": mlforge.__version__,
        "captured_ts": time.time(),
    }
    _write_json(run_dir / "environment" / ENV_META_NAME, env_meta)
    return source_meta


# ---------------------------------------------------------------------------
# Gate providers — steps 3 (source_code) and 7 (environment)
# ---------------------------------------------------------------------------


def _check(step: int, label: str, verdict: str, detail: str,
           data: dict[str, Any] | None = None) -> Check:
    return Check(step, "source_code" if step == 3 else "environment",
                 label, verdict, detail, data or {})


def provide_source_code(root: str | Path) -> Provider:
    """Step 3 (12 §18): source artifact present + code hash match."""

    def _p(ctx: GateContext) -> Check:
        label = "Source artifact + code hash"
        code_dir = ctx.run_dir / "code"
        meta_path = code_dir / SOURCE_META_NAME
        if not meta_path.is_file():
            return _check(3, label, FAIL,
                          "no source capture (code/source.json) — runs record "
                          "source identity at creation (12 §6.4); this run "
                          "folder predates capture or is incomplete — recreate "
                          "the run")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _check(3, label, FAIL,
                          f"source capture unreadable ({exc}) — fail-closed")
        snap = code_dir / str(meta.get("snapshot", {}).get("file", SNAPSHOT_NAME))
        if not snap.is_file():
            return _check(3, label, FAIL,
                          "source snapshot missing "
                          f"({snap.name}) — the captured artifact is gone")
        recorded = meta.get("snapshot", {}).get("sha256")
        found = file_hash(snap)
        if recorded != found:
            return _check(3, label, FAIL,
                          f"snapshot digest mismatch — recorded {recorded}, "
                          f"file hashes to {found} (artifact tampered/corrupt)")
        try:
            live_hash, live_count = source_tree_hash(ctx.root)
        except ValidationBlock as exc:
            return _check(3, label, FAIL, f"source tree unreadable: {exc.message}")
        recorded_tree = meta.get("source_tree_hash")
        if live_hash != recorded_tree:
            return _check(
                3, label, FAIL,
                f"source tree changed since run creation — recorded "
                f"{recorded_tree}, live {live_hash} ({live_count} files); "
                "code identity is frozen (12 §13.3): the experiment's code "
                "must not drift under a run",
                {"recorded": recorded_tree, "live": live_hash},
            )
        git = meta.get("git")
        provenance = (f" · git {git['commit'][:12]}"
                      + (" (dirty)" if git.get("dirty") else "")
                      if isinstance(git, dict) and git.get("commit") else "")
        return _check(
            3, label, PASS,
            f"source tree verified ({live_count} files) + snapshot "
            f"{str(recorded)[:19]}…{provenance}",
            {"source_tree_hash": live_hash, "snapshot_sha256": recorded},
        )

    return _p


def provide_environment(root: str | Path) -> Provider:
    """Step 7 (12 §18): environment identity captured at creation matches
    the live environment (v1: python + platform fingerprint)."""

    def _p(ctx: GateContext) -> Check:
        label = "Environment identity captured"
        path = ctx.run_dir / "environment" / ENV_META_NAME
        if not path.is_file():
            return _check(7, label, FAIL,
                          "no environment capture (environment/fingerprint.json)"
                          " — runs record it at creation (12 §6.4); recreate "
                          "the run")
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _check(7, label, FAIL,
                          f"environment capture unreadable ({exc}) — fail-closed")
        recorded = meta.get("fingerprint")
        live = env_fingerprint()
        if not recorded:
            return _check(7, label, FAIL,
                          "environment capture has no fingerprint — fail-closed")
        if live != recorded:
            return _check(
                7, label, FAIL,
                f"environment changed since run creation — recorded "
                f"{str(recorded)[:19]}…, live {live[:19]}… (python/platform "
                "identity is part of the experiment, 12 §6.4)",
                {"recorded": recorded, "live": live},
            )
        return _check(
            7, label, PASS,
            f"environment verified — {meta.get('python', '?')} on "
            f"{meta.get('platform', '?')}",
            {"fingerprint": live},
        )

    return _p
