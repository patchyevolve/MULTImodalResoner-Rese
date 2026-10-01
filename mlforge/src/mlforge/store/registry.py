"""Artifact registry + reachability-based GC — build step 2.

Normative: 12_training_system.md §6.1 (registry transactions, GC rules).

Registry transaction (crash-safe registration):

    1. verify blob exists in the CAS (identity claim checked, not assumed)
    2. write entry to temp file
    3. fsync temp
    4. atomic rename into registry index      ← THE commit
    5. any crash before (4) leaves only a .tmp → `recover()` sweeps it

References (what makes an artifact *reachable*) live with the run that
uses them — `runs/<id>/refs.json` — so GC roots are derived from real
run folders, never from a cached refcount.

GC rules (hard, from the spec):
  * never GC by age or size alone — unreachability is the ONLY criterion
  * grace period: unreachable blobs younger than `grace_seconds` are kept
  * active run leases block GC outright (never GC during training)
  * double-check: re-scan reachability immediately before deletion
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, PreconditionFailed, ValidationBlock
from mlforge.store.content_store import ContentStore


@dataclass(frozen=True)
class GCReport:
    dry_run: bool
    reachable: int
    swept: list[str] = field(default_factory=list)
    kept_grace: list[str] = field(default_factory=list)
    skipped_lease: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "reachable": self.reachable,
            "swept": self.swept,
            "kept_grace": self.kept_grace,
            "skipped_lease": self.skipped_lease,
        }


class ArtifactRegistry:
    def __init__(self, workspace_root: str | Path, store: ContentStore):
        self.workspace = Path(workspace_root)
        self.store = store
        self.index_dir = self.workspace / "artifact_registry" / "index"

    # -- registration (transactional) ------------------------------------

    def entry_path(self, content_hash: str) -> Path:
        safe = content_hash.replace(":", "_")
        return self.index_dir / f"{safe}.json"

    def register(
        self,
        content_hash: str,
        *,
        kind: str,
        size: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        """Commit a registry entry. Fails if the blob is not in the store —
        registering a claim that was never verified is forbidden (§6.1)."""
        if not self.store.contains(content_hash):
            raise NotFound(
                f"cannot register {content_hash}: blob not in store",
                hint="put the artifact into the ContentStore first",
            )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "hash": content_hash,
            "kind": kind,
            "size": size if size is not None else self.store.size(content_hash),
            "registered_ts": time.time(),
            "meta": meta or {},
            "committed": True,  # written before rename — the rename is the commit
        }
        tmp = self.index_dir / f".{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(entry, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.entry_path(content_hash))
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return self.entry_path(content_hash)

    def get_entry(self, content_hash: str) -> dict[str, Any] | None:
        p = self.entry_path(content_hash)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def is_registered(self, content_hash: str) -> bool:
        return self.get_entry(content_hash) is not None

    def recover(self) -> dict[str, list[str]]:
        """Startup recovery scan (§6.1):
        - registry .tmp orphans (crash before commit) → deleted (uncommitted)
        - staging .tmp orphans in the store → deleted
        - blobs without a committed entry → registered from the blob
          (registration completed except for the index write)"""
        swept_tmp: list[str] = []
        if self.index_dir.is_dir():
            for tmp in list(self.index_dir.glob(".*.tmp")):
                tmp.unlink()
                swept_tmp.append(tmp.name)
        swept_staging = self.store.recover_staging()
        completed: list[str] = []
        for h in self.store.list_hashes():
            if not self.is_registered(h):
                self.register(h, kind="recovered")
                completed.append(h)
        return {
            "registry_tmp_swept": swept_tmp,
            "staging_swept": swept_staging,
            "entries_completed": completed,
        }

    # -- references (GC roots live with the runs) ------------------------

    def refs_path(self, run_id: str) -> Path:
        return self.workspace / "runs" / run_id / "refs.json"

    def link(self, run_id: str, *content_hashes: str) -> None:
        """A run references artifacts (12 §5: "the run folder contains
        references"). Atomic read-modify-write; refs are a set."""
        if not (self.workspace / "runs" / run_id).is_dir():
            raise NotFound(f"run {run_id!r} not found")
        p = self.refs_path(run_id)
        current: set[str] = set()
        if p.is_file():
            current = set(json.loads(p.read_text(encoding="utf-8")).get("refs", []))
        current.update(content_hashes)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"refs": sorted(current)}, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def unlink(self, run_id: str, content_hash: str) -> None:
        p = self.refs_path(run_id)
        if not p.is_file():
            return
        refs = set(json.loads(p.read_text(encoding="utf-8")).get("refs", []))
        refs.discard(content_hash)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"refs": sorted(refs)}, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def all_run_refs(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        runs_dir = self.workspace / "runs"
        if not runs_dir.is_dir():
            return out
        for run in runs_dir.iterdir():
            p = run / "refs.json"
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue  # corrupt refs of one run must not crash GC
                out[run.name] = set(data.get("refs", []))
        return out

    def reachable(self, extra_roots: set[str] | None = None) -> set[str]:
        reachable: set[str] = set(extra_roots or set())
        for refs in self.all_run_refs().values():
            reachable |= refs
        return reachable

    def active_leases(self) -> list[str]:
        """Runs currently holding a run lease (12 §23) — GC must not run."""
        held = []
        runs_dir = self.workspace / "runs"
        if runs_dir.is_dir():
            for run in runs_dir.iterdir():
                if (run / ".lease").exists():
                    held.append(run.name)
        return held

    # -- GC (reachability only) ------------------------------------------

    def gc(
        self,
        *,
        grace_seconds: float = 7 * 24 * 3600,
        dry_run: bool = True,
        extra_roots: set[str] | None = None,
    ) -> GCReport:
        """Reachability-based collection. Never age/size based (§6.1).

        Default is dry-run: deletion must be an explicit user decision
        (`mlforge store gc --execute`), never automatic during training."""
        held = self.active_leases()
        if held and not dry_run:
            raise PreconditionFailed(
                f"GC refused: active run lease(s): {', '.join(held)}",
                hint="stop the run(s) first — never GC during training",
            )
        reach = self.reachable(extra_roots)
        now = time.time()
        swept: list[str] = []
        kept_grace: list[str] = []
        for h in self.store.list_hashes():
            if h in reach:
                continue
            blob = self.store.path_for(h)
            if now - blob.stat().st_mtime < grace_seconds:
                kept_grace.append(h)
                continue
            # Re-verify reachability immediately before deletion (TOCTOU §23.3)
            if h in self.reachable(extra_roots):
                reach.add(h)
                continue
            if dry_run:
                swept.append(h)
                continue
            self.store.delete(h)
            entry = self.entry_path(h)
            if entry.is_file():
                entry.unlink()
            swept.append(h)
        return GCReport(
            dry_run=dry_run,
            reachable=len(reach),
            swept=swept,
            kept_grace=kept_grace,
            skipped_lease=held,
        )


__all__ = ["ArtifactRegistry", "GCReport", "ContentStore", "ValidationBlock"]
