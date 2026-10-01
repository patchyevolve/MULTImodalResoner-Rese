"""Transactional checkpoint store — 12_training_system.md §11.

Write protocol (§11.1, in order — never `torch.save` directly):

    write staging payload files → calculate hashes → fsync
        → atomic rename staging → final directory
        → write manifest.json → fsync → write commit marker

Power loss at any point before the commit marker ⇒ the checkpoint does
not exist for recovery purposes (§11.2 rule 1).

Recovery selection (§11.2) is a PREDICATE OVER IDENTITIES, never ordinal
arithmetic ("resume = N − 1" is the buggy model):

    candidates = committed AND manifest verifies AND all file hashes verify
    resume_point = argmax (global_step, ordinal) over candidates
                 = newest VALID, not newest ATTEMPTED

Every skipped candidate carries an explicit reason (NO COMMIT MARKER /
MANIFEST HASH MISMATCH / FILE HASH MISMATCH / ...) — reported, never
silently dropped (§11.2 rule 3–4). If nothing is valid → NO VALID
CHECKPOINT (never silent restart from step 0).

Checkpoint completeness (§11.4): the manifest records the component set
(model, optimizer, lr_scheduler, rng hierarchy, sampler, ...); resume
verification checks the required components are present — "same weights +
different future data" is not a continuation.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, ValidationBlock
from mlforge.hashing import content_hash_bytes, sha256_hex

#: §11.4 complete checkpoint contents (component names recorded in the
#: manifest; the runtime's trainer owns the actual state for each).
REQUIRED_COMPONENTS: tuple[str, ...] = (
    "model",
    "optimizer",
    "lr_scheduler",
    "amp_scaler",
    "ema",
    "global_step",
    "epoch",
    "batch_position",
    "sampler_state",
    "dataloader_state",
    "distributed_state",
    "grad_accum_state",
    "early_stopping_state",
    "best_model_state",
    "rng_hierarchy",
)

COMMIT_MARKER = "COMMIT"
MANIFEST_NAME = "manifest.json"
_CKPT_DIR_RE = re.compile(r"^ckpt-(\d{6})$")
_STAGING_SUFFIX = ".staging"


@dataclass(frozen=True)
class Candidate:
    ordinal: int
    path: Path
    valid: bool
    reason: str
    global_step: int | None = None
    manifest: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "valid": self.valid,
            "reason": self.reason,
            "global_step": self.global_step,
        }


@dataclass(frozen=True)
class Selection:
    """Result of the newest-valid predicate (§11.2)."""

    selected: Candidate | None
    skips: tuple[Candidate, ...] = ()
    attempted_newest: int | None = None

    @property
    def resume_ordinal(self) -> int | None:
        return self.selected.ordinal if self.selected else None

    @property
    def resume_point(self) -> str | None:
        return f"ckpt-{self.selected.ordinal:06d}" if self.selected else None

    @property
    def no_valid_checkpoint(self) -> bool:
        return self.selected is None

    def report_lines(self) -> list[str]:
        """§11.2 rule 3: report the recovery path explicitly."""
        lines = []
        for c in self.skips:
            lines.append(f"[RECOVERY] checkpoint-{c.ordinal:06d}: {c.reason}")
        if self.selected is not None:
            lines.append(
                f"[RECOVERY] checkpoint-{self.selected.ordinal:06d}: VERIFIED "
                f"(global_step={self.selected.global_step}, valid)"
            )
            lines.append(f"RESUME POINT: checkpoint-{self.selected.ordinal:06d}")
            steps = [c.ordinal for c in self.skips]
            if steps:
                path = " → ".join(str(s) for s in steps + [self.selected.ordinal])
                lines.append(
                    f"recovery_from: {path} ({len(self.skips)} skips, "
                    "all recorded in event log)"
                )
        else:
            lines.append(
                "NO VALID CHECKPOINT — resume impossible; fork from last "
                "known state or restart"
            )
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "resume_point": self.resume_point,
            "resume_ordinal": self.resume_ordinal,
            "attempted_newest": self.attempted_newest,
            "skips": [s.to_dict() for s in self.skips],
            "no_valid_checkpoint": self.no_valid_checkpoint,
        }


class CheckpointStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "checkpoints"

    # -- addressing -------------------------------------------------------

    def dir_for(self, ordinal: int) -> Path:
        return self.root / f"ckpt-{ordinal:06d}"

    def _staging_dir(self, ordinal: int) -> Path:
        return self.root / f"ckpt-{ordinal:06d}{_STAGING_SUFFIX}"

    # -- transactional write (§11.1) ---------------------------------------

    def write(
        self,
        ordinal: int,
        payload: dict[str, bytes],
        *,
        global_step: int,
        epoch: int,
        components: set[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Commit one checkpoint generation atomically.

        Crash windows: the final directory only ever appears complete-and-
        renamed; manifest and commit marker follow — any gap leaves an
        incomplete directory that recovery ignores and reports."""
        if not payload:
            raise ValidationBlock(f"checkpoint {ordinal}: empty payload")
        dest = self.dir_for(ordinal)
        if dest.exists():
            raise ValidationBlock(
                f"checkpoint {ordinal} already exists at {dest} — "
                "checkpoints are write-once (never overwrite a generation)"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self._staging_dir(ordinal)
        if staging.exists():  # leftover from a crashed write — replace it
            _rmtree(staging)
        staging.mkdir(parents=True)

        files: dict[str, str] = {}
        try:
            # 1-3: write payload files, hash, fsync
            for rel, data in sorted(payload.items()):
                if "/" in rel or rel in (".", ".."):
                    raise ValidationBlock(f"illegal payload path: {rel!r}")
                fpath = staging / rel
                fpath.parent.mkdir(parents=True, exist_ok=True)
                with open(fpath, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                files[rel] = content_hash_bytes(data)

            # 4: atomic rename staging → final (directory appears whole)
            os.rename(staging, dest)

            # 5: manifest.json + fsync
            manifest = {
                "schema_version": 1,
                "ordinal": ordinal,
                "global_step": global_step,
                "epoch": epoch,
                "components": sorted(components or set()),
                "files": files,
                "created_ts": time.time(),
                "extra": extra or {},
            }
            manifest_path = dest / MANIFEST_NAME
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            manifest_hash = content_hash_bytes(
                json.dumps(manifest, sort_keys=True).encode("utf-8")
            )

            # 6: commit marker + fsync (THE authority for existence)
            marker = dest / COMMIT_MARKER
            with open(marker, "w", encoding="utf-8") as f:
                f.write(manifest_hash)
                f.flush()
                os.fsync(f.fileno())
            _fsync_dir(dest)
            _fsync_dir(self.root)
        except BaseException:
            # Leave the incomplete directory in place — recovery reports it
            # (§11.1: no commit marker ⇒ treated as incomplete, not hidden).
            raise
        return dest

    # -- verification (§11.3) ----------------------------------------------

    def verify(self, ordinal: int) -> tuple[bool, str]:
        """load manifest → verify per-file sha256 → verify commit marker."""
        dest = self.dir_for(ordinal)
        if not dest.is_dir():
            return False, "NOT FOUND"
        marker = dest / COMMIT_MARKER
        if not marker.is_file():
            return False, "NO COMMIT MARKER (incomplete)"
        manifest_path = dest / MANIFEST_NAME
        if not manifest_path.is_file():
            return False, "MANIFEST MISSING (incomplete)"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False, "MANIFEST UNREADABLE (corrupt)"
        recomputed = content_hash_bytes(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        )
        try:
            recorded = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return False, "COMMIT MARKER UNREADABLE (corrupt)"
        if recorded != recomputed:
            return False, "MANIFEST HASH MISMATCH (corrupt — skipped)"
        files = manifest.get("files", {})
        for rel, expected in files.items():
            fpath = dest / rel
            if not fpath.is_file():
                return False, f"FILE MISSING: {rel}"
            digest = sha256_hex(fpath.read_bytes())
            if f"sha256:{digest}" != expected:
                return False, f"FILE HASH MISMATCH: {rel} (corrupt — skipped)"
        return True, "VERIFIED"

    # -- newest-valid predicate (§11.2) ------------------------------------

    def scan_candidates(self) -> list[Candidate]:
        """All generation directories (staging leftovers included as
        explicitly-invalid candidates so they are REPORTED, not hidden)."""
        if not self.root.is_dir():
            return []
        out: list[Candidate] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            m = _CKPT_DIR_RE.match(entry.name)
            if m:
                ordinal = int(m.group(1))
            elif entry.name.endswith(_STAGING_SUFFIX) and entry.name.startswith("ckpt-"):
                # staging crash leftover: ordinal parseable for reporting
                sm = re.match(r"^ckpt-(\d{6})", entry.name)
                ordinal = int(sm.group(1)) if sm else -1
                out.append(Candidate(
                    ordinal, entry, False,
                    "STAGING LEFTOVER (crash mid-write — incomplete)",
                ))
                continue
            else:
                continue
            ok, reason = self.verify(ordinal)
            global_step = None
            manifest = None
            manifest_path = entry / MANIFEST_NAME
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    global_step = int(manifest.get("global_step", -1))
                except (json.JSONDecodeError, TypeError, ValueError):
                    manifest = None
            out.append(Candidate(
                ordinal, entry, ok,
                reason if not ok else "VERIFIED",
                global_step=global_step,
                manifest=manifest,
            ))
        return out

    def newest_valid(self) -> Selection:
        """§11.2: recovery_candidates with verification DURING selection;
        argmax (global_step, ordinal) over VALID candidates only; every
        *walked-over* attempt carries a reason (invalid generations newer
        than the selected one — the recovery path), never ordinal math."""
        candidates = self.scan_candidates()
        if not candidates:
            return Selection(selected=None, skips=(), attempted_newest=None)

        attempted_newest = max(
            (c.ordinal for c in candidates if not c.path.name.endswith(_STAGING_SUFFIX)),
            default=None,
        )
        valid = [c for c in candidates if c.valid]
        if valid:
            selected = max(
                valid, key=lambda c: (c.global_step if c.global_step is not None else -1,
                                      c.ordinal)
            )
            # Recovery walk: attempts NEWER than the selected candidate that
            # failed verification (reported, never silently dropped).
            skips = sorted(
                (c for c in candidates
                 if not c.valid and c.ordinal > selected.ordinal),
                key=lambda c: c.ordinal,
                reverse=True,
            )
        else:
            selected = None
            skips = sorted(candidates, key=lambda c: c.ordinal, reverse=True)
        return Selection(
            selected=selected, skips=tuple(skips), attempted_newest=attempted_newest
        )

    def components_of(self, ordinal: int) -> set[str]:
        """§11.4 completeness check — which components the manifest claims."""
        manifest_path = self.dir_for(ordinal) / MANIFEST_NAME
        if not manifest_path.is_file():
            return set()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return set()
        return set(manifest.get("components", []))

    def missing_components(
        self, ordinal: int, required: tuple[str, ...] = REQUIRED_COMPONENTS
    ) -> list[str]:
        present = self.components_of(ordinal)
        return [c for c in required if c not in present]

    def load_payload(self, ordinal: int, rel: str) -> bytes:
        """§11.3: corrupt → BLOCK, not 'try anyway'."""
        ok, reason = self.verify(ordinal)
        if not ok:
            raise ValidationBlock(
                f"checkpoint {ordinal} failed verification: {reason}",
                hint="recovery selection skips invalid generations (12 §11.2)",
            )
        fpath = self.dir_for(ordinal) / rel
        if not fpath.is_file():
            raise NotFound(f"checkpoint {ordinal}: payload {rel!r} not found")
        return fpath.read_bytes()

    def prune(self, keep: int) -> list[int]:
        """Keep top-K *valid* generations (§11.2); delete older committed
        ones only after the retained set verifies. Never touches invalid
        dirs (they are reported, not silently reaped)."""
        valid = [c for c in self.scan_candidates() if c.valid]
        valid.sort(key=lambda c: (c.global_step if c.global_step is not None else -1,
                                  c.ordinal), reverse=True)
        removed: list[int] = []
        for cand in valid[keep:]:
            _rmtree(cand.path)
            removed.append(cand.ordinal)
        return sorted(removed)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "CheckpointStore",
    "Candidate",
    "Selection",
    "REQUIRED_COMPONENTS",
    "COMMIT_MARKER",
    "MANIFEST_NAME",
]
