"""Content-addressed artifact store — build step 2.

Normative: 12_training_system.md §6.1.

    ~/.mlforge/store/sha256/ab/abcdef...

Rules enforced here:
  * identity = content hash (`sha256:<hex>`), never path or size
  * write protocol: staging → fsync → atomic rename (no partial blob can
    ever appear at the final path — a reader either sees nothing or the
    complete, correct bytes)
  * dedup: same bytes in twice → one blob
  * verify: re-hash on demand; corruption detected, never "trusted"
  * staging orphans (crash mid-write) are recoverable garbage, not
    half-artifacts: `recover_staging()` sweeps them (§6.1 recovery scan)
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path

from mlforge.errors import NotFound, ValidationBlock
from mlforge.hashing import content_hash_bytes

_CHUNK = 1 << 20  # 1 MiB


class ContentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.shard_dir = self.root / "sha256"
        self.staging_dir = self.root / "staging"

    # -- addressing -----------------------------------------------------

    def path_for(self, content_hash: str) -> Path:
        if not content_hash.startswith("sha256:") or len(content_hash) != 7 + 64:
            raise ValidationBlock(f"malformed content hash: {content_hash!r}")
        hexdigest = content_hash[7:]
        return self.shard_dir / hexdigest[:2] / hexdigest

    def contains(self, content_hash: str) -> bool:
        return self.path_for(content_hash).is_file()

    # -- writes (staging → fsync → atomic rename) ------------------------

    def put_bytes(self, data: bytes) -> str:
        content_hash = content_hash_bytes(data)
        dest = self.path_for(content_hash)
        if dest.is_file():
            return content_hash  # dedup — content-addressed identity
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.staging_dir / f"{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            # Atomic: dest either absent or the complete correct bytes.
            os.replace(tmp, dest)
            self._fsync_dir(dest.parent)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return content_hash

    def put_file(self, src: str | Path, *, chunk_hash: str | None = None) -> str:
        """Copy a file into the store. If `chunk_hash` (the expected
        `sha256:...`) is given it is verified before commit — the caller's
        identity claim is checked, never assumed (§6.2)."""
        src = Path(src)
        if not src.is_file():
            raise NotFound(f"source file not found: {src}")
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.staging_dir / f"{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        try:
            with open(src, "rb") as fin, open(tmp, "wb") as fout:
                while True:
                    chunk = fin.read(_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    fout.write(chunk)
                fout.flush()
                os.fsync(fout.fileno())
            h = "sha256:" + digest.hexdigest()
            if chunk_hash is not None and chunk_hash != h:
                raise ValidationBlock(
                    f"content hash mismatch for {src}: expected {chunk_hash}, got {h}"
                )
            dest = self.path_for(h)
            if dest.is_file():
                return h  # dedup
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, dest)
            self._fsync_dir(dest.parent)
            return h
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # -- reads -----------------------------------------------------------

    def get_bytes(self, content_hash: str, *, verify: bool = False) -> bytes:
        p = self.path_for(content_hash)
        if not p.is_file():
            raise NotFound(f"artifact {content_hash} not in store")
        data = p.read_bytes()
        if verify and content_hash_bytes(data) != content_hash:
            raise ValidationBlock(
                f"artifact {content_hash} failed integrity verification (corrupt blob)",
                hint="do not use — re-fetch or re-derive the artifact",
            )
        return data

    def verify(self, content_hash: str) -> bool:
        p = self.path_for(content_hash)
        if not p.is_file():
            return False
        digest = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
        return "sha256:" + digest.hexdigest() == content_hash

    def size(self, content_hash: str) -> int:
        p = self.path_for(content_hash)
        if not p.is_file():
            raise NotFound(f"artifact {content_hash} not in store")
        return p.stat().st_size

    def list_hashes(self) -> list[str]:
        if not self.shard_dir.is_dir():
            return []
        out = []
        for shard in sorted(self.shard_dir.iterdir()):
            if shard.is_dir():
                for blob in sorted(shard.iterdir()):
                    if blob.is_file():
                        out.append("sha256:" + blob.name)
        return out

    # -- crash recovery --------------------------------------------------

    def recover_staging(self, *, max_age_seconds: float = 0.0) -> list[str]:
        """Sweep staging orphans left by a crash mid-write (§6.1: blobs
        without committed references are orphans — deleted). A staging file
        can never be partially committed: commit is a single atomic rename."""
        if not self.staging_dir.is_dir():
            return []
        removed = []
        now = time.time()
        for tmp in list(self.staging_dir.iterdir()):
            if not tmp.is_file():
                continue
            if now - tmp.stat().st_mtime < max_age_seconds:
                continue
            tmp.unlink()
            removed.append(tmp.name)
        return removed

    def delete(self, content_hash: str) -> bool:
        """GC primitive — ONLY reachable-based GC may call this (§6.1)."""
        p = self.path_for(content_hash)
        if p.is_file():
            p.unlink()
            # prune empty shard dir
            try:
                p.parent.rmdir()
            except OSError:
                pass
            return True
        return False

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """Persist the rename itself (directory entry), not just contents."""
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
