"""Dataset configuration — identity registry vs machine-local paths.

Two files, two lifetimes (12 §6.3, §10.1, 13 §10):

  <workspace>/datasets.yaml                 # identity registrations
                                            # (portable, syncable)
  $MLFORGE_HOME/datasets_<project>.yaml     # machine-local PATHS only
                                            # (never part of identity)

`MLFORGE_HOME` defaults to `~/.mlforge` (13 §10 shows `~/.mlforge/…`);
tests and multi-workspace machines may redirect it. Paths are verified,
never scanned for: the system resolves a configured path and hashes what
is there — a wrong path is a mismatch, not a discovery opportunity.

Both files are YAML (the documented spec format) parsed by the strict
stdlib subset loader (`mlforge.yamlmini`) — anything outside the subset
is a config error (exit 3), never a guess.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed
from mlforge.yamlmini import YamlError, dump, load_file

ENV_HOME = "MLFORGE_HOME"
DEFAULT_HOME = ".mlforge"


def mlforge_home() -> Path:
    """$MLFORGE_HOME or ~/.mlforge (13 §10 machine-local config)."""
    env = os.environ.get(ENV_HOME)
    if env:
        return Path(env)
    try:
        return Path.home() / DEFAULT_HOME
    except (OSError, RuntimeError):
        return Path.cwd() / DEFAULT_HOME


def project_slug(root: str | Path) -> str:
    """Filename-safe project token. `mlforge init` (pending) will pin a
    real project name; until then the workspace directory name is the
    most stable machine-local key (12 §6.3: per-machine file)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(root).resolve().name) or "workspace"


def paths_file(root: str | Path) -> Path:
    return mlforge_home() / f"datasets_{project_slug(root)}.yaml"


def registry_file(root: str | Path) -> Path:
    return Path(root) / "datasets.yaml"


def _load_yaml(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = load_file(path)
    except YamlError as exc:
        raise PreconditionFailed(
            f"{what} unreadable ({path}): {exc}",
            hint="fix the YAML (supported subset: mappings, lists, flow "
                 "collections) or delete it to re-register",
        ) from exc
    if not isinstance(data, dict):
        raise PreconditionFailed(f"{what} must be a mapping: {path}")
    return data


def load_paths(root: str | Path) -> dict[str, str]:
    """name → configured absolute path (machine-local, mutable)."""
    data = _load_yaml(paths_file(root), "dataset path config")
    out: dict[str, str] = {}
    for name, entry in data.items():
        if isinstance(entry, str):
            out[str(name)] = entry  # accepted flat form: `coco_2017: /data/coco`
        elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
            out[str(name)] = entry["path"]
        elif entry is not None:
            raise PreconditionFailed(
                f"dataset path config: bad entry for {name!r} "
                f"(expected `path: ...`)"
            )
    return out


def set_path(root: str | Path, name: str, path: str | Path) -> Path:
    """Record the local path for a dataset (12 §10.1 — explicit, verified)."""
    data = load_paths(root)
    data[name] = str(Path(path))
    target = paths_file(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".yaml.tmp")
    tmp.write_text(dump({k: {"path": v} for k, v in sorted(data.items())}),
                   encoding="utf-8")
    tmp.replace(target)
    return target


def load_registry(root: str | Path) -> dict[str, Any]:
    """Identity registrations (13 §10 `datasets.yaml`) — the dataset
    directories remain authoritative; this index is the syncable view."""
    data = _load_yaml(registry_file(root), "dataset registry")
    return {str(k): v for k, v in data.items()}


def write_registry(root: str | Path, name: str, entry: dict[str, Any]) -> Path:
    data = load_registry(root)
    data[name] = entry
    target = registry_file(root)
    tmp = target.with_suffix(".yaml.tmp")
    tmp.write_text(dump(data), encoding="utf-8")
    tmp.replace(target)
    return target


__all__ = [
    "ENV_HOME",
    "load_paths",
    "load_registry",
    "mlforge_home",
    "paths_file",
    "project_slug",
    "registry_file",
    "set_path",
    "write_registry",
]
