"""Packaging checks — `mlforge` is a real installable CLI (13 §4.1).

A wheel built from this tree must put ONE command on PATH whose entry
point is `mlforge.cli.main:main`, with zero runtime dependencies, and
the same entry point must be reachable as `python -m mlforge` for
environments without a scripts directory on PATH.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(*argv: str, cwd: Path) -> subprocess.CompletedProcess:
    # run from an arbitrary directory: the CLI must not depend on cwd
    return subprocess.run(
        [sys.executable, "-m", "mlforge", *argv],
        capture_output=True, text=True, cwd=cwd, timeout=60,
    )


def test_pyproject_declares_console_script_and_zero_deps():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["mlforge"] == "mlforge.cli.main:main"
    assert data["project"]["dependencies"] == []          # zero-dep core
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"


def test_module_execution_prints_version(tmp_path):
    proc = _run("--version", cwd=tmp_path)
    assert proc.returncode == 0
    assert proc.stdout.startswith("mlforge ")


def test_bare_module_execution_opens_help(tmp_path):
    """Typing `mlforge` with no subcommand opens the reference, exit 0."""
    proc = _run(cwd=tmp_path)
    assert proc.returncode == 0
    assert "usage:" in proc.stdout and "MLForge" in proc.stdout


def test_module_execution_status_from_empty_workspace(tmp_path):
    proc = _run("--root", str(tmp_path), "status", cwd=tmp_path)
    assert proc.returncode == 0
    assert "No runs." in proc.stdout


def test_module_execution_pending_command_fails_closed(tmp_path):
    """Installed binary keeps the spec's exit 4 (never fake success).

    `init`/`configure` are implemented now; `serve` (no build step)
    remains the honest NOT_IMPLEMENTED example."""
    proc = _run("--root", str(tmp_path), "serve", cwd=tmp_path)
    assert proc.returncode == 4
    assert "[NOT_IMPLEMENTED]" in proc.stderr
