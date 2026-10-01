"""Validation report — the fail-closed gate's product-visible output.

Normative: 12_training_system.md §7.2 (report format), §7.1 (BLOCK vs WARN).

Rules encoded here:
  * PASS / WARN never blocks; FAIL always blocks ("probably compatible"
    does not exist — 12 §7.1).
  * Fail-fast: execution stops at the first FAIL ("If steps 1–N fail,
    TRAINING DOES NOT START" — 12 §7.2). Checks after the first FAIL are
    not executed and never appear as PASS.
  * A check whose verdict cannot be produced is a FAIL, not a SKIP
    (fail-closed: unverifiable == failed verification).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

#: Valid verdicts. Anything else is a programming error → fail-closed.
VERDICTS = frozenset({PASS, WARN, FAIL})


@dataclass(frozen=True)
class Check:
    step: int
    id: str
    label: str
    verdict: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"invalid verdict {self.verdict!r} (fail-closed: PASS|WARN|FAIL only)")

    @property
    def blocking(self) -> bool:
        return self.verdict == FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "id": self.id,
            "label": self.label,
            "verdict": self.verdict,
            "detail": self.detail,
            "data": self.data,
        }

    def render(self) -> str:
        out = f"[{self.verdict}] {self.label}"
        if self.detail:
            out += "\n" + "\n".join(f"    {line}" for line in self.detail.splitlines())
        return out


@dataclass(frozen=True)
class ValidationReport:
    run_id: str
    checks: tuple[Check, ...]
    #: EXACT | PORTABLE (12 §8); None when the gate blocked.
    mode: str | None
    segment: str | None
    title: str = "VALIDATION REPORT"
    pass_result: str = "SAFE TO RESUME"
    fail_result: str = "RESUME BLOCKED"

    @property
    def blocked(self) -> bool:
        return any(c.verdict == FAIL for c in self.checks)

    @property
    def first_failure(self) -> Check | None:
        for c in self.checks:
            if c.verdict == FAIL:
                return c
        return None

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == WARN]

    @property
    def result(self) -> str:
        return self.fail_result if self.blocked else self.pass_result

    @property
    def failed_step(self) -> int | None:
        f = self.first_failure
        return f.step if f else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "title": self.title,
            "result": self.result,
            "blocked": self.blocked,
            "mode": self.mode,
            "segment": self.segment,
            "failed_step": self.failed_step,
            "checks": [c.to_dict() for c in self.checks],
        }

    def render(self) -> str:
        """12 §7.2 format."""
        lines = [self.title, "─" * 41]
        for c in self.checks:
            lines.append(c.render())
        lines.append("")
        lines.append("RESULT:")
        lines.append(f"    {self.result}")
        if not self.blocked:
            if self.mode:
                lines.append(f"    MODE: {self.mode} CONTINUATION")
            if self.segment:
                lines.append(f"    SEGMENT: {self.segment}")
        return "\n".join(lines)

    def block_message(self) -> str:
        """Message for the ValidationBlock error (exit 1)."""
        f = self.first_failure
        if f is None:  # pragma: no cover — blocked implies a failure
            return "validation blocked"
        return f"step {f.step} failed: {f.label}"
