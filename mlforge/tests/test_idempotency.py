"""Build step 5 tests — command idempotency journal (13 §4.4, 12 §23.4).

Specs as executable checks:
  * duplicate of a SUCCEEDED command → original result, fn never re-runs
  * duplicate while IN FLIGHT → exit 3 (command lease, 12 §23.1)
  * FAILED command → same command_id retryable (failures retryable,
    duplicates of successes are not)
  * crash orphan (stale `started`) → retryable after the lease window
  * duplicates journaled as COMMAND_DEDUPED in run events (13 §4.4),
    never as a state transition
  * journal is append-only fsynced; corrupt record → fail-closed
"""

from __future__ import annotations

import os

import pytest

from mlforge.commands import (
    KIND_DUPLICATE,
    KIND_IN_FLIGHT,
    KIND_NEW,
    CommandJournal,
    execute,
)
from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.journal import CorruptJournal
from mlforge.run_spec import RunSpec
from mlforge.workflow import WorkflowAPI


class FakeClock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, s: float) -> None:
        self.now += s


@pytest.fixture
def journal(tmp_path):
    return CommandJournal(tmp_path)


def test_begin_then_succeed_then_duplicate(journal, tmp_path):
    d1 = journal.begin("cmd_1", "train")
    assert d1.kind == KIND_NEW
    journal.succeed("cmd_1", {"run_id": "run_A"})

    d2 = journal.begin("cmd_1", "train")
    assert d2.kind == KIND_DUPLICATE
    assert d2.result == {"run_id": "run_A"}
    # exactly one started + one succeeded record total (append-only history)
    assert len(journal.history("cmd_1")) == 2


def test_execute_runs_fn_exactly_once(journal):
    calls = []

    def fn():
        calls.append(1)
        return "result-A"

    r1 = execute(journal, "cmd_X", "train", fn)
    r2 = execute(journal, "cmd_X", "train", fn)
    assert r1 == r2 == "result-A"
    assert calls == [1]  # never re-executed


def test_in_flight_duplicate_blocked(journal):
    journal.begin("cmd_Y", "train")  # no terminal record yet
    assert journal.begin("cmd_Y", "train").kind == KIND_IN_FLIGHT
    with pytest.raises(PreconditionFailed) as exc:
        execute(journal, "cmd_Y", "train", lambda: "x")
    assert exc.value.exit_code == 3


def test_failed_command_is_retryable(journal):
    def failing():
        raise RuntimeError("OOM during setup")

    with pytest.raises(RuntimeError):
        execute(journal, "cmd_Z", "train", failing)
    rec = journal.latest("cmd_Z")
    assert rec["status"] == "failed"
    assert "OOM during setup" in rec["error"]

    # same command_id retried → executes again (failures are retryable)
    calls = []

    def ok():
        calls.append(1)
        return {"run_id": "run_B"}

    assert execute(journal, "cmd_Z", "train", ok) == {"run_id": "run_B"}
    assert calls == [1]


def test_stale_started_is_retryable(tmp_path):
    clock = FakeClock()
    journal = CommandJournal(tmp_path, stale_after=600.0, clock=clock)
    journal.begin("cmd_S", "train")   # client crashed, no terminal record
    clock.advance(601)
    decision = journal.begin("cmd_S", "train")
    assert decision.kind == KIND_NEW
    assert decision.stale_recovery is True
    # history shows the crash orphan was superseded
    statuses = [r["status"] for r in journal.history("cmd_S")]
    assert statuses == ["started", "started"]
    assert journal.history("cmd_S")[1]["retry_of"] == "started"


def test_in_flight_within_window_not_stale(journal):
    journal.begin("cmd_W", "train")
    assert journal.begin("cmd_W", "train").kind == KIND_IN_FLIGHT


def test_unserializable_result_fails_closed(journal):
    journal.begin("cmd_U", "export")
    with pytest.raises(ValidationBlock):
        journal.succeed("cmd_U", object())  # not JSON → refuse to journal


def test_success_without_started_refused(journal):
    with pytest.raises(ValidationBlock):
        journal.succeed("cmd_N", {"x": 1})


def test_corrupt_journal_is_fail_closed(tmp_path):
    p = tmp_path / "commands.jsonl"
    p.write_text('{"command_id": "a", "status": "started"}\n{truncated\n')
    with pytest.raises(CorruptJournal):
        CommandJournal(tmp_path)._records()


def test_commands_journal_is_append_only_on_disk(tmp_path):
    journal = CommandJournal(tmp_path)
    journal.begin("cmd_A", "train")
    journal.succeed  # noqa: B018 — ensure attribute exists
    journal.begin("cmd_A", "train")  # duplicate decision, no overwrite
    stat_before = os.stat(journal.path).st_size
    journal.begin("cmd_A", "train")
    assert os.stat(journal.path).st_size >= stat_before  # only grows


# ---------------------------------------------------------------------------
# Workflow-level (13 §4.4: visible in `mlforge events` as COMMAND_DEDUPED)
# ---------------------------------------------------------------------------

def make_spec() -> RunSpec:
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={k: v for k, v in {
            "optimizer": "adamw", "learning_rate": 1e-4,
            "scheduler": "cosine", "loss": "l1", "seed": 42,
            "global_batch": 32, "epochs": 50,
            "precision_policy": "bf16",
        }.items()},
    )


def test_workflow_dedupe_journals_command_deduped(tmp_path):
    wf = WorkflowAPI(tmp_path)
    run_id = wf.create_run(make_spec()).run_id
    calls = []

    def start_run():
        calls.append(1)
        return {"run_id": run_id}

    r1 = wf.execute_idempotent("cmd_01JABC", "train", start_run, run_id=run_id)
    r2 = wf.execute_idempotent("cmd_01JABC", "train", start_run, run_id=run_id)
    assert r1 == r2
    assert calls == [1]

    events = wf.get_run_events(run_id)
    names = [e["event"] for e in events]
    assert "COMMAND_DEDUPED" in names
    deduped = events[-1]
    assert deduped["command_id"] == "cmd_01JABC"
    assert "to" not in deduped          # never a state transition
    assert wf.get_run_state(run_id) == "CREATED"  # unchanged (13 §4.4: exit 0)


def test_workflow_dedupe_without_run_still_dedupes(tmp_path):
    wf = WorkflowAPI(tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return 42

    assert wf.execute_idempotent("cmd_noop", "evaluate", fn) == 42
    assert wf.execute_idempotent("cmd_noop", "evaluate", fn) == 42
    assert calls == [1]
