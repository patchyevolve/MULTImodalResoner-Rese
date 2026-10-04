"""Training worker — the process that owns a RUNNING run (12 §12.4).

Ownership rules enforced here:
  * the worker is spawned by the SUPERVISOR daemon, never by an
    interactive shell (spawn happens in supervisor.py); closing a
    terminal cannot kill training.
  * single writer: startup re-proves lease ownership (session_token) —
    without a valid renew the worker exits before touching anything.
  * every state change goes through WorkflowAPI (single write path);
    the worker never writes state.json or events itself.
  * heartbeat + lease renew go TOGETHER every `heartbeat_interval`
    (12 §23.2) — a dead worker stops beating → supervisor marks
    INTERRUPTED → reconciliation derives the resume point (§12.3).
  * pause/stop arrive as control intents (runtime/control.py) and are
    performed as graceful machine transitions; the worker NEVER decides
    to pause/stop by itself (13 §1 "no auto-anything").

Start contract: the CLI's train/resume flow leaves the run in READY;
the worker fires `preflight_pass` (READY → RUNNING) only AFTER lease
ownership is proven — READY is the safe pre-start state (a run can sit
in READY forever without anyone claiming it).

Exit codes (13 §4.2): 0 ok/terminal · 1 validation block · 2 not found ·
3 precondition (lease/state) · 4 runtime error.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mlforge.errors import MlforgeError, NotFound, PreconditionFailed, ValidationBlock
from mlforge.leases import RunLeaseManager
from mlforge.planner import PLAN_FILENAME, ExecutionPlan, detect_capabilities
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS, CheckpointStore
from mlforge.runtime.control import clear_control, read_control
from mlforge.runtime.heartbeat import DEFAULT_HEARTBEAT_INTERVAL, HeartbeatWriter
from mlforge.runtime.trainer import Trainer, TrainState, resolve_trainer
from mlforge.states import RunState
from mlforge.workflow import WorkflowAPI

#: worker → supervisor duplicate-spawn tolerance: a heartbeat fresher than
#: twice the interval means another live worker owns the run.
_DUPLICATE_GRACE = 2


class WorkerExit(Exception):
    """Worker finished with a specific exit code (already recorded)."""

    def __init__(self, code: int, message: str = ""):
        super().__init__(message)
        self.code = code


class Worker:
    def __init__(
        self,
        root: str | Path,
        run_id: str,
        session_token: str,
        *,
        trainer: Trainer | None = None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        checkpoint_interval: int = 5,
        poll_interval: float = 0.02,
        clock: Any = time.time,
        sleep: Any = time.sleep,
        max_iterations: int | None = None,
    ):
        self.root = Path(root)
        self.run_id = run_id
        self.session_token = session_token
        self.trainer = trainer
        self.heartbeat_interval = heartbeat_interval
        self.checkpoint_interval = max(1, checkpoint_interval)
        self.poll_interval = poll_interval
        self.clock = clock
        self.sleep = sleep
        self.max_iterations = max_iterations
        self._owns = False  # proven lease owner; gates cleanup rights
        self._progress: dict[str, Any] | None = None  # live on_progress cell

    # -- plumbing -----------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_id

    def _runtime_config(self) -> dict[str, Any]:
        p = self.run_dir / "state" / "runtime.json"
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _append_metric(self, record: dict[str, Any]) -> None:
        mdir = self.run_dir / "metrics"
        mdir.mkdir(parents=True, exist_ok=True)
        with open(mdir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _write_live(self, payload: dict[str, Any]) -> None:
        """`state/live.json` — current transient metrics (13 §9.4).
        Atomic (readers never see a partial file); removed on graceful
        exit: live state dies with the process (12 §16)."""
        state = self.run_dir / "state"
        state.mkdir(parents=True, exist_ok=True)
        payload = {"run_id": self.run_id, "ts": self.clock(), **payload}
        tmp = state / "live.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, state / "live.json")

    def _clear_live(self) -> None:
        (self.run_dir / "state" / "live.json").unlink(missing_ok=True)

    def _load_plan(self) -> Any:
        """The persisted execution plan (build step 9) — absent is fine
        (caller-passed gate wiring may not have written one), corrupt is
        not: fail-closed before touching any state (12 §17)."""
        p = self.run_dir / PLAN_FILENAME
        if not p.is_file():
            return None
        try:
            return ExecutionPlan.read(p)
        except ValidationBlock:
            raise
        except (KeyError, ValueError, TypeError) as exc:
            raise ValidationBlock(
                f"persisted execution plan unreadable: {p}: {exc}"
            ) from exc

    def _load_spec(self) -> Any:
        """run_spec.json — immutable, written once (12 §17). Needed to
        select the REAL trainer (run_spec.model chooses the learning
        loop); absent/corrupt ⇒ fail-closed before READY → RUNNING."""
        from mlforge.run_spec import RunSpec

        p = self.run_dir / "run_spec.json"
        if not p.is_file():
            raise ValidationBlock(
                f"run {self.run_id}: run_spec.json missing — cannot "
                "select a trainer"
            )
        try:
            return RunSpec.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except ValidationBlock:
            raise
        except Exception as exc:
            raise ValidationBlock(
                f"run {self.run_id}: run_spec.json unreadable: {exc}"
            ) from exc

    def _restore_payloads(self, store: CheckpointStore, ordinal: int) -> dict[str, bytes]:
        """Load every REQUIRED_COMPONENTS payload of a verified checkpoint
        (12 §11.4) — one verify pass, then read. Corrupt/incomplete ⇒
        ValidationBlock: a trainer must never resume from a partial lie."""
        ok, reason = store.verify(ordinal)
        if not ok:
            raise ValidationBlock(
                f"checkpoint {ordinal} failed verification: {reason}"
            )
        directory = store.dir_for(ordinal)
        out: dict[str, bytes] = {}
        missing: list[str] = []
        for component in REQUIRED_COMPONENTS:
            path = directory / component
            if not path.is_file():
                missing.append(component)
                continue
            out[component] = path.read_bytes()
        if missing:
            raise ValidationBlock(
                f"checkpoint {ordinal}: components missing "
                f"({', '.join(missing)}) — not resumable (12 §11.4)"
            )
        return out

    def _init_weights(self) -> bytes | None:
        """Fine-tune initialization (13 §6.4, 12 §15.3): the parent run's
        committed `model` component, matched by the lineage's
        `parent_checkpoint` — the parent's COMMIT-marker manifest hash.

        Only `origin == "finetune"` gets parent weights: retrain/fork/
        train start fresh BY DESIGN (13 §6.3 "fresh init"). Resume (own
        payloads present) bypasses this entirely — the caller decides.
        """
        from mlforge.lineage import read_lineage

        lineage = read_lineage(self.run_dir)
        if lineage.get("origin") != "finetune":
            return None
        parent_run = str(lineage.get("parent_run") or "")
        parent_ck = str(lineage.get("parent_checkpoint") or "")
        if not parent_run or not parent_ck:
            raise PreconditionFailed(
                f"run {self.run_id}: finetune lineage lacks parent_run / "
                "parent_checkpoint — cannot initialize from weights",
            )
        parent_ckpt_root = self.root / "runs" / parent_run / "checkpoints"
        if not parent_ckpt_root.is_dir():
            raise PreconditionFailed(
                f"parent run {parent_run} has no checkpoints — nothing to "
                "initialize from",
            )
        short = parent_ck[:19]
        for d in sorted(parent_ckpt_root.iterdir(), reverse=True):
            marker = d / "COMMIT" if d.is_dir() else None
            if marker is None or not marker.is_file():
                continue
            try:
                digest = marker.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if digest != parent_ck:
                continue
            # Match found — full §11.3 verification before trusting bytes.
            try:
                ordinal = int(d.name.split("-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise PreconditionFailed(
                    f"parent checkpoint directory unreadable: {d.name}"
                ) from exc
            # CheckpointStore is rooted at the RUN dir (it appends
            # `checkpoints/` itself) — passing the checkpoints dir would
            # verify `checkpoints/checkpoints/…` and refuse every match.
            ok, reason = CheckpointStore(
                self.root / "runs" / parent_run
            ).verify(ordinal)
            if not ok:
                raise PreconditionFailed(
                    f"parent checkpoint {short}… failed verification: "
                    f"{reason} — finetune cannot initialize fail-closed",
                )
            model_file = d / "model"
            if not model_file.is_file():
                raise PreconditionFailed(
                    f"parent checkpoint {short}… has no `model` component "
                    "— nothing to initialize from (12 §15.3)",
                )
            return model_file.read_bytes()
        raise PreconditionFailed(
            f"parent checkpoint COMMIT {short}… not found under run "
            f"{parent_run} — the producing run's checkpoint is missing "
            "or was replaced; finetune cannot initialize fail-closed",
        )

    def _progress_beater(self, hb: HeartbeatWriter, step: int, epoch: int):
        """`on_progress` closure for trainer steps that outlive the
        heartbeat interval (12 §23.2: 30s beats, 120s supervisor timeout —
        a long RF-DETR epoch must beat DURING training or the run is
        marked INTERRUPTED). Throttled: at most one beat per interval."""
        cell = {
            "global_step": step,
            "epoch": epoch,
            "last_beat": float(self.clock()),
        }
        self._progress = cell

        def on_progress(detail: Any = None) -> None:
            now = float(self.clock())
            if now - float(cell["last_beat"]) < self.heartbeat_interval:
                return
            hb.beat(global_step=int(cell["global_step"]), state="RUNNING")
            cell["last_beat"] = float(self.clock())

        return on_progress


    # -- lifecycle ------------------------------------------------------------

    def run(self) -> int:
        wf = WorkflowAPI(self.root)
        lease = RunLeaseManager(self.root, clock=self.clock)
        hb = HeartbeatWriter(
            self.root, self.run_id, self.session_token, lease=lease, clock=self.clock
        )
        store = CheckpointStore(self.run_dir)
        try:
            # 1. Single-writer proof — renew with OUR token (owner check).
            #    Wrong/absent/SUSPECT lease ⇒ PreconditionFailed, nothing touched.
            lease.renew(self.run_id, self.session_token)
            self._owns = True

            # 2. Start contract: READY is the only startable state.
            state = wf.get_run_state(self.run_id)
            if state == RunState.READY.value:
                # Capability negotiation (12 §13.1) BEFORE the transition:
                # detection/plan failure must leave the run in READY with
                # no segment behind (12 §12.2 execution segments are
                # created only once the run actually starts).
                caps = detect_capabilities()
                plan = self._load_plan()
                # 3. Restore point: newest-valid predicate decides (§11.2)
                #    — NEVER "start from step 0" when a valid checkpoint exists.
                selection = store.newest_valid()
                start_step = 0
                start_epoch = 0
                if selection.selected is not None and selection.selected.manifest:
                    start_step = int(selection.selected.manifest.get("global_step", 0) or 0)
                    start_epoch = int(selection.selected.manifest.get("epoch", 0) or 0)
                # 3a. Trainer BEFORE READY → RUNNING: resolved fail-closed
                #     (no silent scaffold default) — a refusal leaves the
                #     run READY and untouched, never a faked or dirtied run.
                #     The REAL trainer is selected by run_spec.model and
                #     restored from the newest-valid checkpoint payload
                #     (model/optimizer/RNG — 12 §11.4), so a resume
                #     CONTINUES training instead of restarting it.
                if self.trainer is not None:
                    trainer = self.trainer
                else:
                    spec = self._load_spec()
                    payloads: dict[str, bytes] | None = None
                    if selection.selected is not None:
                        payloads = self._restore_payloads(
                            store, selection.selected.ordinal
                        )
                    # Fine-tune init from the lineage parent — only when
                    # this run is NOT resuming (resume and fine-tune init
                    # are mutually exclusive; resume wins, never both).
                    init_weights = (
                        None if payloads is not None else self._init_weights()
                    )
                    trainer = resolve_trainer(
                        self._runtime_config(),
                        start_step=start_step,
                        start_epoch=start_epoch,
                        model=spec.model,
                        semantic=dict(spec.semantic),
                        train_datasets=list(spec.train_datasets),
                        root=self.root,
                        payloads=payloads,
                        plan=plan,
                        run_dir=self.run_dir,
                        on_progress=self._progress_beater(
                            hb, start_step, start_epoch
                        ),
                        init_weights=init_weights,
                    )
                wf.preflight_pass(self.run_id)  # READY → RUNNING (runtime's step)
                wf.create_execution_segment(
                    self.run_id, capabilities=caps, plan=plan
                )
            elif state == RunState.RUNNING.value:
                if not self._heartbeat_fresh(hb):
                    raise PreconditionFailed(
                        f"run {self.run_id}: state RUNNING but no live heartbeat — "
                        "reconcile first (12 §12.3); refusing to double-start"
                    )
                # Duplicate spawn: a live worker owns the run. Do NOT touch
                # its lease or heartbeat — we merely stand down (the lease
                # token is shared, so "releasing" would evict the owner).
                self._owns = False
                return 0
            else:
                raise PreconditionFailed(
                    f"run {self.run_id}: worker cannot start from state {state} "
                    "(expected READY — train/resume must validate first)"
                )

            tstate = TrainState(start_step, start_epoch, selection.resume_point)

            # 4. First heartbeat (lease renewed with it), then the loop.
            last_beat = hb.beat(global_step=tstate.global_step, state="RUNNING")["ts"]
            return self._loop(wf, store, hb, trainer, tstate, last_beat)
        except WorkerExit as exit_:
            return exit_.code
        except Exception as exc:
            return self._fail(wf, store, hb, lease, exc)
        finally:
            # A worker that still owns the lease must never leave it stale.
            if self._owns:
                try:
                    lease.release(self.run_id, self.session_token)
                except MlforgeError:
                    pass
                hb.clear()
                self._clear_live()

    # -- main loop ------------------------------------------------------------

    def _loop(
        self,
        wf: WorkflowAPI,
        store: CheckpointStore,
        hb: HeartbeatWriter,
        trainer: Trainer,
        tstate: TrainState,
        last_beat: float,
    ) -> int:
        iterations = 0
        while True:
            iterations += 1
            if self.max_iterations is not None and iterations > self.max_iterations:
                raise PreconditionFailed(
                    f"worker {self.run_id}: max_iterations ({self.max_iterations}) reached"
                )

            if hb.due(last_beat, self.heartbeat_interval):
                # renews the lease too — losing it raises
                hb.beat(global_step=tstate.global_step, state="RUNNING")
                last_beat = self.clock()

            ctrl = read_control(self.root, self.run_id)
            if ctrl is not None:
                action = ctrl.get("action")
                if action == "pause":
                    return self._do_pause(wf, store, trainer, tstate, hb)
                if action == "stop":
                    return self._do_stop(wf, hb)
                if action == "checkpoint":
                    # explicit "checkpoint now" (13 §9.6 watch keybinding):
                    # obey once, then forget the intent.
                    clear_control(self.root, self.run_id)
                    if getattr(trainer, "checkpointable", True):
                        self._checkpoint(wf, store, trainer, tstate)
                    else:
                        # e.g. RF-DETR: first epoch still training — there
                        # is no framework state yet. Deferring is honest;
                        # failing a healthy run over a stray keypress is
                        # not (the periodic path checkpoints after step 1).
                        self._write_live({
                            "stage": "TRAINING",
                            "global_step": tstate.global_step,
                            "epoch": tstate.epoch,
                            "note": "checkpoint deferred — first epoch not "
                                    "finished yet",
                        })
                # corrupt/unknown intents are quarantined by read_control;
                # they must not crash training — but they must not be obeyed.

            result = trainer.step(tstate)
            self._append_metric({
                "ts": self.clock(),
                "global_step": result.global_step,
                "epoch": result.epoch,
                "loss": result.loss,
                **result.metrics,
            })
            # Advance the state BEFORE any checkpoint: the manifest and the
            # payload must describe "trained THROUGH result.global_step"
            # (a lagged state meant resume restarted one step behind).
            tstate = TrainState(result.global_step, result.epoch, tstate.resume_from)
            cell = getattr(self, "_progress", None)
            if cell is not None:  # what the trainer's on_progress beats
                cell["global_step"] = tstate.global_step
                cell["epoch"] = tstate.epoch
            self._write_live({
                "stage": "TRAINING",
                "global_step": tstate.global_step,
                "epoch": tstate.epoch,
                "loss": result.loss,
                "metrics": result.metrics,
            })

            # Periodic checkpoint — and ALWAYS on completion: publishing a
            # model reads the newest COMMIT marker, so the final weights
            # must be on disk before wf.complete().
            if result.global_step % self.checkpoint_interval == 0 or result.done:
                self._write_live({
                    "stage": "CHECKPOINTING",
                    "saving": f"checkpoint-{tstate.global_step}",
                    "global_step": tstate.global_step,
                    "epoch": tstate.epoch,
                })
                self._checkpoint(wf, store, trainer, tstate)
                self._write_live({
                    "stage": "TRAINING",
                    "global_step": tstate.global_step,
                    "epoch": tstate.epoch,
                    "loss": result.loss,
                    "metrics": result.metrics,
                })

            if result.done:
                wf.complete(self.run_id)
                self._release(RunLeaseManager(self.root, clock=self.clock), hb)
                self._owns = False
                clear_control(self.root, self.run_id)
                return 0

            self.sleep(self.poll_interval)

    # -- transitions ------------------------------------------------------------

    def _checkpoint(
        self, wf: WorkflowAPI, store: CheckpointStore, trainer: Trainer, tstate: TrainState
    ) -> str:
        """RUNNING → CHECKPOINTING → RUNNING. Write failure ⇒ one retry ⇒
        FAILED[RESUME] with the last-good commit recorded (13 §7)."""
        wf.checkpoint_begin(self.run_id)
        try:
            ordinal, name = self._store_checkpoint(store, trainer, tstate)
        except Exception as exc:
            try:  # rollback attempt: the failed staging write is replaceable
                ordinal, name = self._store_checkpoint(store, trainer, tstate)
            except Exception as exc2:
                wf.checkpoint_fail(
                    self.run_id,
                    f"checkpoint write failed twice: {exc2} (first: {exc})",
                )
                raise WorkerExit(4, "checkpoint failed") from exc2
        wf.checkpoint_commit(self.run_id, ordinal)
        return name

    @staticmethod
    def _store_checkpoint(
        store: CheckpointStore, trainer: Trainer, tstate: TrainState
    ) -> tuple[int, str]:
        nxt = (store.newest_valid().attempted_newest or 0) + 1
        dest = store.write(
            nxt,
            trainer.checkpoint_payload(tstate),
            global_step=tstate.global_step,
            epoch=tstate.epoch,
            components=set(REQUIRED_COMPONENTS),
        )
        return nxt, dest.name

    def _do_pause(
        self,
        wf: WorkflowAPI,
        store: CheckpointStore,
        trainer: Trainer,
        tstate: TrainState,
        hb: HeartbeatWriter,
    ) -> int:
        """Graceful pause: RUNNING → PAUSING → final checkpoint (retry once)
        → PAUSED. Checkpoint failure → FAILED[RESUME] (13 §7)."""
        wf.pause(self.run_id)  # explicit: the control intent IS the user command
        try:
            _ordinal, name = self._store_checkpoint(store, trainer, tstate)
        except Exception as first:
            try:
                _ordinal, name = self._store_checkpoint(store, trainer, tstate)
            except Exception as second:
                wf.pause_checkpoint_failed(
                    self.run_id, f"pause checkpoint failed twice: {second} (first: {first})"
                )
                self._release(RunLeaseManager(self.root, clock=self.clock), hb)
                self._owns = False
                clear_control(self.root, self.run_id)
                return 4
        wf.pause_committed(self.run_id, name)
        self._release(RunLeaseManager(self.root, clock=self.clock), hb)
        self._owns = False
        clear_control(self.root, self.run_id)
        return 0

    def _do_stop(self, wf: WorkflowAPI, hb: HeartbeatWriter) -> int:
        """Graceful stop: RUNNING → STOPPING → STOPPED (no resume)."""
        wf.stop(self.run_id)
        wf.stop_committed(self.run_id, "none")
        self._release(RunLeaseManager(self.root, clock=self.clock), hb)
        self._owns = False
        clear_control(self.root, self.run_id)
        return 0

    # -- helpers -----------------------------------------------------------

    def _heartbeat_fresh(self, hb: HeartbeatWriter) -> bool:
        p = hb.path
        if not p.is_file():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        age = self.clock() - float(data.get("ts", 0.0))
        return age <= self.heartbeat_interval * _DUPLICATE_GRACE

    def _release(self, lease: RunLeaseManager, hb: HeartbeatWriter) -> None:
        try:
            lease.release(self.run_id, self.session_token)
        except MlforgeError:
            pass  # already broken/released — never mask the real result
        hb.clear()
        self._clear_live()

    def _fail(
        self,
        wf: WorkflowAPI,
        store: CheckpointStore,
        hb: HeartbeatWriter,
        lease: RunLeaseManager,
        exc: Exception,
    ) -> int:
        """Map an unexpected failure onto a defined state (13 §7 universal
        rule: no command leaves a run indeterminate)."""
        if isinstance(exc, NotFound):
            return 2
        if isinstance(exc, PreconditionFailed):
            code = 3  # lease/state contract refusal (13 §4.2 exit 3)
        elif isinstance(exc, ValidationBlock):
            code = 1
        else:
            code = 4
        if self._owns:
            state = wf.get_run_state(self.run_id)
            if state == RunState.RUNNING.value:
                has_ckpt = store.newest_valid().selected is not None
                wf.runtime_error(
                    self.run_id,
                    f"{type(exc).__name__}: {exc}",
                    has_valid_checkpoint=has_ckpt,
                )
            elif state == RunState.CHECKPOINTING.value:
                wf.checkpoint_fail(self.run_id, f"{type(exc).__name__}: {exc}")
            # other states (FAILED/PAUSED/COMPLETED/...) are already final
            # or owned by another transition — never overwrite them.
        return code


# ---------------------------------------------------------------------------
# process entry (spawned by the supervisor daemon)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="mlforge.runtime.worker",
        description="MLForge training worker (spawned by the supervisor daemon)",
    )
    p.add_argument("--root", required=True)
    p.add_argument("--run", required=True, dest="run_id")
    p.add_argument("--token", required=True, dest="session_token")
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--heartbeat-interval", type=float, default=None)
    p.add_argument("--poll-interval", type=float, default=None)
    args = p.parse_args(argv)

    cfg: dict[str, Any] = {}
    runtime_json = Path(args.root) / "runs" / args.run_id / "state" / "runtime.json"
    if runtime_json.is_file():
        try:
            cfg = json.loads(runtime_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cfg = {}

    worker = Worker(
        args.root,
        args.run_id,
        args.session_token,
        checkpoint_interval=args.checkpoint_interval
        or int(cfg.get("checkpoint_interval", 5)),
        heartbeat_interval=args.heartbeat_interval
        or float(cfg.get("heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)),
        poll_interval=args.poll_interval
        if args.poll_interval is not None
        else float(cfg.get("poll_interval", 0.5)),
    )
    try:
        return worker.run()
    except MlforgeError as exc:
        print(exc.render(), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
