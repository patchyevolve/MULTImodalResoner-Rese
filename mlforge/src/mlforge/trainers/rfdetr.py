"""RF-DETR trainer — the real detection learning loop for C1/C3
(11_model_weights_and_disk_space §C1: RF-DETR-S/L, §C3: RF-DETR-Seg-S;
12_training_system §10.2 `rf_detr` family + `coco_detection`).

One trainer step == ONE EPOCH, executed by rfdetr's blocking Lightning
fit: `RFDETR.train(epochs=k, resume=<ckpt>)` continues to total epoch k
(Lightning's counters live INSIDE the checkpoint; MLForge's manifest
keeps its own global_step/epoch — 12 §11.4, unchanged by the framework).

Checkpoint representation (§11.4 — the trainer owns the state layout):
  model      = the Lightning `.ckpt` bytes (weights + optimizer_states +
               lr_schedulers + callback states — full resume state)
  others     = storage pointers (`{"kind": "lightning_checkpoint",
               "storage": "model"}`) or real counters — never fake blobs.
  Cross-trainer resume is blocked: `dataloader_state` records
  `{"trainer": "rfdetr", "variant": ...}` and every OTHER trainer
  checks it (or simply cannot parse our model blob) — fail-closed both
  directions (12 §11.4 "never faked into compatibility").

Channels (runtime/trainer.py construction surface):
  payloads      — resume: full framework state written back to
                  `rfdetr_out/last.ckpt`, chained from there
  init_weights  — fine-tune: parent's `model` blob loaded via rfdetr's
                  custom-checkpoint loader (its own class-count
                  alignment expands/trims the detection head honestly)
  run_dir       — REQUIRED scratch: `rfdetr_data/` (materialized
                  dataset) + `rfdetr_out/` (framework checkpoints:
                  last.ckpt chain, checkpoint_<epoch>.ckpt archives,
                  best-model files)
  on_progress   — heartbeat DURING fit (12 §23.2: 30s beats / 120s
                  supervisor timeout — a COCO epoch outlives both).
                  rfdetr exposes no public callback registry, so a
                  Lightning callback is injected through a
                  `rfdetr.training.build_trainer` wrapper installed
                  and removed AROUND each step (import inside
                  `RFDETR.train` resolves it at call time).

runtime knobs (execution namespace, 12 §13.3 — never semantic):
  device · num_workers · eval_interval · checkpoint_interval ·
  early_stopping

Honest limits (refused, never silently substituted):
  * semantic.loss must be `l1` — rfdetr's criterion is its own
    (focal cls + l1 + giou); the declared label is never swapped
  * optimizer `adamw` only (rfdetr's EMA/fused stack assumes it)
  * plan world_size > 1 → PreconditionFailed (no DDP integration)
  * non-COCO layouts / missing valid split → ValidationBlock from the
    materializer (RF-DETR validates every epoch — a train-only run
    cannot complete)
  * scratch disk: one full Lightning ckpt (~490MB for RFDETRSmall:
    31.8M params + Adam states) + interval archives + best-model
    files live under `runs/<id>/rfdetr_out/` (file 11 budgeting)
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import StepResult, TrainState
from mlforge.trainers import resolve_device

#: Model name → rfdetr variant class (identity is the run_spec.model key).
_MODEL_CLASSES: dict[str, str] = {
    "rf_detr_s": "RFDETRSmall",
    "rf_detr_l": "RFDETRLarge",
    "rf_detr_seg_s": "RFDETRSegSmall",
}

#: Semantic values this trainer honors — anything else BLOCKs.
_LOSSES = {"l1"}                     # declared box-regression flavor (12 §10.2)
_OPTIMIZERS = {"adamw"}
_SCHEDULERS = {"cosine", "linear", "constant"}

#: plan.precision_effective → rfdetr TrainConfig.amp_dtype (its SOLE
#: mixed-precision authority; None = full fp32, and Lightning forces
#: 32-true on the CPU accelerator regardless).
_AMP_FOR_PRECISION: dict[str, Any] = {
    "fp32": None,
    "bf16": "bf16",
    "fp16": "fp16",
    "fp8": "fp8",
}

#: Metrics copied from Lightning's callback_metrics into StepResult.
_METRIC_PREFIXES = (
    "train/loss", "train/lr",
    "val/mAP", "val/ema_mAP", "val/loss", "val/ema_loss", "val/mAR",
)

#: Scratch names inside runs/<id>/.
OUT_DIRNAME = "rfdetr_out"
RESUME_STATE_NAME = "_mlforge_resume.json"


@dataclass(frozen=True)
class RfDetrVariant:
    """What would train under this model name (registry identity)."""

    class_name: str
    task: str


#: Trainable RF-DETR models (file 11 §C1/§C3 targets).
MODEL_REGISTRY: dict[str, RfDetrVariant] = {
    "rf_detr_s": RfDetrVariant("RFDETRSmall", "detection"),
    "rf_detr_l": RfDetrVariant("RFDETRLarge", "detection"),
    "rf_detr_seg_s": RfDetrVariant("RFDETRSegSmall", "segmentation"),
}

#: Memoized verdict — importing rfdetr.training pulls pytorch_lightning
#: (seconds); the environment does not change within a process.
_DEP_CACHE: list[Any] = []


def dependency_error() -> str | None:
    """None = rfdetr + its training stack are present here."""
    if _DEP_CACHE:
        return _DEP_CACHE[0]
    err: str | None = None
    try:
        import rfdetr
    except Exception as exc:  # noqa: BLE001 - availability probe: import failure becomes an honest pip-install message
        err = (
            f"rfdetr is not installed — pip install rfdetr "
            f"({type(exc).__name__}: {exc})"
        )
    if err is None:
        try:
            from rfdetr.training import build_trainer  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - availability probe: import failure becomes an honest pip-install message
            err = (
                "rfdetr's training stack is not installed — "
                'pip install "rfdetr[train]" '
                f"({type(exc).__name__}: {exc})"
            )
    if err is None:
        import rfdetr

        missing = [
            cls for cls in _MODEL_CLASSES.values() if not hasattr(rfdetr, cls)
        ]
        if missing:
            err = (
                f"installed rfdetr lacks {', '.join(missing)} — "
                "pip install -U rfdetr"
            )
    _DEP_CACHE.append(err)
    return err


def _pulse_callback(on_progress: Callable[..., Any]):
    """Lightning callback that beats the worker heartbeat from inside
    fit — batch/epoch boundaries are the only hooks rfdetr leaves open
    (no public callback registry). Throttling lives in the worker's
    closure; a tick here is a cheap timestamp check."""
    from pytorch_lightning.callbacks import Callback

    class _Pulse(Callback):
        def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
            on_progress("train_batch_end")

        def on_train_epoch_start(self, *args: Any, **kwargs: Any) -> None:
            on_progress("train_epoch_start")

        def on_validation_batch_end(self, *args: Any, **kwargs: Any) -> None:
            on_progress("validation_batch_end")

        def on_validation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
            on_progress("validation_epoch_start")

    return _Pulse()


class RFDETRTrainer:
    """Epoch-per-step adapter over rfdetr's blocking Lightning fit."""

    def __init__(
        self,
        *,
        model_name: str,
        semantic: Mapping[str, Any],
        runtime: Mapping[str, Any],
        train_datasets: Sequence[str],
        root: Path,
        payloads: Mapping[str, bytes] | None = None,
        start_step: int = 0,
        start_epoch: int = 0,
        plan: Any = None,
        run_dir: Path | None = None,
        on_progress: Callable[..., Any] | None = None,
        init_weights: bytes | None = None,
    ):
        # -- pure config validators FIRST: variant/semantic/plan refusals
        # must surface without importing (or even having) the framework;
        # the environment check below still precedes any side effects.
        variant = MODEL_REGISTRY.get(model_name)
        if variant is None:
            raise PreconditionFailed(
                f"model {model_name!r} has no registered RF-DETR variant",
                hint=f"trainable here: {', '.join(sorted(MODEL_REGISTRY))}",
            )

        # -- semantic identity, honored literally -------------------------
        loss_name = str(semantic.get("loss", "")).strip().lower()
        if loss_name not in _LOSSES:
            raise ValidationBlock(
                f"semantic.loss {loss_name!r} is not what the RF-DETR "
                f"trainer computes (supports: {', '.join(sorted(_LOSSES))}; "
                "the criterion itself is focal cls + l1 + giou)",
                hint="semantic identity is frozen — declare the loss the "
                     "trainer actually optimizes, never a substitute",
            )
        opt_name = str(semantic.get("optimizer", "")).strip().lower()
        if opt_name not in _OPTIMIZERS:
            raise ValidationBlock(
                f"semantic.optimizer {opt_name!r} not supported "
                f"(supports: {', '.join(sorted(_OPTIMIZERS))} — rfdetr's "
                "EMA/fused optimizer stack assumes adamw)",
            )
        sched_name = str(semantic.get("scheduler", "cosine")).strip().lower()
        if sched_name not in _SCHEDULERS:
            raise ValidationBlock(
                f"semantic.scheduler {sched_name!r} not supported "
                f"(supports: {', '.join(sorted(_SCHEDULERS))})",
            )
        try:
            self.epochs = int(semantic["epochs"])
            self.seed = int(semantic["seed"])
            self.global_batch = int(semantic["global_batch"])
            self.learning_rate = float(semantic["learning_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationBlock(
                f"semantic field unreadable for training: {exc}"
            ) from exc
        if self.epochs < 1 or self.global_batch < 1 or self.learning_rate <= 0:
            raise ValidationBlock(
                "semantic requires epochs >= 1, global_batch >= 1, "
                f"learning_rate > 0 (got epochs={self.epochs}, "
                f"global_batch={self.global_batch}, lr={self.learning_rate})",
            )
        self.model_name = model_name
        self.variant = variant
        self.optimizer_name = opt_name
        self.scheduler_name = sched_name

        # -- execution plan (micro/accum must reproduce global_batch) ------
        if plan is not None:
            world = int(getattr(plan, "world_size", 1))
            micro = int(getattr(plan, "micro_batch", self.global_batch))
            accum = int(getattr(plan, "grad_accum", 1))
            precision = str(getattr(plan, "precision_effective", "fp32"))
            if micro * accum * world != self.global_batch:
                raise ValidationBlock(
                    f"execution plan breaks the global-batch invariant: "
                    f"micro {micro} × accum {accum} × world {world} != "
                    f"global_batch {self.global_batch}",
                )
            if world != 1:
                raise PreconditionFailed(
                    f"plan wants world_size {world} — multi-process DDP is "
                    "not integrated in this trainer",
                    hint="train single-process (topology world_size 1); "
                         "rfdetr DDP arrives with its own build step",
                )
        else:
            micro = min(4, self.global_batch)
            accum = (self.global_batch // micro
                     if self.global_batch % micro == 0 else 1)
            if micro * accum != self.global_batch:
                micro, accum = self.global_batch, 1
            precision = str(
                semantic.get("precision_policy", "fp32")
            ).strip().lower()
            if isinstance(semantic.get("precision_policy"), dict):
                precision = str(
                    semantic["precision_policy"].get("preferred", "fp32")
                ).strip().lower()
        if precision not in _AMP_FOR_PRECISION:
            raise ValidationBlock(
                f"plan precision {precision!r} is not one of "
                f"{', '.join(sorted(_AMP_FOR_PRECISION))}",
            )
        self.micro = micro
        self.accum = accum
        self.amp = _AMP_FOR_PRECISION[precision]

        # -- framework check: after the pure validators, before the first
        # side effect (run-dir creation / dataset materialization). -----
        err = dependency_error()
        if err is not None:
            raise PreconditionFailed(
                err,
                hint='install training support: pip install "rfdetr[train]"',
            ) from None
        import rfdetr

        cls = getattr(rfdetr, variant.class_name)

        # -- runtime execution knobs (12 §13.3 — never semantic) -----------
        # Resolved through the shared contract: auto → this host's best
        # device, cuda requested on a GPU-less host refuses HERE (before
        # any side effect) — never a silent downgrade.
        self.device = resolve_device(runtime)
        self.num_workers = int(runtime.get("num_workers", 2))
        self.eval_interval = int(runtime.get("eval_interval", 1))
        self.checkpoint_interval = max(
            1, int(runtime.get("checkpoint_interval", 10))
        )
        self.early_stopping = bool(runtime.get("early_stopping", False))
        self.on_progress = on_progress
        self.start_step = int(start_step)
        self.start_epoch = int(start_epoch)

        if run_dir is None:
            raise ValidationBlock(
                "RF-DETR trainer needs a run directory for scratch state "
                "(dataset materialization + framework checkpoints)",
                hint="it is built by the worker via "
                     "resolve_trainer(run_dir=runs/<id>) — ad-hoc "
                     "construction must supply one explicitly",
            )
        self.run_dir = Path(run_dir)
        self.out_dir = self.run_dir / OUT_DIRNAME
        self.out_dir.mkdir(parents=True, exist_ok=True)
        root = Path(root)

        # -- dataset BEFORE weights (a bad dataset must refuse first) ------
        from mlforge.trainers.rfdetr_data import materialize

        self.data = materialize(root, self.run_dir, train_datasets)

        # -- init: resume (full) > fine-tune (weights) > published base -----
        if payloads and init_weights is not None:
            raise ValidationBlock(
                "resume payloads and fine-tune init weights are mutually "
                "exclusive — resume from THIS run's checkpoint or "
                "initialize from the parent's weights, never both",
            )
        pretrain: str | None = None
        trust = False
        if payloads:
            missing = [c for c in REQUIRED_COMPONENTS if c not in payloads]
            if missing:
                raise ValidationBlock(
                    f"checkpoint payload incomplete — missing: "
                    f"{', '.join(missing)} (not resumable, 12 §11.4)",
                )
            self._assert_own_payload(payloads)
            (self.out_dir / "last.ckpt").write_bytes(payloads["model"])
            self._write_state(
                {"path": str((self.out_dir / "last.ckpt").resolve()),
                 "completed": self.start_epoch}
            )
            pretrain = None  # ckpt_path restores real state — no reload
        elif init_weights is not None:
            init_path = self.out_dir / "init_weights.ckpt"
            init_path.write_bytes(init_weights)
            pretrain = str(init_path)
            # The bytes are the parent's COMMITTED checkpoint component:
            # integrity was verified by the checkpoint store + gate (§11.3
            # / §15.3) — ours to load, never a foreign pickle.
            trust = True
            self._write_state({"path": None, "completed": 0})
        else:
            self._reset_scratch()  # no valid checkpoint ⇒ step 0, cleanly
            self._write_state({"path": None, "completed": 0})

        with warnings.catch_warnings():
            if payloads is not None:
                # resuming with pretrain_weights=None is intentional —
                # ckpt_path restores the real state right after.
                warnings.filterwarnings(
                    "ignore",
                    message=".*was instantiated with pretrain_weights=None.*",
                )
            if pretrain is None and not payloads:
                self._prefetch_published_base(cls)
                self._model = cls()
            else:
                self._model = cls(
                    pretrain_weights=pretrain, trust_checkpoint=trust
                )
        # Nothing serializable until an epoch's fit writes a framework
        # checkpoint: a resumed run still holds its payload's last.ckpt,
        # a fresh/fine-tuned one must train first (the worker's
        # control-intent checkpoint skips us until then).
        self.checkpointable = payloads is not None

    # -- state plumbing ----------------------------------------------------

    @property
    def _state_path(self) -> Path:
        return self.out_dir / RESUME_STATE_NAME

    def _write_state(self, state: dict[str, Any]) -> None:
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        tmp.replace(self._state_path)

    def _read_state(self) -> dict[str, Any]:
        if not self._state_path.is_file():
            return {"path": None, "completed": 0}
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationBlock(
                f"rfdetr resume state unreadable: {self._state_path}: {exc}"
            ) from exc
        return {
            "path": data.get("path"),
            "completed": int(data.get("completed", 0)),
        }

    def _reset_scratch(self) -> None:
        """No valid checkpoint ⇒ start from step 0: stale framework files
        from an interrupted earlier attempt must not be chained (the
        manifest counters are the authority — 12 §11.2)."""
        for pattern in ("*.ckpt", "*.pth"):
            for p in self.out_dir.glob(pattern):
                p.unlink(missing_ok=True)
        self._state_path.unlink(missing_ok=True)

    def _assert_own_payload(self, payloads: Mapping[str, bytes]) -> None:
        """Cross-trainer resume blocked CHEAPLY (before a 490MB load):
        our dataloader_state names trainer+variant; text/reid/harness
        payloads do not — fail-closed in every direction (§11.4)."""
        try:
            ds = json.loads(
                payloads.get("dataloader_state", b"{}").decode("utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            ds = {}
        if ds.get("trainer") != "rfdetr" or ds.get("variant") != self.model_name:
            raise ValidationBlock(
                f"checkpoint payload was not produced by trainer "
                f"'rfdetr' for model {self.model_name!r} — a resume "
                "continues the SAME experiment (12 §11.4), never a "
                "different trainer's state",
            )

    def _prefetch_published_base(self, cls: Any) -> None:
        """Download the variant's published base at CONSTRUCTION — a
        network failure then leaves the run in READY (refused before
        READY → RUNNING), not FAILED mid-run."""
        try:
            from rfdetr.assets.model_weights import (
                download_pretrain_weights,
                get_model_cache_dir,
            )
        except ImportError as exc:  # rfdetr moved/renamed its downloader
            raise PreconditionFailed(
                f"rfdetr's published-weight downloader is missing "
                f"({exc})",
                hint="pip install -U rfdetr",
            ) from exc

        import rfdetr.config as rconfig

        config_cls = getattr(
            rconfig, f"{cls.__name__}Config", None
        )
        if config_cls is None:
            return  # variant without a published base (scratch is the point)
        default = getattr(
            config_cls.model_fields.get("pretrain_weights"), "default", None
        )
        if not default:
            return
        # Mirror rfdetr's own resolution: a bare filename goes to the model
        # cache (RF_HOME) — never the working directory, so no stray
        # multi-hundred-MB .pth in the user's project and no second copy
        # when the model itself resolves the same default.
        target = str(default)
        if not os.path.dirname(target):
            cache_dir = get_model_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            target = os.path.join(cache_dir, target)
        try:
            download_pretrain_weights(target)
        except Exception as exc:
            raise PreconditionFailed(
                f"could not fetch the published base weights "
                f"{target!r}: {exc}",
                hint="place the file in rfdetr's model cache (RF_HOME) — "
                     "the network fetch failed and is never silently "
                     "replaced by random init",
            ) from exc

    # -- training ----------------------------------------------------------

    def _train_config(self, target: int, resume: str | None) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "epochs": target,
            "dataset_dir": str(self.data.dir),
            "output_dir": str(self.out_dir),
            "dataset_file": "roboflow",
            "batch_size": self.micro,
            "grad_accum_steps": self.accum,
            "lr": self.learning_rate,
            "optimizer": self.optimizer_name,
            "lr_scheduler": self.scheduler_name,
            "seed": self.seed,
            "amp_dtype": self.amp,
            "tensorboard": False,
            "progress_bar": None,
            "num_workers": self.num_workers,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "early_stopping": self.early_stopping,
        }
        if resume:
            cfg["resume"] = str(resume)
        if self.device:
            cfg["device"] = str(self.device)
        return cfg

    def _newest_ckpt(self, target: int, before_mtime: int) -> Path:
        """Full-state checkpoint produced by THIS epoch's fit.

        Epoch archives (`checkpoint_<N>.ckpt`, N zero-based) are
        unambiguous; `last.ckpt` is preferred only when this fit rewrote
        it (mtime moved) — with checkpoint_interval=1 the archive is the
        sole writer, and a stale ctor-written last.ckpt must never be
        re-chained (it would loop one epoch forever)."""
        exact = self.out_dir / f"checkpoint_{target - 1}.ckpt"
        if exact.is_file():
            return exact
        last = self.out_dir / "last.ckpt"
        if last.is_file() and last.stat().st_mtime_ns > before_mtime:
            return last
        raise PreconditionFailed(
            f"rfdetr finished epoch {target} but wrote no resumable "
            f"checkpoint under {self.out_dir} — refusing to chain "
            "a state we cannot prove",
        )

    def _extract_metrics(self, trainer: Any) -> tuple[dict[str, Any], float | None]:
        out: dict[str, Any] = {}
        if trainer is None:
            return out, None
        raw = getattr(trainer, "callback_metrics", {}) or {}
        for key, value in raw.items():
            key = str(key)
            if not any(
                key == p or key.startswith(p) for p in _METRIC_PREFIXES
            ):
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(num):
                continue
            out[key] = round(num, 6)
        loss = out.pop("train/loss", None)
        return out, loss

    def step(self, state: TrainState) -> StepResult:
        if state.epoch >= self.epochs:  # counters say: already finished
            return StepResult(
                state.global_step, state.epoch, loss=None, done=True,
            )
        target = state.epoch + 1
        st = self._read_state()
        resume = st.get("path")
        if resume is not None and not Path(str(resume)).is_file():
            raise ValidationBlock(
                f"framework resume checkpoint missing: {resume} — "
                "run scratch state was tampered with",
            )
        last = self.out_dir / "last.ckpt"
        before_mtime = last.stat().st_mtime_ns if last.is_file() else 0

        # Install the build_trainer seam ONLY around this fit (import
        # inside RFDETR.train resolves the attribute at call time).
        import rfdetr.training as rtrain

        captured: dict[str, Any] = {}
        original = rtrain.build_trainer

        def _wrapper(tc: Any, mc: Any, **kw: Any) -> Any:
            trainer = original(tc, mc, **kw)
            captured["trainer"] = trainer
            if self.on_progress is not None:
                trainer.callbacks.append(_pulse_callback(self.on_progress))
            return trainer

        rtrain.build_trainer = _wrapper
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Calling train\\(\\) on a model that has "
                            "already been trained",
                )
                self._model.train(
                    **self._train_config(target, str(resume) if resume else None)
                )
        finally:
            rtrain.build_trainer = original

        new_resume = self._newest_ckpt(target, before_mtime)
        self._write_state(
            {"path": str(new_resume.resolve()), "completed": target}
        )
        self.checkpointable = True
        metrics, loss = self._extract_metrics(captured.get("trainer"))
        return StepResult(
            global_step=state.global_step + 1,
            epoch=target,
            loss=loss,
            metrics=metrics,
            done=target >= self.epochs,
        )

    def checkpoint_payload(self, state: TrainState) -> dict[str, bytes]:
        """§11.4: real state — the consolidated Lightning ckpt plus
        honest pointers/counters (the trainer owns the representation;
        an incomplete payload is caught by recovery, never faked)."""
        st = self._read_state()
        path = st.get("path")
        if not path or not Path(str(path)).is_file():
            raise ValidationBlock(
                "no framework checkpoint yet — the first epoch is "
                "still training (RF-DETR serializes only after an epoch "
                "completes); retry after epoch 1 or use stop (13 §9)",
            )
        blob = Path(str(path)).read_bytes()

        def _json(obj: Any) -> bytes:
            return json.dumps(obj, sort_keys=True).encode("utf-8")

        pointer = {
            "kind": "lightning_checkpoint",
            "storage": "model",
            "file": Path(str(path)).name,
        }
        return {
            "model": blob,
            "optimizer": _json(pointer),
            "lr_scheduler": _json(
                {**pointer, "scheduler": self.scheduler_name}
            ),
            "amp_scaler": _json({
                "enabled": self.amp == "fp16",
                "storage": "model",
            }),
            "ema": _json({"enabled": True, "storage": "model"}),
            "global_step": _json(int(state.global_step)),
            "epoch": _json(int(state.epoch)),
            "batch_position": _json({
                "position_in_epoch": 1,
                "steps_per_epoch": 1,
                "note": "one step == one Lightning epoch (fit per epoch)",
            }),
            "sampler_state": _json({
                "protocol": "rfdetr seed_everything(seed) + Lightning "
                            "epoch loop",
                "seed": int(self.seed),
                "epoch": int(state.epoch),
            }),
            "dataloader_state": _json({
                "trainer": "rfdetr",
                "variant": self.model_name,
                "layout": self.data.layout,
                "materialized": str(
                    self.data.dir.relative_to(self.run_dir)
                ),
                "train_images": int(self.data.train_images),
                "valid_images": int(self.data.valid_images),
                "classes": int(self.data.classes),
                "categories": list(self.data.categories),
                "batch_size": int(self.micro),
                "grad_accum": int(self.accum),
                "epochs": int(self.epochs),
            }),
            "distributed_state": _json({"world_size": 1}),
            "grad_accum_state": _json({
                "micro_batch": int(self.micro),
                "grad_accum": int(self.accum),
                "micro_in_step": 0,  # checkpoints land on epoch boundaries
            }),
            "early_stopping_state": _json({
                "enabled": bool(self.early_stopping),
            }),
            "best_model_state": _json(
                {"enabled": True, "storage": "model"}
            ),
            "rng_hierarchy": _json({
                "enabled": False,
                "reason": "rfdetr/Lightning persists loop, optimizer and "
                          "callback state in the checkpoint; sample order "
                          "derives from seed via seed_everything",
                "seed": int(self.seed),
            }),
        }


#: Registry hook consumed by mlforge.trainers.build_trainer.
TRAINER_CLASS = RFDETRTrainer


__all__ = [
    "MODEL_REGISTRY",
    "OUT_DIRNAME",
    "TRAINER_CLASS",
    "RFDETRTrainer",
    "RfDetrVariant",
    "dependency_error",
]
