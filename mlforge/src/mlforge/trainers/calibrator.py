"""C5 Calibrator trainer — temperature scaling + conformal thresholds
(+ optional confidence-decomposition weights). CPU, stdlib-only.

Spec: 10_training_plan/03_training_pipeline.md Model 5 (temperature on
NLL via bounded search, conformal quantile ceil((n+1)(1-alpha))/n,
decomposition weights) + 02_architecture/06_calibration/ (the five
component names and the weighted-combination formula).

Fits are closed-form/search — there is no gradient optimizer and no
learning-rate schedule here, so payload `optimizer`/`lr_scheduler`
components honestly say `enabled: false` (state lives in `model`).
Being dependency-free, this trainer runs on ANY machine:
`dependency_error()` is always None.

Training contract (validated literally, never silently adapted):
  * semantic.loss       = nll                 (temperature objective)
  * semantic.optimizer  = none | search       (no SGD state to save)
  * semantic.scheduler  = constant | none     (no schedule exists)
  * semantic.epochs     = fit phases: 2 (temperature, conformal) or
                          3 with decomposition: true — the run schema's
                          epochs field IS the phase count, checked both
                          ways so a mismatch BLOCKs with guidance
  * semantic.alphas     = conformal error levels (default [0.05, 0.10])
  * semantic.decomposition = bool (default false) — opt-in third phase
                          that requires every record to carry the five
                          component scores (06 §1)
  * semantic.seed / global_batch / learning_rate are REQUIRED by the
    run schema (12 §17) but are not levers for a closed-form fit —
    recorded in dataloader_state as such, never pretended

Honest limits (fail-closed):
  * fine-tune (init_weights)  -> ValidationBlock: nothing warm-starts;
    refit on this calibration set (train/fork), provenance is the spec
  * < 2 classes or < 2 rows   -> ValidationBlock
  * decomposition without both correct AND incorrect predictions ->
    ValidationBlock (weights fit against a constant target is a lie)
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import StepResult, TrainState


def dependency_error() -> str | None:
    """None = runnable here (stdlib-only, by design)."""
    return None


@dataclass(frozen=True)
class CalibratorArch:
    """What this trainer fits, keyed by model name (semantic identity)."""

    fits: tuple[str, ...] = ("temperature", "conformal", "decomposition")
    alpha_default: tuple[float, ...] = (0.05, 0.10)


#: Trainable calibration models (12 §10.2 `calibrator`).
MODEL_REGISTRY: dict[str, CalibratorArch] = {
    "calibrator": CalibratorArch(),
}

#: Semantic values this trainer honors — anything else BLOCKs.
_LOSSES = {"nll"}
_OPTIMIZERS = {"none", "search"}
_SCHEDULERS = {"constant", "none"}

#: Decomposition component names (06 §1 weighted-combination formula).
COMPONENTS: tuple[str, ...] = (
    "perception", "temporal", "motion", "cross_modal_agreement", "reasoning",
)

_PHASES_BASE = ("temperature", "conformal")


def _parse_flag(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        flag = value.strip().lower()
        if flag in ("true", "yes", "1", "on"):
            return True
        if flag in ("false", "no", "0", "off", ""):
            return False
    raise ValidationBlock(
        f"semantic.{key} {value!r} is not understood (use true or false)")


def load_prediction_rows(
    root: Path, train_datasets: Sequence[str], *, need_components: bool
) -> list[dict[str, Any]]:
    """Prepared `calibration` store artifact(s) → prediction rows.

    Mirrors gate semantics locally (registered, version-exact,
    store-backed, transform-exact) — the trainer never trusts a
    differently-prepared artifact."""
    from mlforge.ingest.identity import parse_ref
    from mlforge.store import ContentStore

    if not train_datasets:
        raise ValidationBlock("run_spec.train_datasets is empty")
    store = ContentStore(Path(root) / "store")
    rows: list[dict[str, Any]] = []
    classes: set[int] = set()
    for ref in train_datasets:
        name, version = parse_ref(ref)
        reg_path = Path(root) / "datasets" / name / "identity.json"
        if not reg_path.is_file():
            raise PreconditionFailed(
                f"dataset {name!r} not registered — mlforge dataset add "
                f"{name} <PATH>",
            )
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        reg_version = reg.get("version")
        if reg_version and str(reg_version) != version:
            raise ValidationBlock(
                f"{ref}: registered version {reg_version!r} != {version!r} "
                "(versions are identity, never reinterpreted)",
            )
        identity = str(reg.get("identity") or "")
        if not identity or not store.contains(identity):
            raise PreconditionFailed(
                f"{ref} is not a prepared store artifact — "
                "run `mlforge prepare <model>` first",
            )
        doc = json.loads(store.get_bytes(identity, verify=True).decode("utf-8"))
        schema = str((doc.get("transform") or {}).get("name") or "")
        if schema and schema != "calibration":
            raise ValidationBlock(
                f"{ref} was prepared with transform {schema!r} — the "
                "calibrator needs `calibration` prediction records "
                f"(logits + label), not {schema!r} rows",
            )
        for rec in doc.get("records") or []:
            if not isinstance(rec, dict):
                continue
            logits = rec.get("logits")
            if not isinstance(logits, list) or not logits:
                continue
            rows.append(rec)
            classes.add(len(logits))
    if not rows:
        raise ValidationBlock(
            "prepared dataset(s) contain no prediction records — prepare "
            "with transform `calibration` ({logits|probs, label} rows)",
        )
    if len(classes) > 1:
        raise ValidationBlock(
            f"class count differs across prepared datasets "
            f"({sorted(classes)}) — one calibration pool is one label "
            "space",
        )
    n_classes = next(iter(classes))
    if n_classes < 2:
        raise ValidationBlock(
            "calibrating a single class is meaningless — need >= 2 "
            "logit dimensions",
        )
    if len(rows) < 2:
        raise ValidationBlock(
            f"need at least 2 prediction rows to fit (got {len(rows)})",
        )
    for i, rec in enumerate(rows):
        lab = rec.get("label")
        if isinstance(lab, bool) or not isinstance(lab, (int, float)) \
                or int(lab) != lab or not 0 <= int(lab) < n_classes:
            raise ValidationBlock(
                f"prediction row {i}: label {lab!r} is not a valid class "
                f"index for {n_classes} classes",
            )
        rec["label"] = int(lab)
        if any(not isinstance(v, (int, float)) or not math.isfinite(float(v))
               for v in rec["logits"]):
            raise ValidationBlock(
                f"prediction row {i}: logits must be finite numbers")
        rec["logits"] = [float(v) for v in rec["logits"]]
        if need_components:
            comp = rec.get("components")
            if not isinstance(comp, dict) or any(c not in comp
                                                 for c in COMPONENTS):
                raise ValidationBlock(
                    f"prediction row {i}: decomposition: true needs "
                    f"`components` with exactly {', '.join(COMPONENTS)} "
                    "(06 §1) — regenerate predictions with component "
                    "scores or set decomposition: false",
                )
    if need_components:
        if len(rows) < 8:
            raise ValidationBlock(
                f"decomposition needs >= 8 rows to fit 5 weights "
                f"(got {len(rows)})",
            )
        outcomes = {
            int(max(range(n_classes),
                    key=lambda k: rows[i]["logits"][k]) == rows[i]["label"])
            for i in range(len(rows))
        }
        if outcomes != {0, 1}:
            raise ValidationBlock(
                "decomposition: true needs BOTH correct and incorrect "
                "predictions to fit weights against (found only "
                f"{sorted(outcomes)}) — weights against a constant "
                "target cannot be learned",
            )
    return rows


# -- fits (pure python, deterministic) --------------------------------------


def _probs_at(logits: Sequence[Sequence[float]], t: float) -> list[list[float]]:
    out: list[list[float]] = []
    inv = 1.0 / t
    for row in logits:
        scaled = [v * inv for v in row]
        m = max(scaled)
        exps = [math.exp(v - m) for v in scaled]
        total = sum(exps)
        out.append([e / total for e in exps])
    return out


def _nll(rows: Sequence[Mapping[str, Any]], t: float) -> float:
    total = 0.0
    for rec in rows:
        probs = _probs_at([rec["logits"]], t)[0]
        total -= math.log(max(probs[rec["label"]], 1e-300))
    return total / len(rows)


def _bin_gaps(rows: Sequence[Mapping[str, Any]], t: float,
              n_bins: int = 15):
    """Per non-empty (lo, hi] confidence bin: (count, |acc - conf|).

    The shared 15-bin partition behind BOTH `_ece` (weighted mean of the
    gaps) and `_mce` (the worst gap) — one binning, two spec metrics
    (06 §4: `compute_ece`/`compute_mce(..., n_bins=15)`)."""
    probs = _probs_at([r["logits"] for r in rows], t)
    confidences = [max(p) for p in probs]
    predictions = [max(range(len(p)), key=lambda k: p[k]) for p in probs]
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, c in enumerate(confidences) if lo < c <= hi]
        if not idx:
            continue
        conf = sum(confidences[j] for j in idx) / len(idx)
        acc = sum(1 for j in idx
                  if predictions[j] == rows[j]["label"]) / len(idx)
        yield len(idx), abs(acc - conf)


def _ece(rows: Sequence[Mapping[str, Any]], t: float, n_bins: int = 15) -> float:
    """Expected Calibration Error — spec formula (13 §Model 5, 06:348):
    equal confidence bins (lo, hi], weighted |accuracy - confidence|."""
    n = len(rows)
    return sum((cnt / n) * gap for cnt, gap in _bin_gaps(rows, t, n_bins))


def _mce(rows: Sequence[Mapping[str, Any]], t: float, n_bins: int = 15) -> float:
    """Maximum Calibration Error — spec (06:317 formula, 06:349 target
    < 0.10): the WORST bin's |accuracy - confidence| under the same
    15-bin scheme as `_ece` (`compute_mce(confidences, labels, n_bins)`)."""
    gaps = [gap for _cnt, gap in _bin_gaps(rows, t, n_bins)]
    return max(gaps) if gaps else 0.0


def _brier(rows: Sequence[Mapping[str, Any]], t: float) -> float:
    """Brier score — spec (06:320 formula, 06:350 target < 0.15):
    mean over rows of sum over classes of (p_c - 1[c == label])^2 at
    the fitted temperature (the multiclass form; the spec pseudo-code
    passes confidences only, the harness has the full vector)."""
    probs = _probs_at([r["logits"] for r in rows], t)
    total = 0.0
    for p, rec in zip(probs, rows):
        y = rec["label"]
        total += sum((pk - (1.0 if k == y else 0.0)) ** 2
                     for k, pk in enumerate(p))
    return total / len(rows)


def _fit_temperature(rows: Sequence[Mapping[str, Any]]) -> float:
    """Bounded search for T minimizing NLL — coarse grid (deterministic
    seeds the bracket) then golden-section refine, stdlib-only."""
    lo, hi = 0.1, 10.0
    grid_n = 99
    best_t, best_v = lo, _nll(rows, lo)
    for i in range(1, grid_n):
        t = lo + (hi - lo) * i / (grid_n - 1)
        v = _nll(rows, t)
        if v < best_v:
            best_t, best_v = t, v
    step = (hi - lo) / (grid_n - 1)
    a = max(lo, best_t - step)
    b = min(hi, best_t + step)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = _nll(rows, c), _nll(rows, d)
    while b - a > 1e-4:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = _nll(rows, c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = _nll(rows, d)
    return (a + b) / 2.0


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default method)."""
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    idx = q * (n - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _fit_conformal(
    rows: Sequence[Mapping[str, Any]], t: float, alphas: Sequence[float]
) -> tuple[dict[float, float], dict[float, float]]:
    """Spec rule: threshold = quantile(ceil((n+1)(1-alpha))/n) over
    nonconformity 1 - p(true class) at the fitted temperature."""
    probs = _probs_at([r["logits"] for r in rows], t)
    scores = sorted(1.0 - probs[i][rows[i]["label"]] for i in range(len(rows)))
    n = len(scores)
    thresholds: dict[float, float] = {}
    coverages: dict[float, float] = {}
    for alpha in alphas:
        level = math.ceil((n + 1) * (1.0 - alpha)) / n
        thr = _quantile(scores, min(level, 1.0))
        thresholds[alpha] = thr
        coverages[alpha] = sum(1 for s in scores if s <= thr) / n
    return thresholds, coverages


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (n is 5)."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValidationBlock(
                "decomposition design matrix is singular — component "
                "scores carry no signal (constant columns?)")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _fit_decomposition(
    rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, float], float]:
    """Least-squares weights over the five component scores so the
    weighted combination predicts empirical correctness (06 §1)."""
    design = [[float(r["components"][c]) for c in COMPONENTS] for r in rows]
    target = [
        1.0 if max(range(len(r["logits"])),
                   key=lambda k: r["logits"][k]) == r["label"] else 0.0
        for r in rows
    ]
    k = len(COMPONENTS)
    gram = [[sum(design[i][a] * design[i][b] for i in range(len(design)))
             + (1e-8 if a == b else 0.0)
             for b in range(k)] for a in range(k)]
    rhs = [sum(design[i][a] * target[i] for i in range(len(design)))
           for a in range(k)]
    weights = _solve(gram, rhs)
    residual = sum(
        (sum(weights[j] * design[i][j] for j in range(k)) - target[i]) ** 2
        for i in range(len(design))
    ) / len(design)
    return dict(zip(COMPONENTS, weights)), residual


class CalibratorTrainer:
    """Temperature + conformal (+ decomposition) fits — real numbers,
    real state, honest disabled components."""

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
        run_dir: str | Path | None = None,  # no scratch: fits are closed-form
        on_progress: Any = None,            # phases finish within heartbeat
        init_weights: bytes | None = None,
    ):
        err = dependency_error()
        if err is not None:  # pragma: no cover — stdlib-only today
            raise PreconditionFailed(err)
        arch = MODEL_REGISTRY.get(model_name)
        if arch is None:
            raise PreconditionFailed(
                f"model {model_name!r} has no registered calibrator",
                hint=f"trainable here: {', '.join(sorted(MODEL_REGISTRY))}",
            )
        # -- semantic identity, honored literally -------------------------
        loss_name = str(semantic.get("loss", "")).strip().lower()
        if loss_name not in _LOSSES:
            raise ValidationBlock(
                f"semantic.loss {loss_name!r} is not the calibrator's "
                f"objective (supports: {', '.join(sorted(_LOSSES))}) — "
                "temperature scaling minimizes NLL, never a substitute",
            )
        opt_name = str(semantic.get("optimizer", "")).strip().lower()
        if opt_name not in _OPTIMIZERS:
            raise ValidationBlock(
                f"semantic.optimizer {opt_name!r} not supported "
                f"(supports: {', '.join(sorted(_OPTIMIZERS))}) — a "
                "closed-form fit has no gradient optimizer state",
            )
        sched_name = str(semantic.get("scheduler", "")).strip().lower()
        if sched_name not in _SCHEDULERS:
            raise ValidationBlock(
                f"semantic.scheduler {sched_name!r} not supported "
                f"(supports: {', '.join(sorted(_SCHEDULERS))})",
            )
        precision = str(semantic.get("precision_policy", "")).strip().lower()
        if precision != "fp32":
            raise ValidationBlock(
                f"semantic.precision_policy {precision!r} — fits run in "
                "python floats; declare fp32 (never pretend an AMP mode "
                "a closed-form search does not have)",
            )
        self.decomposition = _parse_flag(
            semantic.get("decomposition", False), "decomposition")
        phases = list(_PHASES_BASE)
        if self.decomposition:
            phases.append("decomposition")
        try:
            self.epochs = int(semantic["epochs"])
            self.seed = int(semantic["seed"])
            self.global_batch = int(semantic["global_batch"])
            self.learning_rate = float(semantic["learning_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationBlock(
                f"semantic field unreadable for training: {exc}") from exc
        if self.epochs != len(phases):
            raise ValidationBlock(
                f"semantic.epochs {self.epochs} != {len(phases)} fit "
                f"phases ({', '.join(phases)}) — epochs IS the phase "
                f"count here: set epochs: {len(phases)}"
                + ("" if self.decomposition
                   else " (or epochs: 3 with decomposition: true)"),
            )
        if self.global_batch < 1 or self.learning_rate <= 0:
            raise ValidationBlock(
                "semantic requires global_batch >= 1 and learning_rate > 0 "
                f"(got global_batch={self.global_batch}, "
                f"lr={self.learning_rate})",
            )
        raw_alphas = semantic.get("alphas", list(arch.alpha_default))
        if not isinstance(raw_alphas, (list, tuple)) or not raw_alphas:
            raise ValidationBlock(
                "semantic.alphas must be a non-empty list of conformal "
                "error levels (e.g. [0.05, 0.10])",
            )
        try:
            self.alphas: tuple[float, ...] = tuple(float(a) for a in raw_alphas)
        except (TypeError, ValueError) as exc:
            raise ValidationBlock(
                f"semantic.alphas unreadable: {exc}") from exc
        if any(not 0.0 < a < 1.0 for a in self.alphas):
            raise ValidationBlock(
                f"semantic.alphas must each be in (0, 1) — got "
                f"{list(self.alphas)}",
            )
        self.phases = tuple(phases)
        self.model_name = model_name
        self.arch = arch

        # -- execution plan (world/precision/batch identity, 12 §12) ------
        if plan is not None:
            world = int(getattr(plan, "world_size", 1))
            micro = int(getattr(plan, "micro_batch", self.global_batch))
            accum = int(getattr(plan, "grad_accum", 1))
            precision_eff = str(getattr(plan, "precision_effective", "fp32"))
            if micro * accum * world != self.global_batch:
                raise ValidationBlock(
                    f"execution plan breaks the global-batch invariant: "
                    f"micro {micro} × accum {accum} × world {world} != "
                    f"global_batch {self.global_batch}",
                )
            if world != 1:
                raise PreconditionFailed(
                    f"plan wants world_size {world} — fitting the "
                    "calibrator is single-process by design",
                    hint="run single-process (topology world_size 1)",
                )
            if precision_eff != "fp32":
                raise PreconditionFailed(
                    f"plan precision {precision_eff!r} — closed-form fits "
                    "have no AMP (the payload would be a lie)",
                    hint="semantic.precision_policy: fp32",
                )
            self.micro, self.accum = micro, accum
        else:
            self.micro, self.accum = self.global_batch, 1

        # -- data ----------------------------------------------------------
        self.rows = load_prediction_rows(
            Path(root), train_datasets, need_components=self.decomposition)
        self.n = len(self.rows)
        self.n_classes = len(self.rows[0]["logits"])
        self.steps_per_epoch = 1  # one phase per step (epoch = phase)

        # -- init: resume (full state) > fresh; fine-tune is refused ------
        if payloads and init_weights is not None:
            raise ValidationBlock(
                "resume payloads and fine-tune init weights are mutually "
                "exclusive — resume from THIS run's checkpoint or start "
                "a fresh fit, never both",
            )
        if payloads:
            missing = [c for c in REQUIRED_COMPONENTS if c not in payloads]
            if missing:
                raise ValidationBlock(
                    f"checkpoint payload incomplete — missing: "
                    f"{', '.join(missing)} (not resumable, 12 §11.4)",
                )
            self._restore(dict(payloads))
        elif init_weights is not None:
            raise ValidationBlock(
                "calibrator fine-tune refused — temperature/conformal "
                "fits are closed-form from THIS calibration set; there "
                "is no weight warm-start to inherit",
                hint="train fresh (`mlforge train`) or fork the run to "
                     "record lineage — combine calibration sets in the "
                     "dataset instead of fine-tuning",
            )
        else:
            self.temperature = 1.0
            self.thresholds: dict[float, float] = {}
            self.coverages: dict[float, float] = {}
            self.weights: dict[str, float] | None = None
            self.ece: float | None = None
            self.nll_value: float | None = None
            random.seed(self.seed)
        self.start_step = int(start_step)
        self.start_epoch = int(start_epoch)

    # -- restore ------------------------------------------------------------

    def _restore(self, payloads: dict[str, bytes]) -> None:
        try:
            state = json.loads(payloads["model"].decode("utf-8"))
            if str(state.get("trainer")) != "calibrator":
                raise ValidationBlock(
                    "checkpoint payload was not produced by trainer "
                    "'calibrator' — a resume continues the SAME "
                    "experiment (12 §11.4), never another trainer's state",
                )
            if int(state.get("classes", -1)) != self.n_classes:
                raise ValidationBlock(
                    f"checkpoint classes {state.get('classes')} != this "
                    f"dataset's {self.n_classes} — label space changed; "
                    "a resume never reinterprets the pool",
                )
            if int(state.get("n", -1)) != self.n:
                raise ValidationBlock(
                    f"checkpoint fit {state.get('n')} rows != this "
                    f"dataset's {self.n} — a resume continues the SAME "
                    "calibration pool",
                )
            rec_alphas = [float(a) for a in state.get("alphas", [])]
            if rec_alphas != list(self.alphas):
                raise ValidationBlock(
                    f"checkpoint alphas {rec_alphas} != semantic.alphas "
                    f"{list(self.alphas)} — same experiment only",
                )
            if bool(state.get("decomposition")) != self.decomposition:
                raise ValidationBlock(
                    "checkpoint decomposition flag differs from semantic — "
                    "a resume continues the SAME experiment",
                )
            self.temperature = float(state["temperature"])
            if not math.isfinite(self.temperature) or self.temperature <= 0:
                raise ValueError("temperature not a positive finite float")
            # alpha keys round-trip through JSON as strings
            self.thresholds = {float(k): float(v) for k, v in
                               state.get("thresholds", {}).items()}
            self.coverages = {float(k): float(v) for k, v in
                              state.get("coverages", {}).items()}
            raw_w = state.get("weights")
            self.weights = ({str(k): float(v) for k, v in raw_w.items()}
                            if raw_w else None)
            self.ece = (float(state["ece"])
                        if state.get("ece") is not None else None)
            self.nll_value = (float(state["nll"])
                              if state.get("nll") is not None else None)
            rng = json.loads(payloads["rng_hierarchy"].decode("utf-8"))
            py = rng.get("python")
            if py is not None:
                random.setstate((py[0], tuple(py[1]), py[2]))
        except ValidationBlock:
            raise
        except Exception as exc:
            raise ValidationBlock(
                f"checkpoint payload not loadable by trainer "
                f"{self.model_name!r}: {exc}",
                hint="a harness checkpoint cannot resume under a real "
                     "trainer (and vice versa) — state is never faked "
                     "into compatibility",
            ) from exc

    # -- phases ---------------------------------------------------------------

    def _phase_temperature(self) -> dict[str, Any]:
        before_nll = _nll(self.rows, 1.0)
        before_ece = _ece(self.rows, 1.0)
        t = _fit_temperature(self.rows)
        self.temperature = t
        self.nll_value = _nll(self.rows, t)
        self.ece = _ece(self.rows, t)
        return {
            "temperature": round(t, 6),
            "nll": round(self.nll_value, 6),
            "nll_before": round(before_nll, 6),
            "ece": round(self.ece, 6),
            "ece_before": round(before_ece, 6),
        }

    def _phase_conformal(self) -> dict[str, Any]:
        thresholds, coverages = _fit_conformal(
            self.rows, self.temperature, self.alphas)
        self.thresholds = thresholds
        self.coverages = coverages
        metrics: dict[str, Any] = {}
        for a in self.alphas:
            metrics[f"threshold_alpha_{a:g}"] = round(thresholds[a], 6)
            metrics[f"coverage_alpha_{a:g}"] = round(coverages[a], 6)
        return metrics

    def _phase_decomposition(self) -> dict[str, Any]:
        weights, residual = _fit_decomposition(self.rows)
        self.weights = weights
        metrics = {f"weight_{k}": round(v, 6) for k, v in weights.items()}
        metrics["residual_mse"] = round(residual, 6)
        return metrics

    # -- protocol -------------------------------------------------------------

    def step(self, state: TrainState) -> StepResult:
        if state.epoch >= self.epochs:
            return StepResult(
                state.global_step, state.epoch, loss=None, done=True)
        phase = self.phases[state.epoch]
        if phase == "temperature":
            metrics = self._phase_temperature()
            loss = round(self.nll_value or 0.0, 6)
        elif phase == "conformal":
            metrics = self._phase_conformal()
            loss = None
        else:  # decomposition (epochs == 3 guarantees the phase exists)
            metrics = self._phase_decomposition()
            loss = None
        metrics["phase"] = phase
        nxt_step = state.global_step + 1
        nxt_epoch = state.epoch + 1
        return StepResult(
            global_step=nxt_step,
            epoch=nxt_epoch,
            loss=loss,
            metrics=metrics,
            done=nxt_epoch >= self.epochs,
        )

    def checkpoint_payload(self, state: TrainState) -> dict[str, bytes]:
        """§11.4: every REQUIRED_COMPONENTS entry — real fitted state in
        `model`, honest `enabled: false` for optimizer/schedule/AMP."""
        def _json(obj: Any) -> bytes:
            return json.dumps(obj, sort_keys=True).encode("utf-8")

        model = {
            "trainer": "calibrator",
            "model": self.model_name,
            "temperature": float(self.temperature),
            "thresholds": {f"{a:g}": float(v)
                           for a, v in self.thresholds.items()},
            "coverages": {f"{a:g}": float(v)
                          for a, v in self.coverages.items()},
            "weights": self.weights,
            "ece": self.ece,
            "nll": self.nll_value,
            "classes": int(self.n_classes),
            "n": int(self.n),
            "alphas": [float(a) for a in self.alphas],
            "decomposition": bool(self.decomposition),
        }
        no_opt = {"enabled": False,
                  "reason": "closed-form fit — no gradient optimizer state"}
        return {
            "model": _json(model),
            "optimizer": _json(no_opt),
            "lr_scheduler": _json({
                "enabled": False,
                "reason": "temperature is a searched constant — no schedule",
            }),
            "amp_scaler": _json({"enabled": False,
                                 "reason": "python float fits, no AMP"}),
            "ema": _json({"enabled": False}),
            "global_step": _json(int(state.global_step)),
            "epoch": _json(int(state.epoch)),
            "batch_position": _json({
                "position_in_epoch": 0,
                "phase_index": int(state.epoch),
                "phase": (self.phases[state.epoch]
                          if state.epoch < len(self.phases) else "done"),
                "phases": list(self.phases),
                "steps_per_epoch": 1,
            }),
            "sampler_state": _json({
                "protocol": "full calibration set (no shuffle, no subsample)",
                "seed": int(self.seed),
            }),
            "dataloader_state": _json({
                "n": int(self.n),
                "classes": int(self.n_classes),
                "alphas": [float(a) for a in self.alphas],
                "decomposition": bool(self.decomposition),
                "global_batch_lever": False,
                "learning_rate_lever": False,
                "seed_role": "recorded by run schema (12 §17); fits are "
                             "deterministic without it",
            }),
            "distributed_state": _json({"world_size": 1}),
            "grad_accum_state": _json({
                "full_pass": True,
                "micro_batch": int(self.micro),
                "grad_accum": int(self.accum),
            }),
            "early_stopping_state": _json({
                "enabled": False,
                "reason": "fixed phase count, no training loop",
            }),
            "best_model_state": _json({
                "enabled": False,
                "metric": None,
                "reason": "the fit IS the model — nothing to gate on",
            }),
            "rng_hierarchy": _json({"python": random.getstate()}),
        }


#: Registry hook consumed by mlforge.trainers.build_trainer.
TRAINER_CLASS = CalibratorTrainer


__all__ = [
    "COMPONENTS",
    "MODEL_REGISTRY",
    "TRAINER_CLASS",
    "CalibratorArch",
    "CalibratorTrainer",
    "dependency_error",
    "load_prediction_rows",
]
