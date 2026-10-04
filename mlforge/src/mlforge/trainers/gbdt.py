"""C4 GBDT Hypothesis Ranker trainer — LightGBM LambdaMART, CPU only.

Spec: 10_training_plan/03_training_pipeline.md Model 4 (params literal:
lambdarank/ndcg/eval_at[3,5,10], num_leaves 31, learning_rate 0.05,
feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 5; FEATURE_NAMES
= the 10 evidence-graph features) + 02_architecture/05_reasoning
(hypothesis engine scores candidate hypotheses before verification).

Training contract (validated literally, never silently adapted):
  * semantic.loss       = lambdarank              (the only objective)
  * semantic.optimizer  = gbdt | boosting | none  (no gradient optimizer
                          exists — these all name the same boosting
                          learner; recorded, never pretended)
  * semantic.scheduler  = constant | none         (shrinkage is constant)
  * semantic.epochs     = boosting rounds (one round per step,
                          steps_per_epoch = 1)
  * semantic.learning_rate — the real shrinkage lever (spec default 0.05)
  * semantic.seed       — real lever: derives seed / bagging_seed /
                          feature_fraction_seed (spec params carry no
                          seeds; run schema 12 §17 requires one)
  * semantic.global_batch — REQUIRED by schema but not a lever: one
                          round consumes ALL train rows (recorded false)
  * optional semantic knobs, spec defaults: num_leaves (31),
    feature_fraction (0.8), bagging_fraction (0.8), bagging_freq (5)
  * optional semantic.label_column / group_column (default "label" /
    "group") and feature_columns (default = the spec FEATURE_NAMES, in
    spec order — data must carry them all)

Empirically pinned LightGBM behavior this trainer honors:
  * min_data_in_leaf defaults to 20 -> every split needs BOTH children
    >= 20 rows, and bagging_fraction samples only its fraction of rows:
    train rows must be >= ceil(2*20 / sampled_fraction)
    (= 50 with spec bagging 0.8, 40 with bagging off) or trees are
    stumps and training silently produces nothing
  * a group (query) with constant labels has zero pairs -> gain -inf;
    the loader refuses data without within-group label pairs
  * continuation via init_model is DETERMINISTIC across
    serialize/reload boundaries (resume is bitwise reproducible) but is
    NOT equivalent to one LightGBM call with N rounds — LightGBM
    re-derives its per-round RNG from the base seeds at every call
    boundary.  The trainer therefore always trains one round per step:
    an uninterrupted run and a resumed run ARE the same call sequence.

Honest limits (fail-closed):
  * too little data / no label variation -> ValidationBlock with the
    arithmetic (never a silent stump model)
  * resume only continues THIS experiment (params/features/pool must
    match); fine-tune (init_weights) continues a PARENT booster with
    our params for new rounds — feature space must match exactly
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

try:
    import lightgbm as lgb
    import numpy as np

    _LGB_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - ImportError (or broken wheel) — honest refusal
    lgb = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _LGB_IMPORT_ERROR = exc


def lightgbm_available() -> bool:
    return _LGB_IMPORT_ERROR is None


def dependency_error() -> str | None:
    """None = this trainer's framework is present here (registry probe)."""
    if not lightgbm_available():
        return (
            "lightgbm is not installed — the LambdaMART ranker needs it "
            f"(import failed: {_LGB_IMPORT_ERROR}); pip install lightgbm"
        )
    return None


#: Spec FEATURE_NAMES, in spec order (03 Model 4) — the default feature
#: schema; `semantic.feature_columns` overrides for custom headers.
FEATURE_NAMES: tuple[str, ...] = (
    "prediction_error",
    "evidence_count_for",
    "evidence_count_against",
    "temporal_consistency",
    "cross_modal_agreement",
    "entity_count",
    "occlusion_level",
    "r_score",
    "hypothesis_age_ms",
    "similar_past_episodes_count",
)

#: LightGBM's default min_data_in_leaf — not a spec knob, so fixed; the
#: row floor below is derived from it and quoted in refusal messages.
_MIN_DATA_IN_LEAF = 20


@dataclass(frozen=True)
class GbdtArch:
    """What this trainer fits, keyed by model name (semantic identity)."""

    objective: str = "lambdarank"
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    eval_at: tuple[int, ...] = (3, 5, 10)


#: Trainable GBDT models (12 §10.2 `hypothesis_ranker`).
MODEL_REGISTRY: dict[str, GbdtArch] = {
    "hypothesis_ranker": GbdtArch(),
}

#: Semantic values this trainer honors — anything else BLOCKs.
_LOSSES = {"lambdarank"}
_OPTIMIZERS = {"gbdt", "boosting", "none"}
_SCHEDULERS = {"constant", "none"}

#: Splits a ranker never trains on (counted, recorded — not silently
#: absorbed; mirrors the re-ID loader's split discipline).
_EXCLUDED_SPLITS = {"test", "query", "holdout"}
_TRAIN_SPLITS = {"", "train"}
_VAL_SPLITS = {"valid", "val", "validation"}


def load_ranker_rows(
    root: Path,
    train_datasets: Sequence[str],
    *,
    label_column: str,
    group_column: str,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    """Prepared `tabular` store artifact(s) → grouped ranker arrays.

    Mirrors gate semantics locally (registered, version-exact,
    store-backed, transform-exact).  Rows are regrouped so each query's
    rows are contiguous (LightGBM group sizes are offsets into the row
    array), first-appearance order preserved.
    """
    from mlforge.ingest.identity import parse_ref
    from mlforge.store import ContentStore

    if not train_datasets:
        raise ValidationBlock("run_spec.train_datasets is empty")
    store = ContentStore(Path(root) / "store")
    features = list(feature_columns)
    want = set(features) | {label_column, group_column}
    train_raw: list[tuple[dict[str, str], str]] = []
    val_raw: list[tuple[dict[str, str], str]] = []
    excluded: dict[str, int] = {}
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
        if schema and schema != "tabular":
            raise ValidationBlock(
                f"{ref} was prepared with transform {schema!r} — the "
                "ranker needs `tabular` rows (label + group + feature "
                f"columns), not {schema!r} rows",
            )
        for rec in doc.get("records") or []:
            if not isinstance(rec, dict):
                continue
            data = rec.get("data")
            if not isinstance(data, dict) or not data:
                continue
            # split discipline first (non-train files need no train columns)
            split = str(rec.get("split") or "").strip().lower()
            if split in _EXCLUDED_SPLITS or (
                    split and split not in _TRAIN_SPLITS
                    and split not in _VAL_SPLITS):
                key = split or "(unknown)"
                excluded[key] = excluded.get(key, 0) + 1
                continue
            pool = (val_raw if split in _VAL_SPLITS else train_raw)
            missing = [c for c in want if c not in data]
            if missing:
                src = (f"{ref}:{rec.get('relative_path', '?')}"
                       f"#{rec.get('row', '?')}")
                raise ValidationBlock(
                    f"{src} is missing column(s) {', '.join(sorted(missing))} "
                    "— the ranker reads label + group + feature columns "
                    "(spec FEATURE_NAMES, 03 Model 4); regenerate the "
                    "tabular source or set semantic.feature_columns",
                )
            pool.append((data, (f"{ref}:{rec.get('relative_path', '?')}"
                               f"#{rec.get('row', '?')}")))
    if not train_raw:
        raise ValidationBlock(
            "prepared dataset(s) contain no train-split rows — the "
            "ranker trains on split `train` (or rows with no split "
            "column); set split: train in the source or prepare rows "
            "with transform `tabular`",
        )

    def _num(data: Mapping[str, str], col: str, src: str) -> float:
        text = str(data.get(col, "")).strip()
        try:
            value = float(text)
        except ValueError as exc:
            raise ValidationBlock(
                f"{src}: column {col!r} value {text!r} is not a number",
            ) from exc
        if not math.isfinite(value):
            raise ValidationBlock(
                f"{src}: column {col!r} value {text!r} is not finite "
                "(NaN/inf never train a tree)",
            )
        return value

    def _build(pool: list[tuple[dict[str, str], str]]
               ) -> tuple[dict[str, list[tuple[int, list[float]]]], list[str]]:
        """group id -> [(label, features)] in row order; preserves first
        appearance so regrouping is stable."""
        order: list[str] = []
        buckets: dict[str, list[tuple[int, list[float]]]] = {}
        for data, src in pool:
            graw = str(data.get(group_column, "")).strip()
            if not graw:
                raise ValidationBlock(
                    f"{src}: group column {group_column!r} is empty — "
                    "ranking rows belong to a hypothesis group (query)",
                )
            if graw not in buckets:
                buckets[graw] = []
                order.append(graw)
            lval = _num(data, label_column, src)
            if lval != int(lval) or lval < 0:
                raise ValidationBlock(
                    f"{src}: label column {label_column!r} value "
                    f"{lval!r} — LambdaMART labels are integer "
                    "relevance grades >= 0",
                )
            feats = [_num(data, c, src) for c in features]
            buckets[graw].append((int(lval), feats))
        return buckets, order

    train_buckets, train_order = _build(train_raw)
    n_train = sum(len(train_buckets[g]) for g in train_order)
    n_groups = len(train_order)
    if n_groups < 2:
        raise ValidationBlock(
            f"train pool has {n_groups} group(s) — LambdaMART needs "
            ">= 2 groups (queries): ranking compares hypotheses WITHIN "
            "a group, one group is unrankable",
        )
    sizes = {g: len(train_buckets[g]) for g in train_order}
    too_small = [g for g in train_order if sizes[g] < 2]
    if too_small:
        g = too_small[0]
        raise ValidationBlock(
            f"train group {g!r} has {sizes[g]} row(s) — every group "
            "needs >= 2 hypotheses to form a ranking pair "
            "(generate_ranker_data: --min-hypotheses 3, 03 §Data)",
        )
    grades: set[int] = set()
    pairs = 0
    for g in train_order:
        seen: dict[int, int] = {}
        for label, _ in train_buckets[g]:
            grades.add(label)
            seen[label] = seen.get(label, 0) + 1
        # ordered pairs with different grades
        total = sum(seen.values())
        same = sum(c * (c - 1) // 2 for c in seen.values())
        pairs += total * (total - 1) // 2 - same
    if len(grades) < 2:
        raise ValidationBlock(
            f"train labels have {len(grades)} distinct grade(s) — "
            "LambdaMART needs >= 2 relevance grades to rank anything "
            "(label every group's hypotheses by relevance)",
        )
    if pairs == 0:
        raise ValidationBlock(
            "no within-group label pairs — every group's labels are "
            "constant, so there is nothing to rank (gain would be "
            "-inf and trees would be stumps); vary relevance WITHIN "
            "groups",
        )

    def _flat(buckets: dict[str, list[tuple[int, list[float]]]],
              order: list[str]) -> tuple[list[list[float]], list[int],
                                          list[int]]:
        xs: list[list[float]] = []
        ys: list[int] = []
        gs: list[int] = []
        for g in order:
            for label, feats in buckets[g]:
                xs.append(feats)
                ys.append(label)
            gs.append(len(buckets[g]))
        return xs, ys, gs

    train_x, train_y, train_g = _flat(train_buckets, train_order)

    val_buckets: dict[str, list[tuple[int, list[float]]]] = {}
    val_order: list[str] = []
    if val_raw:
        val_buckets, val_order = _build(val_raw)
    val_x, val_y, val_g = _flat(val_buckets, val_order) if val_order else ([], [], [])

    return {
        "train": {"X": train_x, "y": train_y, "groups": train_g},
        "val": {"X": val_x, "y": val_y, "groups": val_g} if val_order else None,
        "feature_names": features,
        "n_train": n_train,
        "n_val": len(val_x),
        "n_groups": n_groups,
        "n_val_groups": len(val_order),
        "grades": sorted(grades),
        "train_pairs": pairs,
        "excluded_splits": excluded,
    }


def _lambdarank_loss(y: Sequence[int], scores: Sequence[float],
                     groups: Sequence[int]) -> float:
    """Mean pairwise logistic loss per ordered pair (y_i > y_j) — the
    LambdaMART objective, computed in pure python for metrics."""
    total = 0.0
    count = 0
    off = 0
    for size in groups:
        idx = range(off, off + size)
        for i in idx:
            for j in idx:
                if y[i] > y[j]:
                    d = scores[i] - scores[j]
                    # stable softplus(-(si - sj))
                    total += max(0.0, -d) + math.log1p(math.exp(-abs(d)))
                    count += 1
        off += size
    return total / count if count else 0.0


def _mean_ndcg(y: Sequence[int], scores: Sequence[float],
               groups: Sequence[int], k: int) -> float:
    """Group-mean NDCG@k (binary-free, exponential gain — standard)."""
    total = 0.0
    off = 0
    for size in groups:
        lab = list(y[off:off + size])
        sc = list(scores[off:off + size])
        order = sorted(range(size), key=lambda i: sc[i], reverse=True)[:k]
        dcg = sum((2 ** lab[i] - 1) / math.log2(r + 2)
                  for r, i in enumerate(order))
        ideal = sorted(lab, reverse=True)[:k]
        idcg = sum((2 ** g - 1) / math.log2(r + 2)
                   for r, g in enumerate(ideal))
        total += dcg / idcg if idcg > 0 else 0.0
        off += size
    return total / len(groups) if groups else 0.0


def _opt_int(semantic: Mapping[str, Any], key: str, default: int,
             lo: int) -> int:
    if key not in semantic:
        return default
    try:
        value = int(semantic[key])
    except (TypeError, ValueError) as exc:
        raise ValidationBlock(
            f"semantic.{key} {semantic[key]!r} is not an integer") from exc
    if value < lo:
        raise ValidationBlock(
            f"semantic.{key} must be >= {lo} (got {value})")
    return value


def _opt_float(semantic: Mapping[str, Any], key: str, default: float,
               lo: float, hi: float, *, lo_inclusive: bool = False) -> float:
    if key not in semantic:
        return default
    try:
        value = float(semantic[key])
    except (TypeError, ValueError) as exc:
        raise ValidationBlock(
            f"semantic.{key} {semantic[key]!r} is not a number") from exc
    ok = (value >= lo) if lo_inclusive else (value > lo)
    if not ok or value > hi:
        lhs = ">=" if lo_inclusive else ">"
        raise ValidationBlock(
            f"semantic.{key} {value} out of range — must be {lhs} {lo} "
            f"and <= {hi}")
    return value


class GbdtRankerTrainer:
    """One LambdaMART round per step; payload carries the full booster."""

    def __init__(
        self,
        *,
        model_name: str,
        semantic: Mapping[str, Any],
        runtime: Any = None,
        train_datasets: Sequence[str],
        root: str | Path,
        payloads: Mapping[str, bytes] | None = None,
        start_step: int = 0,
        start_epoch: int = 0,
        plan: Any = None,
        run_dir: str | Path | None = None,   # no scratch: no downloads
        on_progress: Any = None,             # rounds outlive 30s beats
        init_weights: bytes | None = None,
    ):
        err = dependency_error()
        if err is not None:
            raise PreconditionFailed(err)
        arch = MODEL_REGISTRY.get(model_name)
        if arch is None:
            raise PreconditionFailed(
                f"model {model_name!r} has no registered GBDT ranker",
                hint=f"trainable here: {', '.join(sorted(MODEL_REGISTRY))}",
            )
        # -- semantic identity, honored literally -------------------------
        loss_name = str(semantic.get("loss", "")).strip().lower()
        if loss_name not in _LOSSES:
            raise ValidationBlock(
                f"semantic.loss {loss_name!r} is not the ranker's "
                f"objective (supports: {', '.join(sorted(_LOSSES))}) — "
                "the hypothesis ranker is LambdaMART, never a substitute",
            )
        opt_name = str(semantic.get("optimizer", "")).strip().lower()
        if opt_name not in _OPTIMIZERS:
            raise ValidationBlock(
                f"semantic.optimizer {opt_name!r} not supported "
                f"(supports: {', '.join(sorted(_OPTIMIZERS))}) — GBDT "
                "has no gradient optimizer: gbdt/boosting name the "
                "booster, none means 'no optimizer state exists'",
            )
        sched_name = str(semantic.get("scheduler", "")).strip().lower()
        if sched_name not in _SCHEDULERS:
            raise ValidationBlock(
                f"semantic.scheduler {sched_name!r} not supported "
                f"(supports: {', '.join(sorted(_SCHEDULERS))}) — "
                "shrinkage is a constant learning_rate (03 Model 4)",
            )
        precision = str(semantic.get("precision_policy", "")).strip().lower()
        if precision != "fp32":
            raise ValidationBlock(
                f"semantic.precision_policy {precision!r} — tree splits "
                "are exact CPU arithmetic; declare fp32 (no AMP exists "
                "for a booster, the payload would be a lie)",
            )
        try:
            self.epochs = int(semantic["epochs"])
            self.seed = int(semantic["seed"])
            self.global_batch = int(semantic["global_batch"])
            self.learning_rate = float(semantic["learning_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationBlock(
                f"semantic field unreadable for training: {exc}") from exc
        if self.epochs < 1:
            raise ValidationBlock(
                f"semantic.epochs {self.epochs} — boosting rounds must "
                "be >= 1 (epochs IS the round count: one round/step)",
            )
        if self.global_batch < 1:
            raise ValidationBlock(
                f"semantic.global_batch {self.global_batch} must be >= 1 "
                "(recorded, but not a lever: every round sees all rows)",
            )
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValidationBlock(
                f"semantic.learning_rate {self.learning_rate} — shrinkage "
                "must be in (0, 1] (spec: 0.05)",
            )
        self.num_leaves = _opt_int(semantic, "num_leaves",
                                   arch.num_leaves, 2)
        self.feature_fraction = _opt_float(
            semantic, "feature_fraction", arch.feature_fraction, 0.0, 1.0)
        self.bagging_fraction = _opt_float(
            semantic, "bagging_fraction", arch.bagging_fraction, 0.0, 1.0)
        self.bagging_freq = _opt_int(semantic, "bagging_freq",
                                     arch.bagging_freq, 0)

        # -- data schema knobs --------------------------------------------
        self.label_column = str(semantic.get("label_column", "label")).strip()
        self.group_column = str(semantic.get("group_column", "group")).strip()
        if not self.label_column or not self.group_column:
            raise ValidationBlock(
                "semantic.label_column / group_column must be non-empty "
                "names (defaults: label / group)",
            )
        if self.label_column == self.group_column:
            raise ValidationBlock(
                f"semantic.label_column and group_column are both "
                f"{self.label_column!r} — relevance grade and group id "
                "are different columns",
            )
        raw_features = semantic.get("feature_columns")
        if raw_features is None:
            self.feature_names: list[str] = list(FEATURE_NAMES)
        else:
            if (not isinstance(raw_features, (list, tuple))
                    or not raw_features):
                raise ValidationBlock(
                    "semantic.feature_columns must be a non-empty list of "
                    "column names (omit to use spec FEATURE_NAMES)",
                )
            self.feature_names = [str(c) for c in raw_features]
            if len(set(self.feature_names)) != len(self.feature_names):
                raise ValidationBlock(
                    "semantic.feature_columns has duplicates",
                )
            clash = set(self.feature_names) & {
                self.label_column, self.group_column}
            if clash:
                raise ValidationBlock(
                    f"semantic.feature_columns overlaps label/group "
                    f"columns: {', '.join(sorted(clash))}",
                )

        # -- LightGBM params (03 Model 4 literal + derived seeds) --------
        self.params: dict[str, Any] = {
            "objective": arch.objective,
            "metric": "ndcg",
            "eval_at": list(arch.eval_at),
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "verbose": -1,
            "seed": self.seed,
            "bagging_seed": self.seed + 1,
            "feature_fraction_seed": self.seed + 2,
        }

        # -- row floor derived from min_data_in_leaf x bagging ------------
        bagging_on = (self.bagging_fraction < 1.0
                      and self.bagging_freq > 0)
        sampled = self.bagging_fraction if bagging_on else 1.0
        self.min_rows = math.ceil(2 * _MIN_DATA_IN_LEAF / sampled)
        self._bagging_on = bagging_on

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
                    f"plan wants world_size {world} — one booster is "
                    "single-process by design",
                    hint="run single-process (topology world_size 1)",
                )
            if precision_eff != "fp32":
                raise PreconditionFailed(
                    f"plan precision {precision_eff!r} — a booster has "
                    "no AMP (the payload would be a lie)",
                    hint="semantic.precision_policy: fp32",
                )
            self.micro, self.accum = micro, accum
        else:
            self.micro, self.accum = self.global_batch, 1

        # -- data ----------------------------------------------------------
        data = load_ranker_rows(
            Path(root), train_datasets,
            label_column=self.label_column,
            group_column=self.group_column,
            feature_columns=self.feature_names,
        )
        self._data = data
        self.train_x = np.array(data["train"]["X"], dtype=np.float64)
        self.train_y: list[int] = list(data["train"]["y"])
        self.train_groups: list[int] = list(data["train"]["groups"])
        self.train_y_np = np.array(self.train_y, dtype=np.float32)
        self.val = data["val"]
        self.val_x = (np.array(self.val["X"], dtype=np.float64)
                      if self.val else None)
        self.n_train = int(data["n_train"])
        self.n_groups = int(data["n_groups"])
        self.grades: list[int] = list(data["grades"])
        self.train_pairs = int(data["train_pairs"])
        if self.n_train < self.min_rows:
            math_note = (
                f"min_data_in_leaf {_MIN_DATA_IN_LEAF} must fit in BOTH "
                "children of every split"
                + (f", and bagging_fraction "
                   f"{self.bagging_fraction:g} samples only "
                   f"{self.bagging_fraction:.0%} of rows"
                   if self._bagging_on else "")
            )
            raise ValidationBlock(
                f"train pool has {self.n_train} rows, need >= "
                f"{self.min_rows} ({math_note}) — LightGBM would "
                "silently grow stumps (no splits) instead of a ranker; "
                "generate more labeled groups or turn bagging off "
                "(semantic.bagging_fraction: 1.0)",
            )
        self.steps_per_epoch = 1  # one round per step (epoch = round)

        # -- init: resume (full state) > fine-tune > fresh ------------------
        if payloads and init_weights is not None:
            raise ValidationBlock(
                "resume payloads and fine-tune init weights are mutually "
                "exclusive — resume from THIS run's checkpoint or "
                "continue a parent booster, never both",
            )
        self.on_progress = on_progress
        self.model_name = model_name
        self.arch = arch
        self.base_rounds = 0
        if payloads:
            missing = [c for c in REQUIRED_COMPONENTS if c not in payloads]
            if missing:
                raise ValidationBlock(
                    f"checkpoint payload incomplete — missing: "
                    f"{', '.join(missing)} (not resumable, 12 §11.4)",
                )
            self._restore(dict(payloads))
        elif init_weights is not None:
            self._init_from_parent(init_weights)
        else:
            self.booster = None
            random.seed(self.seed)
            np.random.seed(self.seed % (2 ** 32))
        self.start_step = int(start_step)
        self.start_epoch = int(start_epoch)

    # -- init paths ---------------------------------------------------------

    def _parse_booster(self, blob: bytes, origin: str):
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationBlock(
                f"{origin} is not a UTF-8 LightGBM model string") from exc
        try:
            booster = lgb.Booster(model_str=text)
        except Exception as exc:
            raise ValidationBlock(
                f"{origin} is not loadable as a LightGBM booster: {exc}",
                hint="a harness checkpoint / foreign model cannot resume "
                     "under this trainer — state is never faked",
            ) from exc
        return booster

    def _init_from_parent(self, blob: bytes) -> None:
        booster = self._parse_booster(blob, "fine-tune init weights")
        objective = booster.dump_model().get("objective")
        if objective != "lambdarank":
            raise ValidationBlock(
                f"parent booster objective {objective!r} != 'lambdarank' "
                "— the ranker can only continue a ranking parent",
            )
        parent_features = booster.feature_name()
        if parent_features != self.feature_names:
            mismatch = next(
                (i for i, (a, b) in enumerate(
                    zip(parent_features, self.feature_names)) if a != b),
                min(len(parent_features), len(self.feature_names)),
            )
            got = (parent_features[mismatch]
                   if mismatch < len(parent_features) else "(missing)")
            want = (self.feature_names[mismatch]
                    if mismatch < len(self.feature_names) else "(missing)")
            raise ValidationBlock(
                f"parent booster feature {mismatch} is {got!r} != this "
                f"run's {want!r} — fine-tune needs the EXACT feature "
                "space (13 §6.4)",
            )
        if booster.current_iteration() < 1:
            raise ValidationBlock(
                "parent booster has no trees — nothing to fine-tune from",
            )
        self.booster = booster
        self.base_rounds = booster.current_iteration()

    def _restore(self, payloads: dict[str, bytes]) -> None:
        try:
            state = json.loads(
                payloads["dataloader_state"].decode("utf-8"))
            if str(state.get("trainer")) != "gbdt":
                raise ValidationBlock(
                    "checkpoint payload was not produced by trainer "
                    "'gbdt' — a resume continues the SAME experiment "
                    "(12 §11.4), never another trainer's state",
                )
            if str(state.get("variant")) != self.model_name:
                raise ValidationBlock(
                    f"checkpoint variant {state.get('variant')!r} != "
                    f"{self.model_name!r} — a resume continues the "
                    "SAME model",
                )
            stored_params = state.get("params")
            if stored_params != self.params:
                diff = [
                    k for k in sorted(set(self.params) | set(
                        stored_params or {}))
                    if self.params.get(k) != (stored_params or {}).get(k)
                ]
                raise ValidationBlock(
                    f"checkpoint params differ ({', '.join(diff) or 'format'}) "
                    "— a resume continues the SAME experiment (same "
                    "learning_rate, seeds, leaves, bagging)",
                )
            if list(state.get("feature_names") or []) != self.feature_names:
                raise ValidationBlock(
                    "checkpoint feature schema differs from "
                    "semantic.feature_columns / spec FEATURE_NAMES — "
                    "a resume never reinterprets features",
                )
            if int(state.get("n_train", -1)) != self.n_train:
                raise ValidationBlock(
                    f"checkpoint train pool {state.get('n_train')} rows "
                    f"!= this dataset's {self.n_train} — a resume "
                    "continues the SAME ranker data",
                )
            if int(state.get("n_groups", -1)) != self.n_groups:
                raise ValidationBlock(
                    f"checkpoint groups {state.get('n_groups')} != this "
                    f"dataset's {self.n_groups} — a resume continues the "
                    "SAME ranker data",
                )
            if list(state.get("grades") or []) != self.grades:
                raise ValidationBlock(
                    f"checkpoint grades {state.get('grades')} != this "
                    f"dataset's {self.grades} — label space changed",
                )
            booster = self._parse_booster(payloads["model"], "checkpoint")
            if booster.current_iteration() != int(state.get("trees", -1)):
                raise ValidationBlock(
                    f"checkpoint model has {booster.current_iteration()} "
                    f"trees but records {state.get('trees')} — payload "
                    "is inconsistent (never resume a torn checkpoint)",
                )
            self.booster = booster
            self.base_rounds = int(state.get("base_rounds", 0))
            rng = json.loads(payloads["rng_hierarchy"].decode("utf-8"))
            py = rng.get("python")
            if py is not None:
                random.setstate((py[0], tuple(py[1]), py[2]))
            np_state = rng.get("numpy")
            if np_state is not None:
                np.random.set_state((
                    np_state["kind"],
                    np.array(np_state["keys"], dtype=np.uint32),
                    int(np_state["pos"]),
                    bool(np_state["has_gauss"]),
                    float(np_state["cached"]),
                ))
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

    # -- protocol ------------------------------------------------------------

    def step(self, state: TrainState) -> StepResult:
        if state.epoch >= self.epochs:
            return StepResult(
                state.global_step, state.epoch, loss=None, done=True)
        dtrain = lgb.Dataset(
            self.train_x, label=self.train_y_np, group=self.train_groups,
            feature_name=self.feature_names,
        )
        prev_trees = (self.booster.current_iteration()
                      if self.booster is not None else 0)
        if self.booster is None:
            self.booster = lgb.train(self.params, dtrain,
                                     num_boost_round=1)
        else:
            self.booster = lgb.train(
                self.params, dtrain, num_boost_round=1,
                init_model=self.booster)
        trees = self.booster.current_iteration()
        scores = [float(s) for s in self.booster.predict(self.train_x)]
        if any(not math.isfinite(s) for s in scores):
            raise ValidationBlock(
                "ranker produced non-finite scores — data or params are "
                "degenerate (this should be impossible after the loader "
                "guards; please report with the run id)",
            )
        metrics: dict[str, Any] = {
            "round": int(trees),
            "stalled": int(trees == prev_trees),
            "loss": round(_lambdarank_loss(
                self.train_y, scores, self.train_groups), 6),
            "ndcg3": round(_mean_ndcg(
                self.train_y, scores, self.train_groups, 3), 6),
            "ndcg5": round(_mean_ndcg(
                self.train_y, scores, self.train_groups, 5), 6),
        }
        if self.val_x is not None:
            val_scores = [float(s) for s in self.booster.predict(self.val_x)]
            metrics["ndcg3_val"] = round(_mean_ndcg(
                self.val["y"], val_scores, self.val["groups"], 3), 6)
            metrics["ndcg5_val"] = round(_mean_ndcg(
                self.val["y"], val_scores, self.val["groups"], 5), 6)
        if self.on_progress is not None:
            self.on_progress(f"round {trees}")
        nxt_step = state.global_step + 1
        nxt_epoch = state.epoch + 1
        return StepResult(
            global_step=nxt_step,
            epoch=nxt_epoch,
            loss=metrics["loss"],
            metrics=metrics,
            done=nxt_epoch >= self.epochs,
        )

    def checkpoint_payload(self, state: TrainState) -> dict[str, bytes]:
        """§11.4: every REQUIRED_COMPONENTS entry — the booster IS the
        model; optimizer/schedule/AMP honestly `enabled: false`."""
        def _json(obj: Any) -> bytes:
            return json.dumps(obj, sort_keys=True).encode("utf-8")

        trees = (self.booster.current_iteration()
                 if self.booster is not None else 0)
        model_blob = (self.booster.model_to_string().encode("utf-8")
                      if self.booster is not None else b"")
        no_opt = {"enabled": False,
                  "reason": "gradient boosting — trees are the state "
                            "(no gradient optimizer exists)"}
        return {
            "model": model_blob,
            "optimizer": _json(no_opt),
            "lr_scheduler": _json({
                "enabled": False,
                "reason": "constant shrinkage learning_rate (03 Model 4) "
                          "— no schedule configured",
            }),
            "amp_scaler": _json({"enabled": False,
                                 "reason": "tree building is exact CPU "
                                           "arithmetic — no AMP"}),
            "ema": _json({"enabled": False}),
            "global_step": _json(int(state.global_step)),
            "epoch": _json(int(state.epoch)),
            "batch_position": _json({
                "position_in_epoch": 0,
                "steps_per_epoch": 1,
                "steps": int(state.global_step),
                "trees": int(trees),
                "base_rounds": int(self.base_rounds),
            }),
            "sampler_state": _json({
                "protocol": "full train rows in group order "
                            "(no shuffle; bagging samples inside "
                            "LightGBM)",
                "seed": int(self.seed),
            }),
            "dataloader_state": _json({
                "trainer": "gbdt",
                "variant": self.model_name,
                "feature_names": self.feature_names,
                "label_column": self.label_column,
                "group_column": self.group_column,
                "n_train": int(self.n_train),
                "n_groups": int(self.n_groups),
                "n_val": int(self._data["n_val"]),
                "grades": self.grades,
                "train_pairs": int(self.train_pairs),
                "excluded_splits": self._data["excluded_splits"],
                "min_rows": int(self.min_rows),
                "trees": int(trees),
                "base_rounds": int(self.base_rounds),
                "params": self.params,
                "global_batch_lever": False,
                "learning_rate_lever": True,
                "seed_role": "real lever — derives seed / bagging_seed / "
                             "feature_fraction_seed (spec params carry "
                             "no seeds)",
            }),
            "distributed_state": _json({"world_size": 1}),
            "grad_accum_state": _json({
                "full_pass": True,
                "global_batch_lever": False,
                "micro_batch": int(self.micro),
                "grad_accum": int(self.accum),
            }),
            "early_stopping_state": _json({
                "enabled": False,
                "reason": "fixed epochs = boosting rounds (12 §4.2) — "
                          "no early stopping configured",
            }),
            "best_model_state": _json({
                "enabled": False,
                "metric": None,
                "reason": "the final booster IS the model — nothing "
                          "extra to gate on",
            }),
            "rng_hierarchy": _json({
                "python": random.getstate(),
                "numpy": {
                    "kind": np.random.get_state()[0],
                    "keys": np.random.get_state()[1].tolist(),
                    "pos": int(np.random.get_state()[2]),
                    "has_gauss": bool(np.random.get_state()[3]),
                    "cached": float(np.random.get_state()[4]),
                },
                "lgb_seeds": {
                    "seed": int(self.seed),
                    "bagging_seed": int(self.params["bagging_seed"]),
                    "feature_fraction_seed": int(
                        self.params["feature_fraction_seed"]),
                },
            }),
        }


#: Registry hook consumed by mlforge.trainers.build_trainer.
TRAINER_CLASS = GbdtRankerTrainer


__all__ = [
    "FEATURE_NAMES",
    "MODEL_REGISTRY",
    "TRAINER_CLASS",
    "GbdtArch",
    "GbdtRankerTrainer",
    "dependency_error",
    "lightgbm_available",
    "load_ranker_rows",
]
