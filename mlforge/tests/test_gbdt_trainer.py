"""C4 GBDT ranker trainer — semantic contract (LambdaMART only), the
row-floor arithmetic (min_data_in_leaf x bagging), grouped-data guards,
real boosting on synthetic groups (loss down, NDCG up), exact resume
across serialization, parent fine-tune, and honest refusals.

Fixtures are synthetic tabular rows: 12 groups x 6 hypotheses with the
label correlated to r_score so LambdaMART has real signal (NDCG@3
improves), plus degenerate pools for each guard. Deterministic (seeded)
and the trainer is single-process CPU — no environment needed."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import TrainState
from mlforge.store import ContentStore
from mlforge.trainers import gbdt
from mlforge.trainers.gbdt import (
    FEATURE_NAMES,
    MODEL_REGISTRY,
    GbdtRankerTrainer,
    _lambdarank_loss,
    _mean_ndcg,
    dependency_error,
    lightgbm_available,
    load_ranker_rows,
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def sem(**over) -> dict:
    base = {
        "optimizer": "gbdt",
        "learning_rate": 0.05,      # spec default (03 Model 4)
        "scheduler": "constant",
        "loss": "lambdarank",
        "seed": 42,
        "global_batch": 64,         # recorded, not a lever (full rows)
        "epochs": 6,                # boosting rounds (round/step)
        "precision_policy": "fp32",
    }
    base.update(over)
    return base


def build(root, *, semantic=None, datasets=("ranker:v1",), **kw):
    return GbdtRankerTrainer(
        model_name=kw.pop("model", "hypothesis_ranker"),
        semantic=semantic if semantic is not None else sem(),
        runtime=kw.pop("runtime", {}),
        train_datasets=datasets,
        root=Path(root),
        **kw,
    )


def run_steps(trainer, state, n):
    results = []
    for _ in range(n):
        result = trainer.step(state)
        state = TrainState(result.global_step, result.epoch, state.resume_from)
        results.append(result)
    return state, results


class _Plan:
    def __init__(self, **over) -> None:
        self.world_size = over.get("world_size", 1)
        self.micro_batch = over.get("micro_batch", 64)
        self.grad_accum = over.get("grad_accum", 1)
        self.precision_effective = over.get("precision_effective", "fp32")


def ranker_rows(*, groups=12, per=6, seed=7, mode="correlated",
                split="train", val_groups=0, label_col="label",
                group_col="group", extra_split_rows=0) -> list[dict]:
    """Synthetic tabular records. mode:
      correlated — label from r_score (learnable signal)
      constant   — every row in a group shares one label (zero pairs)
      flat       — one grade everywhere (nothing to rank)"""
    rng = random.Random(seed)
    rows: list[dict] = []
    counter = 0
    total = groups + val_groups
    for gi in range(total):
        this_split = "valid" if gi >= groups else split
        for hi in range(per):
            data = {c: str(round(rng.random(), 3)) for c in FEATURE_NAMES}
            r = rng.random()
            data["r_score"] = str(round(r, 3))
            if mode == "correlated":
                data[label_col] = str(min(3, int(r * 4)))
            elif mode == "constant":
                data[label_col] = str(gi % 4)
            else:  # flat
                data[label_col] = "1"
            data[group_col] = f"g{gi}"
            rows.append({"split": this_split, "relative_path": "g.csv",
                         "row": counter, "data": data})
            counter += 1
    for i in range(extra_split_rows):
        data = {c: "0.5" for c in FEATURE_NAMES}
        data[label_col] = "1"
        data[group_col] = f"scoring{i}"
        rows.append({"split": "scoring", "relative_path": "s.csv",
                     "row": i, "data": data})
    return rows


def make_ranker_project(root: Path, rows: list[dict], *,
                        dataset: str = "ranker",
                        transform: str = "tabular",
                        version: str = "v1") -> None:
    """Registered + store-backed prepared dataset (what dataset add +
    prepare leave behind — records only, no media to resolve)."""
    doc = {
        "artifact_schema": "mlforge.prepared_dataset.v1",
        "model": "hypothesis_ranker",
        "transform": {"name": transform},
        "records": rows,
    }
    ident = ContentStore(Path(root) / "store").put_bytes(
        json.dumps(doc, sort_keys=True).encode("utf-8"))
    d = Path(root) / "datasets" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.json").write_text(json.dumps({
        "dataset_id": dataset, "identity": ident, "version": version,
        "file_count": len(rows), "total_bytes": 1,
    }), encoding="utf-8")


@pytest.fixture()
def project(tmp_path) -> Path:
    make_ranker_project(tmp_path, ranker_rows())
    return tmp_path


# ---------------------------------------------------------------------------
# registry / availability
# ---------------------------------------------------------------------------


def test_registry_identity():
    assert "hypothesis_ranker" in MODEL_REGISTRY
    arch = MODEL_REGISTRY["hypothesis_ranker"]
    assert arch.objective == "lambdarank"
    assert arch.eval_at == (3, 5, 10)
    assert arch.num_leaves == 31
    assert arch.learning_rate == 0.05
    assert FEATURE_NAMES[0] == "prediction_error"
    assert len(FEATURE_NAMES) == 10  # spec FEATURE_NAMES, 03 Model 4
    from mlforge.trainers import availability_error, registered_models
    assert availability_error("hypothesis_ranker") is None  # lightgbm here
    assert "hypothesis_ranker" in registered_models()


def test_unknown_model_refuses(tmp_path):
    with pytest.raises(PreconditionFailed, match="no registered GBDT") as ei:
        build(tmp_path, model="ranker_v9")
    assert "hypothesis_ranker" in (ei.value.hint or "")


def test_missing_lightgbm_is_honest(monkeypatch):
    monkeypatch.setattr(gbdt, "_LGB_IMPORT_ERROR",
                        ImportError("No module named 'lightgbm'"))
    assert lightgbm_available() is False
    msg = dependency_error()
    assert msg and "pip install lightgbm" in msg
    monkeypatch.setattr(
        gbdt, "dependency_error",
        lambda: "lightgbm is not installed — the LambdaMART ranker needs "
                "it; pip install lightgbm")
    with pytest.raises(PreconditionFailed, match="pip install lightgbm"):
        build("/nonexistent")  # dependency checked before any data access


# ---------------------------------------------------------------------------
# semantic contract (honored literally)
# ---------------------------------------------------------------------------


def test_semantic_values_are_literal(tmp_path, project):
    with pytest.raises(ValidationBlock, match="semantic.loss"):
        build(project, semantic=sem(loss="cross_entropy"))
    with pytest.raises(ValidationBlock, match="semantic.optimizer"):
        build(project, semantic=sem(optimizer="adamw"))
    with pytest.raises(ValidationBlock, match="semantic.scheduler"):
        build(project, semantic=sem(scheduler="cosine"))
    with pytest.raises(ValidationBlock, match="precision_policy"):
        build(project, semantic=sem(precision_policy="bf16"))
    # optimizer "none" and "boosting" are accepted synonyms (no gradient
    # optimizer exists — both mean the booster learner)
    for opt in ("gbdt", "boosting", "none"):
        t = build(project, semantic=sem(optimizer=opt))
        assert t.params["objective"] == "lambdarank"


def test_epochs_is_the_round_count(tmp_path, project):
    with pytest.raises(ValidationBlock, match="epochs IS the round count"):
        build(project, semantic=sem(epochs=0))
    t = build(project, semantic=sem(epochs=200))  # spec example: 200
    assert t.epochs == 200
    assert t.steps_per_epoch == 1


def test_required_fields_and_ranges(tmp_path, project):
    bad = sem()
    del bad["seed"]
    with pytest.raises(ValidationBlock, match="unreadable"):
        build(project, semantic=bad)
    with pytest.raises(ValidationBlock, match="learning_rate"):
        build(project, semantic=sem(learning_rate=0.0))
    with pytest.raises(ValidationBlock, match="learning_rate"):
        build(project, semantic=sem(learning_rate=1.5))
    with pytest.raises(ValidationBlock, match="global_batch"):
        build(project, semantic=sem(global_batch=0))


def test_optional_knob_validation(tmp_path, project):
    with pytest.raises(ValidationBlock, match="num_leaves"):
        build(project, semantic=sem(num_leaves=1))
    with pytest.raises(ValidationBlock, match="feature_fraction"):
        build(project, semantic=sem(feature_fraction=1.5))
    with pytest.raises(ValidationBlock, match="bagging_freq"):
        build(project, semantic=sem(bagging_freq=-1))
    with pytest.raises(ValidationBlock, match="bagging_fraction"):
        build(project, semantic=sem(bagging_fraction=0))
    # defaults flow into params; overrides honored
    t = build(project, semantic=sem(num_leaves=15, learning_rate=0.1))
    assert t.params["num_leaves"] == 15
    assert t.params["learning_rate"] == 0.1
    assert t.params["bagging_seed"] == 43   # seed + 1 (derived, documented)
    assert t.params["feature_fraction_seed"] == 44


def test_label_group_feature_column_knobs(tmp_path):
    rows = ranker_rows(label_col="grade", group_col="query")
    make_ranker_project(tmp_path, rows)
    t = build(tmp_path, semantic=sem(label_column="grade",
                                     group_column="query"))
    assert t.label_column == "grade" and t.group_column == "query"
    with pytest.raises(ValidationBlock, match="label_column"):
        build(tmp_path, semantic=sem(group_column="label"))  # same col
    with pytest.raises(ValidationBlock, match="feature_columns"):
        build(tmp_path, semantic=sem(
            feature_columns=["a", "a"]))  # duplicates
    with pytest.raises(ValidationBlock, match="overlaps"):
        build(tmp_path, semantic=sem(
            feature_columns=list(FEATURE_NAMES[:-1]) + ["label"]))
    with pytest.raises(ValidationBlock, match="non-empty"):
        build(tmp_path, semantic=sem(feature_columns=[]))
    # custom feature_columns selects a subset of the spec schema
    # (against a project with the default label/group columns)
    make_ranker_project(tmp_path, ranker_rows(), dataset="ranker_default")
    reduced = ["r_score", "temporal_consistency", "evidence_count_for"]
    t2 = build(tmp_path, datasets=("ranker_default:v1",),
               semantic=sem(feature_columns=reduced))
    assert t2.feature_names == reduced


# ---------------------------------------------------------------------------
# row floor arithmetic (min_data_in_leaf x bagging)
# ---------------------------------------------------------------------------


def test_row_floor_refuses_stump_data(tmp_path):
    # default spec bagging (0.8): need ceil(2*20 / 0.8) = 50 rows
    make_ranker_project(tmp_path, ranker_rows(groups=7, per=6))  # 42 rows
    with pytest.raises(ValidationBlock) as ei:
        build(tmp_path)
    msg = str(ei.value)
    assert ">= 50" in msg and "min_data_in_leaf" in msg
    assert "bagging_fraction 0.8" in msg
    assert "stumps" in msg


def test_row_floor_honors_bagging_off(tmp_path):
    # bagging off: floor is 2*20 = 40 — 45 rows now train
    make_ranker_project(tmp_path, ranker_rows(groups=9, per=5))  # 45 rows
    with pytest.raises(ValidationBlock, match=">= 50"):
        build(tmp_path)
    t = build(tmp_path, semantic=sem(bagging_fraction=1.0))
    assert t.min_rows == 40 and t.n_train == 45
    state, results = run_steps(t, TrainState(0, 0), 3)
    assert results[-1].metrics["round"] == 3
    assert results[-1].metrics["stalled"] == 0


# ---------------------------------------------------------------------------
# grouped-data guards (load_ranker_rows)
# ---------------------------------------------------------------------------


def test_loader_requires_registered_and_prepared(tmp_path, project):
    with pytest.raises(PreconditionFailed, match="not registered"):
        load_ranker_rows(project, ["nope:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    (project / "datasets" / "ranker" / "identity.json").write_text(
        json.dumps({"dataset_id": "ranker", "identity": "x",
                    "version": "v2"}), encoding="utf-8")
    with pytest.raises(ValidationBlock, match="registered version"):
        load_ranker_rows(project, ["ranker:v1"], label_column="label",
                         group_column="group",
                         feature_columns=FEATURE_NAMES)


def test_loader_rejects_wrong_transform(tmp_path):
    make_ranker_project(tmp_path, ranker_rows(), transform="imagefolder")
    with pytest.raises(ValidationBlock, match="transform"):
        load_ranker_rows(tmp_path, ["ranker:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)


def test_loader_group_guards(tmp_path):
    # one group only
    make_ranker_project(tmp_path, ranker_rows(groups=1, per=6),
                        dataset="one")
    with pytest.raises(ValidationBlock, match=">= 2 groups"):
        load_ranker_rows(tmp_path, ["one:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    # single-row group: keep group g0 whole, one row from every other
    rows = ranker_rows(groups=12, per=6)
    rows = [r for r in rows
            if r["data"]["group"] == "g0" or r["row"] % 6 == 0]
    make_ranker_project(tmp_path, rows, dataset="single")
    with pytest.raises(ValidationBlock, match="row\\(s\\)"):
        load_ranker_rows(tmp_path, ["single:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    # constant labels within every group -> no pairs
    make_ranker_project(tmp_path, ranker_rows(mode="constant"),
                        dataset="const")
    with pytest.raises(ValidationBlock, match="no within-group label pairs"):
        load_ranker_rows(tmp_path, ["const:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    # one grade everywhere
    make_ranker_project(tmp_path, ranker_rows(mode="flat"), dataset="flat")
    with pytest.raises(ValidationBlock, match="distinct grade"):
        load_ranker_rows(tmp_path, ["flat:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)


def test_loader_row_level_guards(tmp_path):
    rows = ranker_rows()
    rows[0]["data"]["r_score"] = "not-a-number"
    make_ranker_project(tmp_path, rows, dataset="badnum")
    with pytest.raises(ValidationBlock, match="'r_score'.*not a number"):
        load_ranker_rows(tmp_path, ["badnum:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    rows = ranker_rows()
    rows[1]["data"]["prediction_error"] = "nan"
    make_ranker_project(tmp_path, rows, dataset="nan")
    with pytest.raises(ValidationBlock, match="not finite"):
        load_ranker_rows(tmp_path, ["nan:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    rows = ranker_rows()
    rows[2]["data"]["label"] = "-1"
    make_ranker_project(tmp_path, rows, dataset="neg")
    with pytest.raises(ValidationBlock, match=">= 0"):
        load_ranker_rows(tmp_path, ["neg:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    rows = ranker_rows()
    del rows[3]["data"]["entity_count"]
    make_ranker_project(tmp_path, rows, dataset="miss")
    with pytest.raises(ValidationBlock, match="missing column.*entity_count"):
        load_ranker_rows(tmp_path, ["miss:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    rows = ranker_rows()
    rows[4]["data"]["group"] = ""
    make_ranker_project(tmp_path, rows, dataset="nogroup")
    with pytest.raises(ValidationBlock, match="group column.*empty"):
        load_ranker_rows(tmp_path, ["nogroup:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)
    # no train rows at all
    make_ranker_project(tmp_path, ranker_rows(split="valid"), dataset="val")
    with pytest.raises(ValidationBlock, match="no train-split rows"):
        load_ranker_rows(tmp_path, ["val:v1"], label_column="label",
                         group_column="group", feature_columns=FEATURE_NAMES)


def test_loader_split_pools_and_regrouping(tmp_path):
    rows = ranker_rows(groups=12, per=6, val_groups=3, extra_split_rows=4)
    make_ranker_project(tmp_path, rows, dataset="mixed")
    data = load_ranker_rows(tmp_path, ["mixed:v1"], label_column="label",
                            group_column="group",
                            feature_columns=FEATURE_NAMES)
    assert data["n_train"] == 72
    assert data["n_val"] == 18
    assert data["excluded_splits"] == {"scoring": 4}  # counted, recorded
    # regrouping: group sizes contiguous, first-appearance order
    assert len(data["train"]["groups"]) == 12
    assert sum(data["train"]["groups"]) == 72
    # custom column names through the same pool
    data2 = load_ranker_rows(tmp_path, ["mixed:v1"], label_column="label",
                             group_column="group",
                             feature_columns=list(FEATURE_NAMES))
    assert data2["grades"] == data["grades"]


# ---------------------------------------------------------------------------
# execution plan (identity checks; global_batch is NOT a lever)
# ---------------------------------------------------------------------------


def test_plan_guards(tmp_path, project):
    with pytest.raises(ValidationBlock, match="global-batch invariant"):
        build(project, plan=_Plan(micro_batch=8, grad_accum=2))  # 16 != 64
    with pytest.raises(PreconditionFailed, match="world_size 2"):
        build(project, plan=_Plan(world_size=2, micro_batch=8,
                                  grad_accum=4))  # invariant ok, world not
    with pytest.raises(PreconditionFailed, match="AMP"):
        build(project, plan=_Plan(precision_effective="bf16"))
    t = build(project, plan=_Plan())
    assert t.params["bagging_freq"] == 5  # spec param kept


# ---------------------------------------------------------------------------
# real training
# ---------------------------------------------------------------------------


def test_boosting_reduces_loss_and_ranks(project):
    t = build(project, semantic=sem(epochs=6))
    state, results = run_steps(t, TrainState(0, 0), 6)
    assert results[-1].done is True
    assert [r.metrics["round"] for r in results] == [1, 2, 3, 4, 5, 6]
    assert all(r.metrics["stalled"] == 0 for r in results)
    assert results[-1].loss < results[0].loss          # objective drops
    assert results[0].metrics["ndcg3"] > 0.5           # learns the signal
    assert results[-1].metrics["ndcg3"] >= results[0].metrics["ndcg3"]
    assert "ndcg5" in results[0].metrics
    assert "ndcg3_val" not in results[0].metrics       # no val pool here


def test_val_metrics_when_val_pool_present(tmp_path):
    make_ranker_project(tmp_path, ranker_rows(groups=12, per=6,
                                              val_groups=3))
    t = build(tmp_path, semantic=sem(epochs=3))
    _, results = run_steps(t, TrainState(0, 0), 3)
    assert "ndcg3_val" in results[0].metrics
    assert "ndcg5_val" in results[0].metrics
    assert 0.0 <= results[0].metrics["ndcg3_val"] <= 1.0


def test_on_progress_heartbeat(project):
    seen: list = []
    t = build(project, semantic=sem(epochs=2), on_progress=seen.append)
    run_steps(t, TrainState(0, 0), 2)
    assert seen and seen[0].startswith("round ")


def test_step_after_epochs_is_done(project):
    t = build(project, semantic=sem(epochs=1))
    result = t.step(TrainState(0, 5))
    assert result.done is True and result.metrics == {}


def test_metrics_helpers_are_group_aware():
    # groups of 3; perfect scores -> NDCG 1.0; inverted -> < 1
    y = [3, 2, 0, 1, 1, 1]
    groups = [3, 3]
    perfect = [0.9, 0.5, 0.1, 0.7, 0.4, 0.2]
    assert _mean_ndcg(y, perfect, groups, 3) == pytest.approx(1.0)
    assert _lambdarank_loss(y, perfect, groups) > 0.0
    # loss lower when good pairs are ordered correctly
    better = [0.9, 0.5, 0.1, 0.4, 0.7, 0.2]  # 0.7>0.4 wrongly
    worse = [0.1, 0.5, 0.9, 0.4, 0.7, 0.2]
    assert _lambdarank_loss(y, perfect, groups) < _lambdarank_loss(
        y, worse, groups)


# ---------------------------------------------------------------------------
# checkpoint payload / resume
# ---------------------------------------------------------------------------


def test_payload_covers_required_components(project):
    t = build(project, semantic=sem(epochs=3))
    state, _ = run_steps(t, TrainState(0, 0), 3)
    payload = t.checkpoint_payload(state)
    assert set(payload) == set(REQUIRED_COMPONENTS)
    assert payload["model"]  # raw LightGBM booster text, utf-8
    dstate = json.loads(payload["dataloader_state"])
    assert dstate["trainer"] == "gbdt"
    assert dstate["variant"] == "hypothesis_ranker"
    assert dstate["feature_names"] == list(FEATURE_NAMES)
    assert dstate["global_batch_lever"] is False
    assert dstate["learning_rate_lever"] is True
    assert dstate["params"]["objective"] == "lambdarank"
    opt = json.loads(payload["optimizer"])
    assert opt["enabled"] is False and "trees are the state" in opt["reason"]
    sched = json.loads(payload["lr_scheduler"])
    assert sched["enabled"] is False and "shrinkage" in sched["reason"]
    amp = json.loads(payload["amp_scaler"])
    assert amp["enabled"] is False and "no AMP" in amp["reason"]
    step = json.loads(payload["global_step"])
    assert step == 3


def test_resume_equals_uninterrupted(project):
    # A: 6 rounds straight through
    t_a = build(project, semantic=sem(epochs=6))
    state_a, _ = run_steps(t_a, TrainState(0, 0), 6)
    # B: 4 rounds, checkpoint, reload, finish
    t_b = build(project, semantic=sem(epochs=6))
    state_b, _ = run_steps(t_b, TrainState(0, 0), 4)
    mid = t_b.checkpoint_payload(state_b)
    t_c = build(project, semantic=sem(epochs=6), payloads=mid,
                start_step=state_b.global_step, start_epoch=state_b.epoch)
    state_c, results_c = run_steps(t_c, state_b, 2)
    assert t_c.booster.current_iteration() == 6
    assert t_a.checkpoint_payload(state_a)["model"] == \
        t_c.checkpoint_payload(state_c)["model"]
    assert results_c[-1].done is True


def test_resume_refuses_foreign_and_drifted_state(project):
    t = build(project, semantic=sem(epochs=3))
    state, _ = run_steps(t, TrainState(0, 0), 3)
    payload = t.checkpoint_payload(state)

    # another trainer's marker
    drifted = dict(payload)
    dstate = json.loads(payload["dataloader_state"])
    dstate["trainer"] = "calibrator"
    drifted["dataloader_state"] = json.dumps(dstate).encode()
    with pytest.raises(ValidationBlock, match="not produced by trainer"):
        build(project, payloads=drifted)

    # learning_rate drift
    with pytest.raises(ValidationBlock, match="params differ.*learning_rate"):
        build(project, semantic=sem(epochs=3, learning_rate=0.1),
              payloads=payload)

    # feature schema drift
    drifted2 = dict(payload)
    dstate2 = json.loads(payload["dataloader_state"])
    dstate2["feature_names"] = ["something_else"] + list(FEATURE_NAMES)[1:]
    drifted2["dataloader_state"] = json.dumps(dstate2).encode()
    with pytest.raises(ValidationBlock, match="feature schema differs"):
        build(project, payloads=drifted2)

    # model blob inconsistent with recorded tree count
    drifted3 = dict(payload)
    dstate3 = json.loads(payload["dataloader_state"])
    dstate3["trees"] = 99
    drifted3["dataloader_state"] = json.dumps(dstate3).encode()
    with pytest.raises(ValidationBlock, match="inconsistent"):
        build(project, payloads=drifted3)

    # not a booster at all
    drifted4 = dict(payload)
    drifted4["model"] = b"definitely not lightgbm"
    with pytest.raises(ValidationBlock, match="not loadable"):
        build(project, payloads=drifted4)


def test_resume_data_pool_must_match(project, tmp_path):
    t = build(project, semantic=sem(epochs=3))
    state, _ = run_steps(t, TrainState(0, 0), 3)
    payload = t.checkpoint_payload(state)
    # a different pool (more rows/groups)
    other = tmp_path / "other"
    other.mkdir()
    make_ranker_project(other, ranker_rows(groups=20, per=6, seed=99))
    with pytest.raises(ValidationBlock, match="train pool .* rows"):
        build(other, payloads=payload)


def test_payload_mutual_exclusions(project):
    t = build(project, semantic=sem(epochs=2))
    state, _ = run_steps(t, TrainState(0, 0), 2)
    payload = t.checkpoint_payload(state)
    with pytest.raises(ValidationBlock, match="mutually exclusive"):
        build(project, payloads=payload, init_weights=payload["model"])
    incomplete = {k: v for k, v in payload.items()
                  if k != "early_stopping_state"}
    with pytest.raises(ValidationBlock, match="missing: "
                       "early_stopping_state"):
        build(project, payloads=incomplete)


def test_resume_rng_restored(project):
    t = build(project, semantic=sem(epochs=6))
    state, _ = run_steps(t, TrainState(0, 0), 3)
    payload = t.checkpoint_payload(state)
    rng = json.loads(payload["rng_hierarchy"])
    assert rng["lgb_seeds"]["seed"] == 42
    assert rng["lgb_seeds"]["bagging_seed"] == 43
    assert "numpy" in rng and "python" in rng
    t2 = build(project, semantic=sem(epochs=6), payloads=payload,
               start_step=state.global_step, start_epoch=state.epoch)
    result = t2.step(state)
    assert result.metrics["round"] == 4


# ---------------------------------------------------------------------------
# fine-tune (init_weights: parent booster continuation)
# ---------------------------------------------------------------------------


def test_finetune_continues_parent(project):
    t = build(project, semantic=sem(epochs=6))
    state, _ = run_steps(t, TrainState(0, 0), 6)
    payload = t.checkpoint_payload(state)
    t2 = build(project, semantic=sem(epochs=3, learning_rate=0.02),
               init_weights=payload["model"])
    assert t2.base_rounds == 6          # parent trees
    assert t2.booster.current_iteration() == 6
    state2, results = run_steps(t2, TrainState(0, 0), 3)
    # new rounds stack on the parent with OUR learning_rate
    assert results[-1].metrics["round"] == 9
    assert results[-1].loss is not None


def test_finetune_refuses_foreign_objective(project):
    import lightgbm as lgb
    import numpy as np
    x = np.random.default_rng(0).random((60, 10))
    y = np.random.default_rng(1).random(60)
    reg = lgb.train({"objective": "regression", "verbose": -1},
                    lgb.Dataset(x, label=y,
                                feature_name=list(FEATURE_NAMES)),
                    num_boost_round=3)
    with pytest.raises(ValidationBlock, match="objective"):
        build(project, init_weights=reg.model_to_string().encode())


def test_finetune_refuses_feature_drift(project):
    # parent trained on a DIFFERENT feature schema
    import lightgbm as lgb
    import numpy as np
    x = np.random.default_rng(0).random((72, 10))
    y = np.array([i % 4 for i in range(72)], dtype=np.float32)
    other_names = [f"feat_{i}" for i in range(10)]
    par = lgb.train({"objective": "lambdarank", "verbose": -1,
                     "num_leaves": 7},
                    lgb.Dataset(x, label=y, group=[6] * 12,
                                feature_name=other_names),
                    num_boost_round=3)
    with pytest.raises(ValidationBlock, match="EXACT feature space"):
        build(project, init_weights=par.model_to_string().encode())


def test_finetune_refuses_empty_parent(project):
    import lightgbm as lgb
    import numpy as np
    x = np.random.default_rng(0).random((10, 10))
    # a Booster constructed but never trained has no trees
    empty = lgb.Booster(params={"objective": "lambdarank"},
                        train_set=lgb.Dataset(
                            x, label=np.zeros(10, dtype=np.float32),
                            group=[5, 5],
                            feature_name=list(FEATURE_NAMES)))
    with pytest.raises(ValidationBlock, match="no trees"):
        build(project, init_weights=empty.model_to_string().encode())
