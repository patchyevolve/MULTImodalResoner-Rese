"""C5 calibrator trainer — phase contract (epochs == fit phases), the
honest disabled components, data guards, temperature/conformal/decomposition
fits on synthetic pools, exact resume, and the fine-tune refusal.

Fixtures are synthetic prediction pools: overconfident binary logits
(scale 3, 25% label flips) so temperature scaling has real work to do
(ECE ~0.2 at T=1), plus planted-weight component rows for the optional
decomposition phase. Everything is deterministic (seeded)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import TrainState
from mlforge.store import ContentStore
from mlforge.trainers.calibrator import (
    COMPONENTS,
    MODEL_REGISTRY,
    CalibratorTrainer,
    _fit_decomposition,
    _solve,
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def sem(**over) -> dict:
    base = {
        "optimizer": "none",
        "learning_rate": 1.0,      # required by run schema, not a lever
        "scheduler": "constant",
        "loss": "nll",
        "seed": 42,
        "global_batch": 1,         # full-pool fits (recorded, not a lever)
        "epochs": 2,               # temperature + conformal
        "precision_policy": "fp32",
    }
    base.update(over)
    return base


def build(root, *, semantic=None, datasets=("preds:v1",), **kw):
    return CalibratorTrainer(
        model_name=kw.pop("model", "calibrator"),
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
        self.micro_batch = over.get("micro_batch", 1)
        self.grad_accum = over.get("grad_accum", 1)
        self.precision_effective = over.get("precision_effective", "fp32")


def overconfident_rows(n: int = 40, scale: float = 3.0,
                       flip: float = 0.25, seed: int = 7) -> list[dict]:
    """Binary pool: model always confident, right ~75% of the time —
    T=1 ECE ≈ 0.2, so a fitted T > 1 must help."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        pred = i % 2
        label = 1 - pred if rng.random() < flip else pred
        logits = [scale, -scale] if pred == 0 else [-scale, scale]
        rows.append({"logits": logits, "label": label})
    return rows


def component_rows(n: int = 64, seed: int = 11) -> list[dict]:
    """Planted dominance: correctness driven mostly by `perception`."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        comp = {c: rng.random() for c in COMPONENTS}
        score = 0.9 * comp["perception"] + 0.02 * sum(
            comp[c] for c in COMPONENTS if c != "perception")
        outcome = 1 if score > 0.5 else 0
        pred = i % 2
        label = pred if outcome else 1 - pred
        logits = [2.0, -2.0] if pred == 0 else [-2.0, 2.0]
        rows.append({"logits": logits, "label": label, "components": comp})
    return rows


def make_calib_project(root: Path, rows: list[dict], *,
                       dataset: str = "preds",
                       transform: str = "calibration") -> None:
    """Registered + store-backed prepared dataset (what dataset add +
    prepare leave behind — records only, no image paths to resolve)."""
    doc = {
        "artifact_schema": "mlforge.prepared_dataset.v1",
        "model": "calibrator",
        "transform": {"name": transform},
        "records": rows,
    }
    ident = ContentStore(Path(root) / "store").put_bytes(
        json.dumps(doc, sort_keys=True).encode("utf-8"))
    d = Path(root) / "datasets" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.json").write_text(json.dumps({
        "dataset_id": dataset, "identity": ident, "version": "v1",
        "file_count": len(rows), "total_bytes": 1,
    }), encoding="utf-8")


# -- registry / availability --------------------------------------------------


def test_registry_identity():
    assert "calibrator" in MODEL_REGISTRY
    assert MODEL_REGISTRY["calibrator"].fits == (
        "temperature", "conformal", "decomposition")
    from mlforge.trainers import availability_error, registered_models
    assert availability_error("calibrator") is None  # stdlib-only
    assert "calibrator" in registered_models()


def test_unknown_model_refuses(tmp_path):
    with pytest.raises(PreconditionFailed, match="no registered calibrator") as ei:
        build(tmp_path, model="temp_scaler")
    assert "calibrator" in (ei.value.hint or "")


# -- semantic contract (honored literally) ------------------------------------


def test_semantic_refusals(tmp_path):
    with pytest.raises(ValidationBlock, match="nll"):
        build(tmp_path, semantic=sem(loss="cross_entropy"))
    with pytest.raises(ValidationBlock, match="optimizer"):
        build(tmp_path, semantic=sem(optimizer="adamw"))
    with pytest.raises(ValidationBlock, match="scheduler"):
        build(tmp_path, semantic=sem(scheduler="cosine"))
    with pytest.raises(ValidationBlock, match="precision_policy"):
        build(tmp_path, semantic=sem(precision_policy="bf16"))


def test_epochs_is_the_phase_count(tmp_path):
    with pytest.raises(ValidationBlock, match="fit phases") as ei:
        build(tmp_path, semantic=sem(epochs=5))
    assert "epochs: 2" in str(ei.value)
    with pytest.raises(ValidationBlock, match="epochs: 3") as ei:
        build(tmp_path, semantic=sem(epochs=2, decomposition=True))
    assert "temperature, conformal, decomposition" in str(ei.value)


def test_alpha_and_flag_validation(tmp_path):
    with pytest.raises(ValidationBlock, match="alphas"):
        build(tmp_path, semantic=sem(alphas=[1.5]))
    with pytest.raises(ValidationBlock, match="alphas"):
        build(tmp_path, semantic=sem(alphas="0.05"))
    with pytest.raises(ValidationBlock, match="decomposition"):
        build(tmp_path, semantic=sem(decomposition="banana"))


def test_plan_guards(tmp_path):
    with pytest.raises(ValidationBlock, match="global-batch invariant"):
        build(tmp_path, plan=_Plan(micro_batch=2, grad_accum=3))  # 6 != 1
    with pytest.raises(PreconditionFailed, match="world_size 2"):
        build(tmp_path, semantic=sem(global_batch=4),
              plan=_Plan(world_size=2, micro_batch=1, grad_accum=2))
    with pytest.raises(PreconditionFailed, match="no AMP"):
        build(tmp_path, plan=_Plan(precision_effective="bf16"))


# -- fine-tune / payload guards ------------------------------------------------


def test_finetune_refused_with_guidance(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    with pytest.raises(ValidationBlock, match="fine-tune refused") as ei:
        build(tmp_path, init_weights=b"parent-model-bytes")
    assert "mlforge train" in (ei.value.hint or "")


def test_incomplete_payload_blocks(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    with pytest.raises(ValidationBlock, match="missing"):
        build(tmp_path, payloads={"model": b"{}"})


def test_payload_and_resume_mutually_exclusive(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    with pytest.raises(ValidationBlock, match="mutually exclusive"):
        build(tmp_path, payloads={"model": b"{}"}, init_weights=b"x")


# -- data guards ----------------------------------------------------------------


def test_unregistered_dataset_refuses(tmp_path):
    with pytest.raises(PreconditionFailed, match="not registered"):
        build(tmp_path)


def test_wrong_transform_blocks(tmp_path):
    make_calib_project(tmp_path, overconfident_rows(), transform="tabular")
    with pytest.raises(ValidationBlock, match="transform 'tabular'"):
        build(tmp_path)


def test_too_few_rows_and_single_class_block(tmp_path):
    make_calib_project(tmp_path, [{"logits": [1.0, -1.0], "label": 0}])
    with pytest.raises(ValidationBlock, match="at least 2 prediction rows"):
        build(tmp_path)
    make_calib_project(tmp_path, [
        {"logits": [1.0], "label": 0}, {"logits": [2.0], "label": 0}])
    with pytest.raises(ValidationBlock, match=">= 2"):
        build(tmp_path)


def test_decomposition_data_requirements(tmp_path):
    rows = overconfident_rows()  # no components anywhere
    make_calib_project(tmp_path, rows)
    with pytest.raises(ValidationBlock, match="decomposition: true needs"):
        build(tmp_path, semantic=sem(epochs=3, decomposition=True))
    # components present but the target is constant (all correct)
    const = [dict(r, components={c: 0.5 for c in COMPONENTS})
             for r in rows]
    for r in const:
        r["label"] = 0 if r["logits"][0] > 0 else 1  # always correct
    make_calib_project(tmp_path, const)
    with pytest.raises(ValidationBlock, match="BOTH correct and incorrect"):
        build(tmp_path, semantic=sem(epochs=3, decomposition=True))
    # too few rows for 5 weights
    few = component_rows(n=5)
    make_calib_project(tmp_path, few)
    with pytest.raises(ValidationBlock, match=">= 8 rows"):
        build(tmp_path, semantic=sem(epochs=3, decomposition=True))


# -- the fits (real numbers on synthetic pools) ---------------------------------


def test_temperature_fit_reduces_ece(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    t = build(tmp_path)
    state, results = run_steps(t, TrainState(0, 0), 1)
    metrics = results[0].metrics
    assert metrics["phase"] == "temperature"
    assert metrics["ece"] < metrics["ece_before"] - 0.05
    assert t.temperature > 1.2  # overconfident pool must warm up
    assert 0.1 <= t.temperature <= 10.0
    assert results[0].loss == pytest.approx(metrics["nll"])


def test_conformal_thresholds_meet_coverage(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    t = build(tmp_path)
    state, _ = run_steps(t, TrainState(0, 0), 2)
    assert state.epoch == 2
    for alpha in (0.05, 0.10):
        assert t.thresholds[alpha] > 0
        assert t.coverages[alpha] >= 1.0 - alpha  # by construction


def test_phase_sequence_and_done(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    t = build(tmp_path)
    state, results = run_steps(t, TrainState(0, 0), 2)
    assert [r.metrics["phase"] for r in results] == [
        "temperature", "conformal"]
    assert results[0].loss is not None and results[1].loss is None
    assert state.global_step == 2 and state.epoch == 2
    # done is sticky
    extra = t.step(state)
    assert extra.done and extra.loss is None


def test_decomposition_fit_prefers_planted_dominant_component(tmp_path):
    make_calib_project(tmp_path, component_rows())
    t = build(tmp_path, semantic=sem(epochs=3, decomposition=True))
    state, results = run_steps(t, TrainState(0, 0), 3)
    assert results[2].metrics["phase"] == "decomposition"
    assert max(t.weights, key=t.weights.get) == "perception"
    assert all(math.isfinite(v) for v in t.weights.values())
    assert results[2].metrics["residual_mse"] < 0.25


def test_solver_and_linear_fit_helpers():
    got = _solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0])
    assert got == pytest.approx([1.0, 2.0])
    weights, residual = _fit_decomposition(component_rows())
    assert set(weights) == set(COMPONENTS)
    assert residual >= 0.0


# -- payload / resume (12 §11.4) ------------------------------------------------


def test_payload_covers_required_components(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    t = build(tmp_path)
    state, _ = run_steps(t, TrainState(0, 0), 2)
    payload = t.checkpoint_payload(state)
    assert set(payload) == set(REQUIRED_COMPONENTS)
    model = json.loads(payload["model"])
    assert model["trainer"] == "calibrator"
    assert model["temperature"] == t.temperature
    optimizer = json.loads(payload["optimizer"])
    assert optimizer["enabled"] is False and optimizer["reason"]
    loader = json.loads(payload["dataloader_state"])
    assert loader["global_batch_lever"] is False
    assert loader["n"] == 40 and loader["classes"] == 2


def test_exact_resume_matches_uninterrupted_run(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    full = build(tmp_path)
    state_a, _ = run_steps(full, TrainState(0, 0), 2)

    # same experiment, checkpointed after phase 1
    first = build(tmp_path)
    state_b, _ = run_steps(first, TrainState(0, 0), 1)
    payload = first.checkpoint_payload(state_b)

    resumed = build(tmp_path, payloads=payload,
                    start_step=state_b.global_step,
                    start_epoch=state_b.epoch)
    assert resumed.temperature == full.temperature  # phase 1 state restored
    state_c, results = run_steps(resumed, state_b, 1)
    assert results[0].metrics["phase"] == "conformal"
    assert state_c.epoch == state_a.epoch == 2
    # identical fitted state — deterministic fits, restored pool
    assert resumed.thresholds == full.thresholds
    assert resumed.coverages == full.coverages


def test_resume_rejects_other_trainers_and_changed_pools(tmp_path):
    make_calib_project(tmp_path, overconfident_rows())
    t = build(tmp_path)
    state, _ = run_steps(t, TrainState(0, 0), 1)
    payload = t.checkpoint_payload(state)

    hijacked = dict(payload)
    hijacked["model"] = json.dumps(
        {"trainer": "text_lm", "temperature": 9.9}).encode()
    with pytest.raises(ValidationBlock, match="not produced by trainer"):
        build(tmp_path, payloads=hijacked, start_epoch=1)

    other = tmp_path / "other"
    other.mkdir()
    make_calib_project(other, overconfident_rows(n=41, seed=9))
    with pytest.raises(ValidationBlock, match="calibration pool"):
        build(other, payloads=payload, start_epoch=1,
              datasets=("preds:v1",))
