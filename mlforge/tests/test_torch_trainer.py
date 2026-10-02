"""The REAL torch learning loop (build step 6 — 12 §11.4 contents,
§12.4 injected trainer): real gradients, real loss decrease, real
resume continuation — the scaffold can no longer stand in for this."""

from __future__ import annotations

import json
import math

import pytest

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import TrainState, resolve_trainer, require_trainable
from mlforge.store import ContentStore
from mlforge.trainers import availability_error
from mlforge.trainers.text_lm import (
    MODEL_REGISTRY,
    TorchTextTrainer,
    TextArch,
)


@pytest.fixture
def tiny_model(monkeypatch):
    """A miniature architecture so CPU unit tests train in <1s."""
    monkeypatch.setitem(
        MODEL_REGISTRY, "unit_tiny_lm",
        TextArch(n_layer=1, n_head=2, d_model=32, block_size=64),
    )
    return "unit_tiny_lm"


def make_project(root, texts, name="books", version="v1"):
    """Registered + store-backed prepared text dataset (what
    `mlforge dataset add` + `mlforge prepare` leave behind)."""
    records = [
        {
            "source": f"{name}:train",
            "split": "train",
            "relative_path": f"t{i}.txt#0",
            "chars": len(t),
            "text": t,
        }
        for i, t in enumerate(texts)
    ]
    doc = {
        "artifact_schema": "mlforge.prepared_dataset.v1",
        "model": "unit",
        "records": records,
    }
    blob = json.dumps(doc, sort_keys=True).encode("utf-8")
    identity = ContentStore(root / "store").put_bytes(blob)
    reg = root / "datasets" / name
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "identity.json").write_text(json.dumps({
        "version": version,
        "identity": identity,
        "total_bytes": len(blob),
    }), encoding="utf-8")


PARAGRAPH = (
    "the quick system trains the model on honest gradients every step "
    "and the checkpoint keeps the optimizer state for the next run "
)

def sem(**over):
    base = {
        "optimizer": "adamw",
        "learning_rate": 3e-3,
        "scheduler": "cosine",
        "loss": "byte_cross_entropy",
        "seed": 42,
        "global_batch": 4,
        "epochs": 6,
        "precision_policy": "fp32",
        "windows_per_epoch": 64,
    }
    base.update(over)
    return base


def build(model, root, *, semantic=None, payloads=None, plan=None,
          datasets=("books:v1",)):
    return TorchTextTrainer(
        model_name=model,
        semantic=semantic or sem(),
        runtime={},
        train_datasets=datasets,
        root=root,
        payloads=payloads,
        plan=plan,
    )


def run_steps(trainer, state, n):
    losses = []
    for _ in range(n):
        result = trainer.step(state)
        state = TrainState(result.global_step, result.epoch, state.resume_from)
        if result.loss is not None:
            losses.append(result.loss)
    return state, losses


# -- learning --------------------------------------------------------------

def test_real_trainer_loss_decreases(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 60, PARAGRAPH * 40])
    trainer = build(tiny_model, tmp_path)
    state, losses = run_steps(trainer, TrainState(0, 0), 96)
    assert len(losses) == 96
    assert all(math.isfinite(v) for v in losses)
    # byte vocab: start ≈ ln(256) = 5.545 — real learning must pull it down
    assert losses[0] > 5.0
    assert losses[-1] < losses[0] - 0.15, (
        f"loss did not improve: first={losses[0]} last={losses[-1]}"
    )
    # gradient update actually moved the weights
    after = trainer.checkpoint_payload(state)
    assert after["model"]  # bytes exist


def test_step_result_progress_and_done(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 30])
    # steps_per_epoch = min(n_windows // gb, wpe // gb); wpe 64 / gb 4 = 16
    trainer = build(tiny_model, tmp_path, semantic=sem(epochs=2))
    state, _ = run_steps(trainer, TrainState(0, 0), 32)  # 2 epochs × 16
    assert state.epoch == 2
    # finished — further steps are a no-op with done=True
    result = trainer.step(state)
    assert result.done is True and result.loss is None


# -- checkpoint payload (12 §11.4) -----------------------------------------

def test_payload_covers_every_required_component(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 20])
    trainer = build(tiny_model, tmp_path)
    state, _ = run_steps(trainer, TrainState(0, 0), 3)
    payload = trainer.checkpoint_payload(TrainState(state.global_step,
                                                    state.epoch))
    assert set(payload) == set(REQUIRED_COMPONENTS)
    # honest disabled flags for capabilities not integrated
    assert json.loads(payload["amp_scaler"]) == {"enabled": False}
    assert json.loads(payload["ema"]) == {"enabled": False}
    assert json.loads(payload["distributed_state"]) == {"world_size": 1}
    # real state, not a stand-in
    assert len(payload["model"]) > 1000
    assert len(payload["optimizer"]) > 1000


def test_resume_continues_exactly(tmp_path, tiny_model):
    """kill → payload → fresh trainer → SAME weights → identical next
    steps (the real meaning of `mlforge resume`)."""
    make_project(tmp_path, [PARAGRAPH * 50])
    a = build(tiny_model, tmp_path)
    state, _ = run_steps(a, TrainState(0, 0), 8)
    payload = a.checkpoint_payload(state)

    b = build(tiny_model, tmp_path, payloads=payload)
    ta = a.model.state_dict()
    tb = b.model.state_dict()
    assert all(torch_equal(ta[k], tb[k]) for k in ta), "weights diverge on restore"
    oa = a.optimizer.state_dict()
    ob = b.optimizer.state_dict()
    assert oa["param_groups"][0]["lr"] == ob["param_groups"][0]["lr"]
    assert a.scheduler.last_epoch == b.scheduler.last_epoch

    # identical continuation: same state in, same losses out
    _, losses_a = run_steps(a, state, 5)
    _, losses_b = run_steps(b, state, 5)
    assert losses_a == losses_b, "resume replays a different trajectory"


def torch_equal(x, y):
    import torch

    return torch.equal(x, y)


def test_resume_payload_missing_component_blocks(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 20])
    trainer = build(tiny_model, tmp_path)
    state, _ = run_steps(trainer, TrainState(0, 0), 2)
    payload = trainer.checkpoint_payload(state)
    del payload["optimizer"]
    with pytest.raises(ValidationBlock) as exc:
        build(tiny_model, tmp_path, payloads=payload)
    assert "missing" in str(exc.value)


def test_harness_payload_cannot_resume_into_real_trainer(tmp_path, tiny_model):
    """State is never faked into compatibility."""
    make_project(tmp_path, [PARAGRAPH * 20])
    fake = {c: json.dumps({"component": c}).encode() for c in REQUIRED_COMPONENTS}
    with pytest.raises(ValidationBlock) as exc:
        build(tiny_model, tmp_path, payloads=fake)
    assert "not loadable" in str(exc.value)


# -- semantic identity honored literally -----------------------------------

def test_unsupported_semantic_loss_blocks(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 10])
    with pytest.raises(ValidationBlock) as exc:
        build(tiny_model, tmp_path, semantic=sem(loss="l1"))
    assert "semantic.loss 'l1'" in str(exc.value)


def test_unsupported_optimizer_blocks(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 10])
    with pytest.raises(ValidationBlock):
        build(tiny_model, tmp_path, semantic=sem(optimizer="lion"))


# -- plan contract (§13/§11 execution plan) --------------------------------

class _Plan:
    def __init__(self, micro=2, accum=2, world=1, precision="fp32"):
        self.micro_batch = micro
        self.grad_accum = accum
        self.world_size = world
        self.precision_effective = precision


def test_plan_world_size_one_only(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 10])
    with pytest.raises(PreconditionFailed) as exc:
        build(tiny_model, tmp_path, plan=_Plan(world=2, micro=1, accum=2))
    assert "distributed" in str(exc.value)


def test_plan_precision_fp32_only(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 10])
    with pytest.raises(PreconditionFailed) as exc:
        build(tiny_model, tmp_path, plan=_Plan(precision="bf16"))
    assert "AMP" in str(exc.value)


def test_plan_global_batch_invariant(tmp_path, tiny_model):
    make_project(tmp_path, [PARAGRAPH * 10])
    with pytest.raises(ValidationBlock) as exc:
        build(tiny_model, tmp_path, plan=_Plan(micro=3, accum=2, world=1))
    assert "global-batch" in str(exc.value)


# -- resolution / availability (fail-closed) -------------------------------

def test_resolve_trainer_selects_real_by_model(tmp_path, tiny_model, monkeypatch):
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    make_project(tmp_path, [PARAGRAPH * 10])
    trainer = resolve_trainer(
        {}, model=tiny_model, semantic=sem(),
        train_datasets=["books:v1"], root=tmp_path,
    )
    assert isinstance(trainer, TorchTextTrainer)


def test_resolve_unknown_model_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    with pytest.raises(PreconditionFailed) as exc:
        resolve_trainer({}, model="rf_detr_s", root=tmp_path)
    out = exc.value.render()
    assert "no real trainer integrated for model 'rf_detr_s'" in out
    assert "harness-scaffold" in out  # hint still names the only opt-in


def test_require_trainable_is_cheap_and_model_aware(monkeypatch):
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    require_trainable({}, model="reasoner_s")  # torch present → ok
    with pytest.raises(PreconditionFailed):
        require_trainable({}, model="rf_detr_s")
    with pytest.raises(PreconditionFailed) as exc:
        require_trainable({})
    assert "no model context" in str(exc.value)


def test_missing_torch_message_is_honest(monkeypatch):
    from mlforge.trainers import text_lm

    monkeypatch.setattr(text_lm, "torch_available", lambda: False)
    err = availability_error("reasoner_s")
    assert err and "torch is not installed" in err
    assert "pip install torch" in err


def test_unregistered_model_error_lists_trainable(monkeypatch):
    err = availability_error("some_future_model")
    assert "no integrated trainer" in err
    assert "reasoner_s" in err


def test_missing_corpus_names_prepare(tmp_path, tiny_model, monkeypatch):
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    with pytest.raises(PreconditionFailed) as exc:
        resolve_trainer(
            {}, model=tiny_model, semantic=sem(),
            train_datasets=["books:v1"], root=tmp_path,
        )
    assert "mlforge dataset add" in str(exc.value)
