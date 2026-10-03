"""OSNet re-ID trainer (file 11 §M5) — architecture invariants, the
micro-batch BatchNorm guard, cross-trainer refusal, real loss decrease,
and exact payload resume (12 §11.4).

The fixture is a miniature Market1501-style crop set (3 identities × 8
images, deterministic colors) so CPU steps stay in the ~0.3s range while
still training REAL gradients through OSNet x1_0 at 256×128."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.ingest import config as ingest_config
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import TrainState
from mlforge.store import ContentStore
from mlforge.trainers.reid import MODEL_REGISTRY, OSNet, ReidTrainer


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """Machine-local dataset paths live outside the workspace (13 §10)."""
    h = tmp_path / "mlforge-home"
    monkeypatch.setenv("MLFORGE_HOME", str(h))
    return h


def sem(**over) -> dict:
    base = {
        "optimizer": "adamw",
        "learning_rate": 3e-3,
        "scheduler": "cosine",
        "loss": "cross_entropy",
        "seed": 42,
        "global_batch": 4,
        "epochs": 12,
        "precision_policy": "fp32",
        "pretrained": False,   # no network in unit tests (M5 is gated)
    }
    base.update(over)
    return base


def build(root, *, semantic=None, datasets=("market:v1",), **kw):
    return ReidTrainer(
        model_name=kw.pop("model", "osnet_x1_0"),
        semantic=semantic if semantic is not None else sem(),
        runtime=kw.pop("runtime", {}),
        train_datasets=datasets,
        root=Path(root),
        **kw,
    )


def run_steps(trainer, state, n):
    losses = []
    for _ in range(n):
        result = trainer.step(state)
        state = TrainState(result.global_step, result.epoch, state.resume_from)
        if result.loss is not None:
            losses.append(result.loss)
    return state, losses


class _Plan:
    """Execution-plan stub — mirrors the planner's attribute surface."""

    def __init__(self, **over) -> None:
        self.world_size = over.get("world_size", 1)
        self.micro_batch = over.get("micro_batch", 4)
        self.grad_accum = over.get("grad_accum", 1)
        self.precision_effective = over.get("precision_effective", "fp32")


def make_reid_project(root: Path, *, n_ids: int = 3, per_id: int = 8) -> None:
    """Registered + store-backed prepared `reid_crops` dataset + the
    machine-local image path (what dataset add + prepare leave behind)."""
    tree = Path(root) / "market"
    tree.mkdir(parents=True, exist_ok=True)
    palette = {1: (200, 40, 40), 2: (40, 40, 200), 3: (40, 200, 40)}
    records = []
    for pid in range(1, n_ids + 1):
        for i in range(per_id):
            rel = f"{pid}_c1_{i:04d}.jpg"
            jitter = (i * 3) % 21  # per-image variation (BN needs variance)
            color = tuple(min(255, c + jitter) for c in palette[pid])
            Image.new("RGB", (64, 32), color).save(
                tree / rel, "JPEG", quality=85
            )
            records.append({
                "source": "market:train",
                "split": "train",
                "relative_path": rel,
                "identity": str(pid),
                "camera": 1,
            })
    doc = {
        "artifact_schema": "mlforge.prepared_dataset.v1",
        "model": "osnet_x1_0",
        "transform": {"name": "reid_crops"},
        "records": records,
    }
    ident = ContentStore(Path(root) / "store").put_bytes(
        json.dumps(doc, sort_keys=True).encode("utf-8")
    )
    d = Path(root) / "datasets" / "market"
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.json").write_text(json.dumps({
        "dataset_id": "market", "identity": ident, "version": "v1",
        "file_count": len(records), "total_bytes": 1,
    }), encoding="utf-8")
    ingest_config.set_path(root, "market", tree)


# -- architecture invariants (file 11 §M5) ---------------------------------


def test_osnet_arch_invariants():
    arch = MODEL_REGISTRY["osnet_x1_0"]
    assert arch.channels == (64, 256, 384, 512)
    assert arch.layers == (2, 2, 2)
    assert arch.feature_dim == 512
    assert arch.input_hw == (256, 128)  # re-ID standard input
    m = OSNet(arch, 1000)
    sd = m.state_dict()
    assert len(sd) == 567  # M5 tensor surface (loads official weights)
    non_head = sum(v.numel() for k, v in sd.items()
                   if not k.startswith("classifier"))
    assert non_head == 2_192_628  # file 11's "2.2M" for osnet_x1_0
    assert tuple(sd["classifier.weight"].shape) == (1000, 512)


# -- config refusals (no dataset needed — they fire first) -----------------


def test_unknown_model_refuses(tmp_path):
    with pytest.raises(PreconditionFailed, match="re-ID architecture") as ei:
        build(tmp_path, model="osnet_x0_5")
    assert "osnet_x1_0" in (ei.value.hint or "")


def test_semantic_refusals(tmp_path):
    with pytest.raises(ValidationBlock, match="cross_entropy"):
        build(tmp_path, semantic=sem(loss="focal"))
    with pytest.raises(ValidationBlock, match="optimizer"):
        build(tmp_path, semantic=sem(optimizer="rmsprop"))
    with pytest.raises(ValidationBlock, match="scheduler"):
        build(tmp_path, semantic=sem(scheduler="plateau"))
    missing = {k: v for k, v in sem().items() if k != "seed"}
    with pytest.raises(ValidationBlock, match="unreadable"):
        build(tmp_path, semantic=missing)
    with pytest.raises(ValidationBlock, match="epochs >= 1"):
        build(tmp_path, semantic=sem(epochs=0))


def test_micro_batch_below_two_blocks(tmp_path):
    """OSNet is BatchNorm-heavy: one sample per micro-batch dies in train
    mode ("Expected more than 1 value"), so refuse up front (12 §11.4)."""
    with pytest.raises(ValidationBlock, match="micro_batch 1 < 2") as ei:
        build(tmp_path, semantic=sem(global_batch=1))
    assert "global_batch" in (ei.value.hint or "")
    # an explicit plan cannot sneak micro=1 in either
    with pytest.raises(ValidationBlock, match="micro_batch 1 < 2"):
        build(tmp_path, plan=_Plan(micro_batch=1, grad_accum=4))


# -- checkpoint payload (12 §11.4) -----------------------------------------


def test_payload_covers_every_required_component(tmp_path, home):
    make_reid_project(tmp_path)
    trainer = build(tmp_path)
    state, _ = run_steps(trainer, TrainState(0, 0), 2)
    payload = trainer.checkpoint_payload(state)
    assert set(payload) == set(REQUIRED_COMPONENTS)
    # honest disabled flags for capabilities not integrated
    assert json.loads(payload["amp_scaler"]) == {"enabled": False}
    assert json.loads(payload["ema"]) == {"enabled": False}
    assert json.loads(payload["distributed_state"]) == {"world_size": 1}
    # real state, not a stand-in
    assert json.loads(payload["global_step"]) == state.global_step
    assert len(payload["model"]) > 10_000
    assert len(payload["optimizer"]) > 1_000
    ids = json.loads(payload["dataloader_state"])
    assert ids["id_map"] == {"1": "0", "2": "1", "3": "2"}


def test_resume_payload_missing_component_blocks(tmp_path, home):
    make_reid_project(tmp_path)
    payload = build(tmp_path).checkpoint_payload(TrainState(0, 0))
    del payload["optimizer"]
    with pytest.raises(ValidationBlock) as ei:
        build(tmp_path, payloads=payload)
    assert "missing" in str(ei.value)


def test_harness_payload_cannot_resume_into_real_trainer(tmp_path, home):
    """State is never faked into compatibility (12 §11.4)."""
    make_reid_project(tmp_path)
    fake = {c: json.dumps({"component": c}).encode()
            for c in REQUIRED_COMPONENTS}
    with pytest.raises(ValidationBlock, match="not loadable"):
        build(tmp_path, payloads=fake)


def test_resume_continues_exactly(tmp_path, home):
    """kill → payload → fresh trainer → SAME weights → identical next
    steps (the real meaning of `mlforge resume`)."""
    import torch

    make_reid_project(tmp_path)
    a = build(tmp_path)
    state, _ = run_steps(a, TrainState(0, 0), 8)
    payload = a.checkpoint_payload(state)

    b = build(tmp_path, payloads=payload,
              start_step=state.global_step, start_epoch=state.epoch)
    ta, tb = a.model.state_dict(), b.model.state_dict()
    assert all(torch.equal(ta[k], tb[k]) for k in ta), \
        "weights diverge on restore"
    oa = a.optimizer.state_dict()
    ob = b.optimizer.state_dict()
    assert oa["param_groups"][0]["lr"] == ob["param_groups"][0]["lr"]
    assert a.scheduler.last_epoch == b.scheduler.last_epoch

    # identical continuation: same state in, same losses out
    _, losses_a = run_steps(a, state, 5)
    _, losses_b = run_steps(b, state, 5)
    assert losses_a == losses_b, "resume replays a different trajectory"


# -- learning --------------------------------------------------------------


def test_real_loss_decreases(tmp_path, home):
    make_reid_project(tmp_path)
    trainer = build(tmp_path)
    state, losses = run_steps(trainer, TrainState(0, 0), 32)
    assert len(losses) == 32
    assert all(math.isfinite(v) for v in losses)
    first8 = sum(losses[:8]) / 8
    last8 = sum(losses[-8:]) / 8
    # seeded trajectory on this fixture: 1.31 → 0.53 — real learning must
    # pull the cross-entropy down, and 3-class chance (ln 3 ≈ 1.10) is
    # decisively beaten
    assert last8 < first8
    assert last8 < 1.0
