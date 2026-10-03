"""RF-DETR trainer (file 11 §C1/§C3) — registry identity, literal
semantic refusals, and the COCO materializer (12 §11.4, §12.4).

The constructor validates pure config (variant → semantic → plan) BEFORE
touching the framework, so refusal tests run without rfdetr installed;
tests that genuinely need it skip honestly. The real-epoch smoke is gated
behind MLFORGE_RFDETR_SMOKE=1 (downloads the published base and runs a
full Lightning epoch on CPU — run it with TMPDIR on a roomy disk)."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash_bytes
from mlforge.ingest import config as ingest_config
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import TrainState
from mlforge.store import ContentStore
from mlforge.trainers import rfdetr as rfdetr_mod
from mlforge.trainers.rfdetr import MODEL_REGISTRY, RFDETRTrainer
from mlforge.trainers.rfdetr_data import materialize


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
        "learning_rate": 1e-4,
        "scheduler": "cosine",
        "loss": "l1",
        "seed": 42,
        "global_batch": 4,
        "epochs": 2,
        "precision_policy": "fp32",
    }
    base.update(over)
    return base


def build(root, *, model="rf_detr_s", semantic=None, run_dir="auto", **kw):
    """Direct construction (what resolve_trainer hands the worker).
    Refusals fire before any side effect — no dataset/run dir needed."""
    if run_dir == "auto":
        run_dir = Path(root) / "run"
    return RFDETRTrainer(
        model_name=model,
        semantic=semantic if semantic is not None else sem(),
        runtime=kw.pop("runtime", {}),
        train_datasets=kw.pop("datasets", ()),
        root=Path(root),
        run_dir=run_dir,
        **kw,
    )


class _FrameworkProbe:
    """dependency_error() must NOT be reached by a pure-config refusal."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str | None:
        self.calls += 1
        return "rfdetr is not installed — pip install rfdetr"


class _Plan:
    """Execution-plan stub (12 §11.4 invariants checked by the trainer)."""

    def __init__(self, **over) -> None:
        self.world_size = over.get("world_size", 1)
        self.micro_batch = over.get("micro_batch", 4)
        self.grad_accum = over.get("grad_accum", 1)
        self.precision_effective = over.get("precision_effective", "fp32")


def _require_rfdetr() -> None:
    err = rfdetr_mod.dependency_error()
    if err:
        pytest.skip(f"rfdetr unavailable here: {err}")


# -- prepared COCO fixture (what `mlforge prepare` leaves behind) ----------


def _coco_tree(base: Path, *, n_train: int = 4, n_valid: int = 2) -> Path:
    """Roboflow-style source tree: train/ + valid/ with COCO annotations."""
    def _ann(ids: list[int], names: list[str]) -> bytes:
        return json.dumps({
            "images": [{"id": i, "file_name": n, "width": 32, "height": 32}
                       for i, n in zip(ids, names)],
            "annotations": [{"id": i, "image_id": i, "category_id": 1,
                             "bbox": [0, 0, 16, 16], "area": 256,
                             "iscrowd": 0}
                            for i in ids],
            "categories": [{"id": 1, "name": "widget"}],
        }, sort_keys=True).encode("utf-8")

    for split, n, color in (("train", n_train, (200, 40, 40)),
                            ("valid", n_valid, (40, 40, 200))):
        d = base / split
        d.mkdir(parents=True, exist_ok=True)
        ids, names = [], []
        for i in range(n):
            name = f"img{i}.jpg"
            Image.new("RGB", (32, 32), color).save(d / name, "JPEG")
            ids.append(i + 1)
            names.append(name)
        (d / "_annotations.coco.json").write_bytes(_ann(ids, names))
    return base


def _prepared(root: Path, tree: Path, *, transform: str = "coco_detection",
              version: str = "v1") -> None:
    """Store-backed prepared artifact + registration + machine-local path."""
    records = []
    for rel in sorted(p.relative_to(tree).as_posix()
                      for p in tree.rglob("*") if p.is_file()):
        data = (tree / rel).read_bytes()
        split = rel.split("/", 1)[0]
        records.append({
            "source": f"coco:{split}",
            "split": "train" if split == "train" else "val",
            "relative_path": rel,
            "sha256": content_hash_bytes(data),
            "size": len(data),
        })
    doc = {
        "artifact_schema": "mlforge.prepared_dataset.v1",
        "model": "rf_detr_s",
        "transform": {"name": transform},
        "records": records,
    }
    ident = ContentStore(root / "store").put_bytes(
        json.dumps(doc, sort_keys=True).encode("utf-8"))
    d = root / "datasets" / "coco"
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.json").write_text(json.dumps({
        "dataset_id": "coco", "identity": ident, "version": version,
        "file_count": len(records), "total_bytes": 1,
    }), encoding="utf-8")
    ingest_config.set_path(root, "coco", tree)


# -- registry identity ------------------------------------------------------


def test_registry_matches_file11_targets():
    assert set(MODEL_REGISTRY) == {"rf_detr_s", "rf_detr_l", "rf_detr_seg_s"}
    assert MODEL_REGISTRY["rf_detr_s"].class_name == "RFDETRSmall"
    assert MODEL_REGISTRY["rf_detr_l"].class_name == "RFDETRLarge"
    assert MODEL_REGISTRY["rf_detr_seg_s"].class_name == "RFDETRSegSmall"
    assert MODEL_REGISTRY["rf_detr_s"].task == "detection"
    assert MODEL_REGISTRY["rf_detr_seg_s"].task == "segmentation"
    # the class names must exist in _MODEL_CLASSES (what the dep probe
    # checks against the INSTALLED package)
    from mlforge.trainers.rfdetr import _MODEL_CLASSES
    for name, variant in MODEL_REGISTRY.items():
        assert _MODEL_CLASSES[name] == variant.class_name


# -- pure-config refusals (framework never imported) ------------------------


def test_unknown_model_refuses_before_framework(tmp_path, monkeypatch):
    probe = _FrameworkProbe()
    monkeypatch.setattr(rfdetr_mod, "dependency_error", probe)
    with pytest.raises(PreconditionFailed,
                       match="no registered RF-DETR variant") as ei:
        build(tmp_path, model="rf_detr_xl")
    assert "rf_detr_s" in (ei.value.hint or "")  # what IS trainable here
    assert probe.calls == 0


def test_semantic_refusals_before_framework(tmp_path, monkeypatch):
    probe = _FrameworkProbe()
    monkeypatch.setattr(rfdetr_mod, "dependency_error", probe)
    with pytest.raises(ValidationBlock, match="loss"):
        build(tmp_path, semantic=sem(loss="mse"))
    with pytest.raises(ValidationBlock, match="optimizer"):
        build(tmp_path, semantic=sem(optimizer="sgd"))
    with pytest.raises(ValidationBlock, match="scheduler"):
        build(tmp_path, semantic=sem(scheduler="step"))
    missing = {k: v for k, v in sem().items() if k != "epochs"}
    with pytest.raises(ValidationBlock, match="unreadable"):
        build(tmp_path, semantic=missing)
    with pytest.raises(ValidationBlock, match="epochs >= 1"):
        build(tmp_path, semantic=sem(epochs=0))
    with pytest.raises(ValidationBlock, match="learning_rate"):
        build(tmp_path, semantic=sem(learning_rate=0))
    assert probe.calls == 0  # config errors, not environment errors


def test_precision_policy_refuses_before_framework(tmp_path, monkeypatch):
    probe = _FrameworkProbe()
    monkeypatch.setattr(rfdetr_mod, "dependency_error", probe)
    with pytest.raises(ValidationBlock, match="fp9"):
        build(tmp_path, semantic=sem(precision_policy="fp9"))
    assert probe.calls == 0


def test_plan_invariants_refuse_before_framework(tmp_path, monkeypatch):
    probe = _FrameworkProbe()
    monkeypatch.setattr(rfdetr_mod, "dependency_error", probe)
    with pytest.raises(ValidationBlock, match="global-batch invariant"):
        build(tmp_path, plan=_Plan(micro_batch=3, grad_accum=1))
    with pytest.raises(PreconditionFailed, match="world_size 2"):
        build(tmp_path, plan=_Plan(micro_batch=1, grad_accum=2,
                                   world_size=2))
    assert probe.calls == 0


def test_framework_missing_is_honest(tmp_path, monkeypatch):
    """With valid config, a missing framework names the install command
    (never a silent scaffold, 12 §12.4)."""
    monkeypatch.setattr(
        rfdetr_mod, "dependency_error",
        lambda: "rfdetr is not installed — pip install rfdetr",
    )
    with pytest.raises(PreconditionFailed, match="pip install rfdetr") as ei:
        build(tmp_path)
    assert 'rfdetr[train]' in (ei.value.hint or "")


def test_run_dir_required(tmp_path):
    """Side-effect-free until a run directory exists (12 §12.4)."""
    _require_rfdetr()
    with pytest.raises(ValidationBlock, match="run directory"):
        build(tmp_path, run_dir=None)


def test_prefetch_resolves_bare_default_to_model_cache(tmp_path, monkeypatch):
    """The published base lands in rfdetr's model cache (RF_HOME), never
    the working directory — no stray multi-hundred-MB .pth dropped in the
    user's project, and no duplicate download when the model itself
    resolves the same bare default to the cache."""
    _require_rfdetr()
    import rfdetr.assets.model_weights as mw

    calls: list[str] = []
    monkeypatch.setattr(mw, "get_model_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mw, "download_pretrain_weights", calls.append)
    # only the NAME matters for the config lookup (rfdetr.<name>Config)
    cls = type("RFDETRSmall", (), {})
    stub = RFDETRTrainer.__new__(RFDETRTrainer)
    stub._prefetch_published_base(cls)
    assert len(calls) == 1
    resolved = Path(calls[0])
    assert resolved.parent == tmp_path  # cache dir, not CWD
    assert resolved.name == "rf-detr-small.pth"
    assert resolved.parent.is_dir()  # cache dir created before the fetch


def test_cross_trainer_payload_blocked():
    """A resume continues the SAME experiment (12 §11.4) — the cheap
    dataloader_state gate fires before any 490MB checkpoint load."""
    stub = RFDETRTrainer.__new__(RFDETRTrainer)  # guard needs model_name only
    stub.model_name = "rf_detr_s"
    with pytest.raises(ValidationBlock, match="not produced by trainer"):
        stub._assert_own_payload(
            {"dataloader_state": json.dumps({"trainer": "text_lm"}).encode()}
        )
    with pytest.raises(ValidationBlock, match="not produced by trainer"):
        stub._assert_own_payload({})  # harness/text payloads: no marker
    with pytest.raises(ValidationBlock, match="not produced by trainer"):
        # right trainer, DIFFERENT variant — still another experiment
        stub._assert_own_payload({"dataloader_state": json.dumps(
            {"trainer": "rfdetr", "variant": "rf_detr_l"}).encode()})


# -- materializer (pure MLForge — no framework involved) --------------------


def test_materialize_roboflow_layout(tmp_path, home):
    root = tmp_path
    tree = _coco_tree(tmp_path / "tree")
    _prepared(root, tree)
    out = materialize(root, tmp_path / "run", ("coco:v1",))
    assert out.layout == "roboflow"
    assert (out.train_images, out.valid_images) == (4, 2)
    assert out.classes == 1 and out.categories == ("widget",)
    train_ann = out.dir / "train" / "_annotations.coco.json"
    assert train_ann.is_file()
    assert (out.dir / "valid" / "_annotations.coco.json").is_file()
    # images are symlinks to the registered path (verify, never scan)
    link = out.dir / "train" / "img0.jpg"
    assert link.is_symlink()
    assert link.resolve() == (tree / "train" / "img0.jpg").resolve()
    # rebuild-from-scratch is idempotent (run-scoped scratch, 13 §10)
    again = materialize(root, tmp_path / "run", ("coco:v1",))
    assert (again.train_images, again.valid_images) == (4, 2)
    # annotations are COPIES — later source edits cannot touch a running
    # run (a REBUILD re-verifying the drift is the drift test's job)
    original = train_ann.read_bytes()
    (tree / "train" / "_annotations.coco.json").write_bytes(b'{"drift": 1}')
    assert train_ann.read_bytes() == original


def test_materialize_requires_valid_split(tmp_path, home):
    tree = _coco_tree(tmp_path / "tree")
    shutil.rmtree(tree / "valid")  # dataset without a valid split at all
    _prepared(tmp_path, tree)
    with pytest.raises(ValidationBlock, match="valid-split"):
        materialize(tmp_path, tmp_path / "run", ("coco:v1",))


def test_materialize_annotation_drift_blocks(tmp_path, home):
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)
    ann = tree / "train" / "_annotations.coco.json"
    doc = json.loads(ann.read_bytes())
    doc["images"][0]["file_name"] = "other.jpg"  # content changed since prepare
    ann.write_bytes(json.dumps(doc).encode("utf-8"))
    with pytest.raises(ValidationBlock, match="no longer matches"):
        materialize(tmp_path, tmp_path / "run", ("coco:v1",))


def test_materialize_wrong_transform_blocks(tmp_path, home):
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree, transform="reid_crops")
    with pytest.raises(ValidationBlock, match="coco_detection"):
        materialize(tmp_path, tmp_path / "run", ("coco:v1",))


def test_materialize_unregistered_blocks(tmp_path, home):
    tree = _coco_tree(tmp_path / "tree")
    with pytest.raises(PreconditionFailed, match="not registered"):
        materialize(tmp_path, tmp_path / "run", ("coco:v1",))


def test_materialize_single_dataset_only(tmp_path, home):
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)
    with pytest.raises(ValidationBlock, match="ONE prepared dataset"):
        materialize(tmp_path, tmp_path / "run", ("coco:v1", "coco:v1"))


# -- gated E2E: one real Lightning epoch + resume ---------------------------


@pytest.mark.skipif(
    os.environ.get("MLFORGE_RFDETR_SMOKE") != "1",
    reason="set MLFORGE_RFDETR_SMOKE=1 for the full RF-DETR epoch smoke "
           "(downloads rf-detr-small base; run with TMPDIR on a roomy disk)",
)
def test_e2e_epoch_payload_and_resume(tmp_path, home):
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)
    cfg = dict(epochs=2, global_batch=2)
    t = build(tmp_path, run_dir=tmp_path / "run", datasets=("coco:v1",),
              semantic=sem(**cfg))
    assert t.checkpointable is False  # fresh run: nothing serializable yet

    res = t.step(TrainState(0, 0))     # ONE full Lightning epoch
    assert t.checkpointable is True    # the epoch wrote a framework ckpt
    assert res.epoch == 1 and res.done is False
    assert res.loss is not None and math.isfinite(res.loss)

    payload = t.checkpoint_payload(TrainState(res.global_step, res.epoch))
    assert set(payload) == set(REQUIRED_COMPONENTS)
    ds = json.loads(payload["dataloader_state"])
    assert ds["trainer"] == "rfdetr" and ds["variant"] == "rf_detr_s"
    assert len(payload["model"]) > 1_000_000  # real Lightning checkpoint

    # a FRESH trainer resumes the SAME experiment (12 §11.4)
    t2 = build(tmp_path, run_dir=tmp_path / "run2", datasets=("coco:v1",),
               payloads=payload, start_step=res.global_step,
               start_epoch=res.epoch, semantic=sem(**cfg))
    assert t2.checkpointable is True
    res2 = t2.step(TrainState(res.global_step, res.epoch))
    assert res2.epoch == 2 and res2.done is True
    assert res2.loss is not None and math.isfinite(res2.loss)
