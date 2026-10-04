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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash_bytes
from mlforge.ingest import config as ingest_config
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import TrainState
from mlforge.store import ContentStore
from mlforge.trainers import rfdetr as rfdetr_mod
from mlforge.trainers.rfdetr import MODEL_REGISTRY, OUT_DIRNAME, RFDETRTrainer
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
    _coco_tree(tmp_path / "tree")
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
    cfg = {"epochs": 2, "global_batch": 2}
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


# -- CPU-only adapter depth: dependency probes, init branches, state, step --
#
# The real-epoch smoke above stays gated; everything BELOW runs on CPU by
# swapping the RF-DETR constructor for a recording fake and stubbing
# rfdetr's build_trainer seam, so every adapter branch (init refusals,
# resume/fine-tune/fresh, state plumbing, step orchestration, payload)
# is exercised without downloading or building the published model.


def test_dependency_error_names_each_missing_layer(monkeypatch):
    """Each probe layer (package → training extra → model classes) gets
    its own honest install instruction."""
    _require_rfdetr()
    real_pkg = sys.modules["rfdetr"]
    real_training = sys.modules["rfdetr.training"]
    saved_cache = list(rfdetr_mod._DEP_CACHE)
    try:
        rfdetr_mod._DEP_CACHE.clear()
        monkeypatch.setitem(sys.modules, "rfdetr", None)  # import halted
        err = rfdetr_mod.dependency_error()
        assert err is not None and "pip install rfdetr" in err

        rfdetr_mod._DEP_CACHE.clear()
        monkeypatch.setitem(sys.modules, "rfdetr", real_pkg)
        monkeypatch.setitem(sys.modules, "rfdetr.training", None)
        err = rfdetr_mod.dependency_error()
        assert err is not None and "rfdetr[train]" in err

        rfdetr_mod._DEP_CACHE.clear()
        monkeypatch.setitem(sys.modules, "rfdetr.training", real_training)
        monkeypatch.setattr(
            rfdetr_mod, "_MODEL_CLASSES",
            {**rfdetr_mod._MODEL_CLASSES, "zz_ghost": "GhostClass"},
        )
        err = rfdetr_mod.dependency_error()
        assert err is not None and "lacks GhostClass" in err
    finally:
        rfdetr_mod._DEP_CACHE[:] = saved_cache


def test_pulse_callback_forwards_epoch_boundaries():
    pytest.importorskip("pytorch_lightning")
    seen: list[str] = []
    pulse = rfdetr_mod._pulse_callback(seen.append)
    pulse.on_train_batch_end()
    pulse.on_train_epoch_start()
    pulse.on_validation_batch_end()
    pulse.on_validation_epoch_start()
    assert seen == [
        "train_batch_end", "train_epoch_start",
        "validation_batch_end", "validation_epoch_start",
    ]


def _fake_rf_detr(monkeypatch, tmp_path, *, fit: str = "archive"):
    """Install a recording RFDETRSmall + a stub Lightning build_trainer.
    `fit` selects what the fake epoch writes: an epoch archive, only
    last.ckpt, or nothing at all (the missing-checkpoint refusal)."""
    import rfdetr as rfdetr_pkg
    import rfdetr.assets.model_weights as mw
    import rfdetr.training as rtrain

    monkeypatch.setattr(mw, "get_model_cache_dir",
                        lambda: str(tmp_path / "model-cache"))
    monkeypatch.setattr(mw, "download_pretrain_weights", lambda target: None)
    made: list[SimpleNamespace] = []

    def fake_build(tc, mc, **kw):
        tr = SimpleNamespace(
            callbacks=[],
            callback_metrics={
                "train/loss": 0.5,
                "val/loss": 0.25,
                "val/mAR": 0.75,
                "val/mAP": "not-a-number",
                "train/lr": float("nan"),
                "epoch": 3,  # wrong prefix: must be filtered out
            },
        )
        made.append(tr)
        return tr

    monkeypatch.setattr(rtrain, "build_trainer", fake_build)

    class _FakeRFDETR:
        def __init__(self, **kw) -> None:
            self.init_kwargs = kw
            self.fit_configs: list[dict] = []

        def train(self, **cfg) -> None:
            self.fit_configs.append(cfg)
            rtrain.build_trainer("tc", "mc")  # enter the installed seam
            out = Path(cfg["output_dir"])
            target = int(cfg["epochs"])
            if fit == "archive":
                (out / f"checkpoint_{target - 1}.ckpt").write_bytes(b"ckpt")
            elif fit == "last":
                (out / "last.ckpt").write_bytes(b"ckpt-last")

    monkeypatch.setattr(rfdetr_pkg, "RFDETRSmall", _FakeRFDETR)
    return made


def test_fresh_init_scrubs_stale_and_honors_prime_batch(tmp_path, home,
                                                        monkeypatch):
    """Fresh construction: global_batch that no micro/accum split divides
    (line: fall back to micro=global_batch), a dict precision_policy,
    stale scratch files scrubbed, and a no-kwarg (scratch) model init."""
    _require_rfdetr()
    _fake_rf_detr(monkeypatch, tmp_path)
    out_dir = tmp_path / "run" / OUT_DIRNAME
    out_dir.mkdir(parents=True)
    (out_dir / "stale.ckpt").write_bytes(b"stale")
    (out_dir / "stale.pth").write_bytes(b"stale")
    (out_dir / "_mlforge_resume.json").write_text("{}", encoding="utf-8")
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)

    t = build(tmp_path, run_dir=tmp_path / "run", datasets=("coco:v1",),
              semantic=sem(global_batch=5,
                           precision_policy={"preferred": "fp32",
                                             "fallback": "bf16"}))
    assert (t.micro, t.accum) == (5, 1)  # prime batch: no split works
    assert t.amp is None                 # fp32 → no autocast
    assert not (out_dir / "stale.ckpt").exists()
    assert not (out_dir / "stale.pth").exists()
    assert t.checkpointable is False
    assert t._read_state() == {"path": None, "completed": 0}
    # fresh ⇒ scratch model, no init weights, and the prefetch ran
    # against the recording fake (no real download, no real build)
    assert t._model.init_kwargs == {}


def test_plan_world_one_flows_through(tmp_path, home, monkeypatch):
    """A single-process plan is accepted (the world≠1 refusal is covered
    above); micro×accum×world must still reproduce global_batch."""
    _require_rfdetr()
    _fake_rf_detr(monkeypatch, tmp_path)
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)
    t = build(tmp_path, run_dir=tmp_path / "run", datasets=("coco:v1",),
              plan=_Plan(micro_batch=1, grad_accum=4, world_size=1))
    assert (t.micro, t.accum) == (1, 4)


def test_finetune_init_weights_path(tmp_path, home, monkeypatch):
    """Fine-tune: the parent's committed weights are written to scratch,
    loaded with trust=True, and the state starts clean."""
    _require_rfdetr()
    _fake_rf_detr(monkeypatch, tmp_path)
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)
    t = build(tmp_path, run_dir=tmp_path / "run", datasets=("coco:v1",),
              init_weights=b"parent-weights")
    init_path = t.out_dir / "init_weights.ckpt"
    assert init_path.read_bytes() == b"parent-weights"
    assert t._model.init_kwargs["pretrain_weights"] == str(init_path)
    assert t._model.init_kwargs["trust_checkpoint"] is True
    assert t._read_state() == {"path": None, "completed": 0}


def test_payload_mutually_exclusive_incomplete_and_foreign(tmp_path, home,
                                                           monkeypatch):
    """Resume refusals fire in order: both sources at once, an incomplete
    payload, and a corrupt dataloader_state (fail-closed, 12 §11.4)."""
    _require_rfdetr()
    _fake_rf_detr(monkeypatch, tmp_path)
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)
    base = {"runtime": {}, "datasets": ("coco:v1",)}

    with pytest.raises(ValidationBlock, match="mutually exclusive"):
        build(tmp_path, run_dir=tmp_path / "r1", init_weights=b"w",
              payloads={c: b"x" for c in REQUIRED_COMPONENTS}, **base)
    with pytest.raises(ValidationBlock, match="payload incomplete"):
        build(tmp_path, run_dir=tmp_path / "r2",
              payloads={"model": b"x"}, **base)
    corrupt = {c: b"x" for c in REQUIRED_COMPONENTS}
    corrupt["dataloader_state"] = b"{not-json"
    with pytest.raises(ValidationBlock, match="not produced by trainer"):
        build(tmp_path, run_dir=tmp_path / "r3", payloads=corrupt, **base)


def test_resume_then_step_then_payload(tmp_path, home, monkeypatch):
    """The full CPU seam: fresh step → payload extraction → resume from
    that payload → second step (frame resume path, state rewrite, metric
    extraction through the stubbed Lightning trainer)."""
    _require_rfdetr()
    made = _fake_rf_detr(monkeypatch, tmp_path, fit="archive")
    progress: list[str] = []
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)
    cfg = {"epochs": 2, "global_batch": 2}

    t1 = build(tmp_path, run_dir=tmp_path / "run1", datasets=("coco:v1",),
               semantic=sem(**cfg), on_progress=progress.append)
    res1 = t1.step(TrainState(0, 0))
    assert res1.epoch == 1 and res1.done is False
    assert res1.loss == 0.5 and res1.metrics["val/loss"] == 0.25
    assert "epoch" not in res1.metrics  # prefix filter dropped it
    assert t1.checkpointable is True
    fit_cfg = t1._model.fit_configs[0]
    assert fit_cfg["epochs"] == 1 and fit_cfg["device"] == "cpu"
    assert len(made[0].callbacks) == 1  # the heartbeat pulse got installed
    made[0].callbacks[0].on_train_epoch_start()
    assert progress == ["train_epoch_start"]

    payload = t1.checkpoint_payload(TrainState(res1.global_step, res1.epoch))
    assert set(payload) == set(REQUIRED_COMPONENTS)
    assert payload["model"] == b"ckpt"

    t2 = build(tmp_path, run_dir=tmp_path / "run2", datasets=("coco:v1",),
               payloads=payload, start_step=res1.global_step,
               start_epoch=res1.epoch, semantic=sem(**cfg))
    assert t2.checkpointable is True
    assert t2._model.init_kwargs == {
        "pretrain_weights": None, "trust_checkpoint": False,
    }
    # corrupt resume state is an honest refusal, never a silent restart
    t2._state_path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValidationBlock, match="resume state unreadable"):
        t2._read_state()
    t2._write_state({"path": str((t2.out_dir / "last.ckpt").resolve()),
                     "completed": res1.epoch})
    res2 = t2.step(TrainState(res1.global_step, res1.epoch))
    assert res2.epoch == 2 and res2.done is True
    assert t2._model.fit_configs[0]["resume"].endswith("last.ckpt")

    # counters already done: cheap early-out before any framework call
    done = t2.step(TrainState(res2.global_step, t2.epochs))
    assert done.done is True and done.loss is None
    # tampered state pointing nowhere is refused (12 §11.2)
    t2._write_state({"path": str(tmp_path / "ghost.ckpt"), "completed": 1})
    with pytest.raises(ValidationBlock, match="resume checkpoint missing"):
        t2.step(TrainState(res2.global_step, 1))


def test_step_checkpoint_finding_variants(tmp_path, home, monkeypatch):
    """_newest_ckpt: prefer a fresh epoch archive, else a rewritten
    last.ckpt (mtime moved), else refuse to chain an unprovable state."""
    _require_rfdetr()
    _fake_rf_detr(monkeypatch, tmp_path, fit="last")
    tree = _coco_tree(tmp_path / "tree")
    _prepared(tmp_path, tree)
    t = build(tmp_path, run_dir=tmp_path / "run", datasets=("coco:v1",),
              semantic=sem(epochs=1))
    res = t.step(TrainState(0, 0))
    assert res.epoch == 1
    assert (t.out_dir / "last.ckpt").is_file()  # found via the mtime path

    # now an epoch that writes NOTHING at all → refuse to chain
    _fake_rf_detr(monkeypatch, tmp_path, fit="nothing")
    t2 = build(tmp_path, run_dir=tmp_path / "run2", datasets=("coco:v1",),
               semantic=sem(epochs=1))
    with pytest.raises(PreconditionFailed, match="wrote no resumable"):
        t2.step(TrainState(0, 0))


def test_step_state_refusals_without_framework(tmp_path):
    """Stubs: the done-early return, the tampered-resume refusal, the
    no-checkpoint payload refusal, and the _extract_metrics(None) path —
    all before any framework import."""
    stub = RFDETRTrainer.__new__(RFDETRTrainer)
    stub.out_dir = tmp_path
    stub.epochs = 1
    done = stub.step(TrainState(0, 1))
    assert done.done is True and done.loss is None

    stub.epochs = 99
    stub._write_state({"path": str(tmp_path / "ghost.ckpt"), "completed": 0})
    with pytest.raises(ValidationBlock, match="resume checkpoint missing"):
        stub.step(TrainState(0, 0))

    stub._state_path.unlink(missing_ok=True)
    with pytest.raises(ValidationBlock, match="no framework checkpoint"):
        stub.checkpoint_payload(TrainState(0, 0))

    assert stub._extract_metrics(None) == ({}, None)
    assert stub._extract_metrics(
        SimpleNamespace(callback_metrics=None)) == ({}, None)


def test_train_config_edges(tmp_path):
    """Pure config dict: absent resume/device keys take the None branches
    (the truthy branches run inside test_resume_then_step_then_payload)."""
    stub = RFDETRTrainer.__new__(RFDETRTrainer)
    stub.data = SimpleNamespace(dir=tmp_path / "data")
    stub.out_dir = tmp_path / "out"
    stub.micro, stub.accum = 4, 2
    stub.learning_rate, stub.optimizer_name = 1e-4, "adamw"
    stub.scheduler_name, stub.seed = "cosine", 42
    stub.amp, stub.num_workers = None, 0
    stub.eval_interval, stub.checkpoint_interval = 1, 10
    stub.early_stopping, stub.device = False, None
    cfg = stub._train_config(7, None)
    assert cfg["epochs"] == 7
    assert cfg["batch_size"] == 4 and cfg["grad_accum_steps"] == 2
    assert "resume" not in cfg and "device" not in cfg


def test_prefetch_refusal_edges(tmp_path, monkeypatch):
    """The published-base prefetch's honest refusals: missing downloader,
    variant without config/default (scratch is the point), a default with
    a directory part (no cache rewrite), and a failed fetch."""
    _require_rfdetr()
    import rfdetr.assets.model_weights as mw
    import rfdetr.config as rconfig

    stub = RFDETRTrainer.__new__(RFDETRTrainer)
    calls: list[str] = []
    monkeypatch.setattr(mw, "download_pretrain_weights", calls.append)
    real_weights_mod = sys.modules["rfdetr.assets.model_weights"]

    monkeypatch.setitem(sys.modules, "rfdetr.assets.model_weights", None)
    cls = type("RFDETRSmall", (), {})
    with pytest.raises(PreconditionFailed, match="downloader is missing"):
        stub._prefetch_published_base(cls)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.model_weights",
                        real_weights_mod)

    # no rfdetr.<Name>Config → return (a variant without a published base)
    never = type("MlforgeNeverConfiged", (), {})
    assert stub._prefetch_published_base(never) is None
    assert calls == []

    # config exists but its pretrain_weights default is falsy → return
    field = type("Field", (), {"default": None})()
    nodef = type("MlforgeNoDefaultConfig", (), {
        "model_fields": {"pretrain_weights": field}})()
    monkeypatch.setattr(rconfig, "MlforgeNoDefaultConfig", nodef,
                        raising=False)
    assert stub._prefetch_published_base(
        type("MlforgeNoDefault", (), {})) is None
    assert calls == []

    # default with a directory part → used verbatim, no cache rewrite
    field = type("Field", (), {"default": "sub/dir.pth"})()
    withdir = type("MlforgeWithDirConfig", (), {
        "model_fields": {"pretrain_weights": field}})()
    monkeypatch.setattr(rconfig, "MlforgeWithDirConfig", withdir, raising=False)
    stub._prefetch_published_base(type("MlforgeWithDir", (), {}))
    assert calls == ["sub/dir.pth"]

    # a failed fetch is a PreconditionFailed, never a silent random init
    def _boom(target):
        raise RuntimeError("network down")

    monkeypatch.setattr(mw, "download_pretrain_weights", _boom)
    with pytest.raises(PreconditionFailed, match="could not fetch"):
        stub._prefetch_published_base(type("MlforgeWithDir", (), {}))
