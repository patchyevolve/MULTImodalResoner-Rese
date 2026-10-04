"""Device placement contract (runtime.device) — no trainer hard-wires
CPU (12 §13.3 execution knobs, never semantic):

  * resolve_device: auto → cuda if THIS host has it, else cpu; explicit
    cpu/cuda honored; cuda requested on a GPU-less host is a
    PreconditionFailed at construction — never a silent downgrade.
  * checkpoint bytes are device-independent (cpu_tree) — a run folder
    saved on a CUDA host stays loadable anywhere.
  * preflight stays consistent: runtime.device naming cuda demands the
    GPU probe (runtime.gpu remains the explicit override).
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import pytest
import torch

# sibling test modules (pytest rootdir): shared fixtures + builders
from test_reid_trainer import build as reid_build
from test_reid_trainer import make_reid_project
from test_reid_trainer import run_steps as reid_run_steps
from test_rfdetr_trainer import sem as rfdetr_sem
from test_torch_trainer import PARAGRAPH, make_project
from test_torch_trainer import run_steps as text_run_steps
from test_torch_trainer import sem as text_sem

from mlforge.cli.main import _gpu_required_for
from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.runtime.trainer import TrainState
from mlforge.trainers import cpu_tree, resolve_device
from mlforge.trainers.text_lm import (
    MODEL_REGISTRY as TEXT_REGISTRY,
)
from mlforge.trainers.text_lm import (
    TextArch,
    TorchTextTrainer,
)


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """Machine-local dataset paths live outside the workspace (13 §10)."""
    h = tmp_path / "mlforge-home"
    monkeypatch.setenv("MLFORGE_HOME", str(h))
    return h


# -- resolve_device: the pure contract ---------------------------------

def test_auto_resolves_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device({}) == torch.device("cpu")
    assert resolve_device(None) == torch.device("cpu")
    assert resolve_device({"device": "auto"}) == torch.device("cpu")
    assert resolve_device({"device": ""}) == torch.device("cpu")


def test_auto_resolves_cuda_when_host_has_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device({}) == torch.device("cuda")
    assert resolve_device({"device": "auto"}) == torch.device("cuda")
    assert resolve_device({"device": "cuda:1"}) == torch.device("cuda:1")


def test_explicit_cpu_wins_even_when_cuda_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device({"device": "cpu"}) == torch.device("cpu")


@pytest.mark.parametrize("raw", ["cuda", "cuda:0"])
def test_cuda_requested_without_host_refuses(raw, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PreconditionFailed, match="requires CUDA"):
        resolve_device({"device": raw})


def test_cuda_index_beyond_host_devices_refuses(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(PreconditionFailed, match="1 CUDA device"):
        resolve_device({"device": "cuda:1"})


@pytest.mark.parametrize("raw", ["tpu", "gpu:0", "cuda:abc", "banana"])
def test_unparseable_device_is_blocked(raw):
    with pytest.raises(ValidationBlock, match="is not a device string"):
        resolve_device({"device": raw})


def test_non_cuda_device_type_is_blocked():
    # torch.device parses "mps" but placement outside cpu/cuda is refused
    # with the accepted list — no silent surprise placement.
    with pytest.raises(ValidationBlock, match="unsupported device type"):
        resolve_device({"device": "mps"})


# -- cpu_tree: portable checkpoint bytes -------------------------------

def test_cpu_tree_moves_nested_tensors_and_keeps_values():
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tree = {
        "a": {"b": [t * 2, 7]},
        "step": torch.tensor(3.0),
        "n": 3,
        "pair": (t * 3,),
    }
    out = cpu_tree(tree)
    assert out["a"]["b"][0].device.type == "cpu"
    assert out["step"].device.type == "cpu"
    assert type(out["a"]["b"]) is list
    assert type(out["pair"]) is tuple
    assert torch.equal(out["a"]["b"][0], t * 2)
    assert torch.equal(out["step"], torch.tensor(3.0))
    assert out["a"]["b"][1] == 7 and out["n"] == 3
    assert torch.equal(out["pair"][0], t * 3)


# -- preflight expectation stays consistent ----------------------------

def test_gpu_required_follows_device_cuda(tmp_path):
    assert _gpu_required_for(tmp_path, "run-x", {"device": "cuda"}) is True
    assert _gpu_required_for(tmp_path, "run-x", {"device": "cuda:0"}) is True


def test_gpu_bool_override_still_wins(tmp_path):
    # runtime.gpu is the documented explicit override (README, 12 §14);
    # a contradictory device: cuda then refuses later, at trainer build.
    assert _gpu_required_for(
        tmp_path, "run-x", {"gpu": False, "device": "cuda"}
    ) is False
    assert _gpu_required_for(
        tmp_path, "run-x", {"gpu": True, "device": "cpu"}
    ) is True


def test_gpu_required_reads_stored_runtime(tmp_path):
    p = tmp_path / "runs" / "run-x" / "state" / "runtime.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"device": "cuda"}', encoding="utf-8")
    assert _gpu_required_for(tmp_path, "run-x", None) is True


# -- trainer constructors: refusal before any side effect --------------

def test_text_trainer_refuses_cuda_without_host(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PreconditionFailed, match="requires CUDA"):
        TorchTextTrainer(
            model_name="reasoner_s",
            semantic={},
            runtime={"device": "cuda"},
            train_datasets=(),
            root=tmp_path,
        )


def test_text_trainer_blocks_unknown_device(tmp_path):
    with pytest.raises(ValidationBlock, match="is not a device string"):
        TorchTextTrainer(
            model_name="reasoner_s",
            semantic={},
            runtime={"device": "not-a-device"},
            train_datasets=(),
            root=tmp_path,
        )


def test_reid_trainer_refuses_cuda_without_host(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from mlforge.trainers.reid import ReidTrainer

    with pytest.raises(PreconditionFailed, match="requires CUDA"):
        ReidTrainer(
            model_name="osnet_x1_0",
            semantic={},
            runtime={"device": "cuda"},
            train_datasets=(),
            root=tmp_path,
        )


def test_rfdetr_trainer_refuses_cuda_without_host(tmp_path, monkeypatch):
    from mlforge.trainers.rfdetr import RFDETRTrainer, dependency_error

    if dependency_error() is not None:
        pytest.skip(dependency_error())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PreconditionFailed, match="requires CUDA"):
        RFDETRTrainer(
            model_name="rf_detr_s",
            semantic=rfdetr_sem(),
            runtime={"device": "cuda"},
            train_datasets=(),
            root=tmp_path,
            run_dir=tmp_path / "run",
        )


# -- real CUDA execution (runs on a CUDA host; skips here) -------------

def _all_cpu(obj) -> bool:
    if isinstance(obj, dict):
        return all(_all_cpu(v) for v in obj.values())
    if type(obj) in (list, tuple):
        return all(_all_cpu(v) for v in obj)
    if hasattr(obj, "device"):
        return obj.device.type == "cpu"
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA host")
def test_text_trains_on_cuda_and_checkpoints_portably(tmp_path, monkeypatch):
    monkeypatch.setitem(
        TEXT_REGISTRY, "unit_tiny_lm",
        TextArch(n_layer=1, n_head=2, d_model=32, block_size=64),
    )
    make_project(tmp_path, [PARAGRAPH * 30])
    trainer = TorchTextTrainer(
        model_name="unit_tiny_lm",
        semantic=text_sem(epochs=1),
        runtime={"device": "cuda"},
        train_datasets=("books:v1",),
        root=tmp_path,
    )
    assert trainer.device.type == "cuda"
    assert next(trainer.model.parameters()).is_cuda
    state, losses = text_run_steps(trainer, TrainState(0, 0), 4)
    assert losses and all(math.isfinite(v) for v in losses)
    payload = trainer.checkpoint_payload(
        TrainState(state.global_step, state.epoch)
    )
    model_sd = torch.load(
        io.BytesIO(payload["model"]), weights_only=True, map_location="cpu"
    )
    opt_sd = torch.load(
        io.BytesIO(payload["optimizer"]), weights_only=True,
        map_location="cpu",
    )
    assert _all_cpu(model_sd), "checkpoint bytes must be device-independent"
    assert _all_cpu(opt_sd)
    resumed = TorchTextTrainer(
        model_name="unit_tiny_lm",
        semantic=text_sem(epochs=1),
        runtime={"device": "cuda"},
        train_datasets=("books:v1",),
        root=tmp_path,
        payloads=payload,
    )
    _, losses2 = text_run_steps(
        resumed, TrainState(state.global_step, state.epoch), 2
    )
    assert losses2 and all(math.isfinite(v) for v in losses2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA host")
def test_reid_trains_on_cuda_and_checkpoints_portably(tmp_path, home):
    make_reid_project(tmp_path)
    trainer = reid_build(tmp_path, runtime={"device": "cuda"})
    assert trainer.device.type == "cuda"
    assert next(trainer.model.parameters()).is_cuda
    state, losses = reid_run_steps(trainer, TrainState(0, 0), 2)
    assert losses and all(math.isfinite(v) for v in losses)
    payload = trainer.checkpoint_payload(
        TrainState(state.global_step, state.epoch)
    )
    model_sd = torch.load(
        io.BytesIO(payload["model"]), weights_only=True, map_location="cpu"
    )
    opt_sd = torch.load(
        io.BytesIO(payload["optimizer"]), weights_only=True,
        map_location="cpu",
    )
    assert _all_cpu(model_sd), "checkpoint bytes must be device-independent"
    assert _all_cpu(opt_sd)
    resumed = reid_build(
        tmp_path, runtime={"device": "cuda"}, payloads=payload
    )
    _, losses2 = reid_run_steps(
        resumed, TrainState(state.global_step, state.epoch), 1
    )
    assert losses2 and all(math.isfinite(v) for v in losses2)
