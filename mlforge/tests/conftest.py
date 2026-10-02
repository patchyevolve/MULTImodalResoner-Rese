from __future__ import annotations

import pytest

from mlforge.run_spec import RunSpec
from mlforge.workflow import WorkflowAPI


@pytest.fixture(autouse=True)
def _harness_opt_in(monkeypatch):
    """System tests may run the labeled scaffold harness (fake loss).
    Production defaults to FAIL-CLOSED: no real trainer integrated ⇒
    train/launch/worker refuse (see mlforge.runtime.trainer). Tests that
    assert that refusal must `monkeypatch.delenv(MLFORGE_HARNESS)`."""
    monkeypatch.setenv("MLFORGE_HARNESS", "1")


@pytest.fixture
def wf(tmp_path) -> WorkflowAPI:
    return WorkflowAPI(tmp_path / "workspace")


@pytest.fixture
def spec() -> RunSpec:
    """A spec satisfying 12 §17 required semantic fields."""
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1", "custom_clips:v1"),
        val_dataset="coco_2017:val",
        semantic={
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "scheduler": "cosine",
            "loss": "composite_detection",
            "seed": 42,
            "global_batch": 32,
            "epochs": 50,
            "precision_policy": {"preferred": "bf16", "fallback": "fp32"},
        },
    )
