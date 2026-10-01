from __future__ import annotations

import pytest

from mlforge.run_spec import RunSpec
from mlforge.workflow import WorkflowAPI


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
