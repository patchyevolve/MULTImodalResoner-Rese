"""Real TEXT engine — the evaluate / infer / export seams with NO
harness (12 §12.4), called DIRECTLY against tiny in-test weights:

  * family surface — dependency_error/input_types/metric_names,
    supports_format honest refusals (13 §4.1)
  * contract — the real text input/output schema, measured operators,
    opset 17 runtime (12 §15.3)
  * execute — one-shot UTF-8 prompt → deterministic prompt loss /
    perplexity + next-byte top-k; non-UTF8 / empty / wrong-contract
    inputs BLOCK before execution (13 §6.8)
  * compute_metrics — real perplexity over a tiny prepared
    `text_corpus` pool + the transform / split guards (13 §6.7, 12
    §15.4)
  * export_bytes — genuine ONNX bytes (ir_version 8), PASSING
    numerical round-trip against the torch forward (13 §6.9)

Weights are built in-test from the registry architecture (random init
is honest here — no downloads, shape + round-trip are what's under
test) and torch.save'd as the trainer's `model` state_dict. The
harness gate is lifted for every test (conftest opts IN; these tests
opt OUT — a seam regressing to the scaffold breaks them).
"""

from __future__ import annotations

import io
import math

import pytest

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.ops.engines import harness_active
from mlforge.ops.engines import text_lm as engine

#: Handcrafted registry entry — what contract()/execute() read off it.
ENTRY: dict = {
    "name": "reasoner_s",
    "version": "v1",
    "model_id": "mdl_text_unit",
    "run_id": None,
    "artifact_hash": "sha256:" + "0" * 64,
}

PROMPT = (
    "the quick system trains the model on honest gradients every step "
    "and the checkpoint keeps the optimizer state for the next run "
)


@pytest.fixture(autouse=True)
def real_path(monkeypatch):
    """These tests must never see the scaffold harness (12 §12.4)."""
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    assert harness_active() is False


@pytest.fixture(scope="module")
def weights() -> bytes:
    """REAL weights bytes: the registry architecture, seeded random
    init, saved exactly like the trainer's `model` component."""
    import torch

    from mlforge.trainers.text_lm import MODEL_REGISTRY, TinyGPT

    torch.manual_seed(7)
    model = TinyGPT(MODEL_REGISTRY["reasoner_s"])
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def contract_doc() -> dict:
    return engine.contract(ENTRY, "onnx")


def _checked(path) -> dict:
    """check_input's SAFE envelope for a text file (ops/infer.py)."""
    return {
        "status": "SAFE",
        "input": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "extension": path.suffix,
            "identity": "sha256:" + "f" * 64,
        },
    }


def _records(*, split: str = "valid", n: int = 3) -> list[dict]:
    """Tiny prepared `text_corpus` records — > one block_size window
    once joined (block_size = 256)."""
    return [
        {
            "source": "books_eval:val",
            "split": split,
            "relative_path": f"t{i}.txt#0",
            "chars": len(PROMPT),
            "text": PROMPT * 3,
        }
        for i in range(n)
    ]


# -- family surface ----------------------------------------------------------


def test_dependency_error_is_none():
    assert engine.FAMILY == "text"
    assert engine.dependency_error() is None


def test_input_types_and_metric_names_match_trainer_truth():
    from mlforge.trainers.text_lm import MODEL_REGISTRY

    assert engine.input_types() == frozenset({"text"})
    # trainer truth: step() records `ppl` from byte cross-entropy and
    # 06_benchmarking_plan.md defines no text metric set to override it
    assert engine.metric_names() == ("perplexity",)
    assert engine.MODEL_NAMES == frozenset(MODEL_REGISTRY)


def test_supports_format_onnx_only():
    assert engine.supports_format("onnx") is None
    for fmt in ("tflite", "openvino"):
        msg = engine.supports_format(fmt)
        assert msg is not None
        assert fmt in msg and "onnx" in msg.lower()
        assert "13 §4.1" in msg or "§4.1" in msg


# -- contract (12 §15.3) ------------------------------------------------------


def test_contract_declares_real_text_contract(contract_doc):
    import onnx.defs

    assert contract_doc["schema_version"] == 1
    assert contract_doc["name"] == "reasoner_s"
    assert contract_doc["harness"] == "text"
    ic = contract_doc["inference_contract"]
    assert ic["input_schema"] == [{"name": "text", "type": "text"}]
    fields = {f["name"] for f in ic["output_schema"][0]["fields"]}
    assert {"prompt_bytes", "loss", "perplexity", "next_byte",
            "next_byte_topk"} <= fields
    assert ic["runtime"] == {"framework": "onnx", "opset": 17}
    assert set(ic["dynamic_axes"]) == {"batch", "sequence"}
    assert ic["numerical_tolerance"]["atol"] == engine.ATOL
    assert ic["dtype"] == "fp32"
    # honest operator list: every declared op exists in the installed
    # ONNX schemas (validate_export's authority, 13 §6.9/§7)
    assert {"Gemm", "Softmax", "LayerNormalization"} <= set(ic["operators"])
    schemas = {s.name.lower().replace("_", "")
               for s in onnx.defs.get_all_schemas()}
    for op in ic["operators"]:
        assert op.lower().replace("_", "") in schemas, op


# -- execute (13 §6.8) --------------------------------------------------------


def test_execute_scores_prompt_deterministically(tmp_path, weights,
                                                 contract_doc):
    p = tmp_path / "prompt.txt"
    p.write_text(PROMPT, encoding="utf-8")
    out = engine.execute(entry=ENTRY, contract_doc=contract_doc,
                         checked=_checked(p), weights=weights)
    assert set(out) == {"prompt_bytes", "window_bytes", "truncated",
                        "loss", "perplexity", "next_byte",
                        "next_byte_topk"}
    assert out["prompt_bytes"] == len(PROMPT.encode("utf-8"))
    assert out["window_bytes"] == out["prompt_bytes"]
    assert out["truncated"] is False
    loss = float(out["loss"])
    ppl = float(out["perplexity"])
    assert math.isfinite(loss) and loss > 0
    assert math.isfinite(ppl) and ppl > 1.0
    # trainer formula: ppl == exp(min(loss, 20)) (rounded)
    assert abs(ppl - math.exp(min(loss, 20.0))) < 1e-3
    assert 0 <= int(out["next_byte"]) < 256
    assert len(out["next_byte_topk"]) == 5
    assert all(0 <= t["byte"] < 256 for t in out["next_byte_topk"])
    # deterministic: same input + same weights ⇒ same result
    again = engine.execute(entry=ENTRY, contract_doc=contract_doc,
                           checked=_checked(p), weights=weights)
    assert again == out


def test_execute_truncates_prompt_to_context(tmp_path, weights,
                                             contract_doc):
    p = tmp_path / "long.txt"
    p.write_text(PROMPT * 4, encoding="utf-8")  # > block_size (256)
    out = engine.execute(entry=ENTRY, contract_doc=contract_doc,
                         checked=_checked(p), weights=weights)
    assert out["truncated"] is True
    assert out["window_bytes"] == 256
    assert out["prompt_bytes"] > 256


def test_execute_non_utf8_input_blocks(tmp_path, weights, contract_doc):
    p = tmp_path / "binary.txt"
    p.write_bytes(b"\xff\xfe\x00\x01\x89PNG not utf8 \xff\xfe")
    with pytest.raises(ValidationBlock) as ei:
        engine.execute(entry=ENTRY, contract_doc=contract_doc,
                       checked=_checked(p), weights=weights)
    out = ei.value.render()
    assert "UTF-8" in out
    assert "6.8" in out  # BLOCK before execution


def test_execute_empty_input_blocks(tmp_path, weights, contract_doc):
    p = tmp_path / "empty.txt"
    p.write_bytes(b"")
    with pytest.raises(ValidationBlock) as ei:
        engine.execute(entry=ENTRY, contract_doc=contract_doc,
                       checked=_checked(p), weights=weights)
    assert "empty" in ei.value.render()


def test_execute_wrong_contract_blocks(tmp_path, weights):
    p = tmp_path / "prompt.txt"
    p.write_text(PROMPT, encoding="utf-8")
    wrong = engine.contract(ENTRY, "onnx")
    wrong["inference_contract"]["input_schema"] = [
        {"name": "structured", "type": "structured"}]
    with pytest.raises(ValidationBlock) as ei:
        engine.execute(entry=ENTRY, contract_doc=wrong,
                       checked=_checked(p), weights=weights)
    assert "no `text` input" in ei.value.render()


def test_execute_rejects_foreign_weights(tmp_path, contract_doc):
    p = tmp_path / "prompt.txt"
    p.write_text(PROMPT, encoding="utf-8")
    import torch

    buf = io.BytesIO()
    torch.save({"not_a_state_dict": torch.zeros(3)}, buf)
    with pytest.raises(ValidationBlock) as ei:
        engine.execute(entry=ENTRY, contract_doc=contract_doc,
                       checked=_checked(p), weights=buf.getvalue())
    out = ei.value.render()
    assert "state_dict" in out
    assert "11.4" in out  # the trainer's model component is cited


# -- compute_metrics (13 §6.7) ------------------------------------------------


def _dataset(**over) -> dict:
    base = {"ref": "books_eval:v1", "transform": "text_corpus",
            "records": _records()}
    base.update(over)
    return base


def test_compute_metrics_reports_real_perplexity(weights):
    from mlforge.ops.evaluation import build_protocol

    protocol = build_protocol(metric_names=engine.metric_names())
    metrics = engine.compute_metrics(
        entry=ENTRY, weights=weights, dataset=_dataset(),
        protocol=protocol, semantic=None)
    assert set(metrics) == {"perplexity"}
    ppl = float(metrics["perplexity"])
    assert math.isfinite(ppl) and ppl > 1.0
    # random-init byte LM ≈ ln(256) nats ⇒ ppl near 256 — an honest
    # number must sit in a sane band, never nan/inf/0
    assert 1.0 < ppl < 100000.0


def test_compute_metrics_preview_is_stable(weights):
    from mlforge.ops.evaluation import build_protocol

    protocol = build_protocol(metric_names=engine.metric_names())
    a = engine.compute_metrics(entry=ENTRY, weights=weights,
                               dataset=_dataset(), protocol=protocol,
                               semantic=None)
    b = engine.compute_metrics(entry=ENTRY, weights=weights,
                               dataset=_dataset(), protocol=protocol,
                               semantic=None)
    assert a == b


def test_compute_metrics_wrong_transform_blocks(weights):
    from mlforge.ops.evaluation import build_protocol

    protocol = build_protocol(metric_names=engine.metric_names())
    with pytest.raises(ValidationBlock) as ei:
        engine.compute_metrics(
            entry=ENTRY, weights=weights,
            dataset=_dataset(transform="tabular"),
            protocol=protocol, semantic=None)
    out = ei.value.render()
    assert "transform" in out and "tabular" in out
    assert "text_corpus" in out
    assert "mlforge prepare" in out


def test_compute_metrics_refuses_train_pool(weights):
    from mlforge.ops.evaluation import build_protocol

    protocol = build_protocol(metric_names=engine.metric_names())
    with pytest.raises(ValidationBlock) as ei:
        engine.compute_metrics(
            entry=ENTRY, weights=weights,
            dataset=_dataset(records=_records(split="train")),
            protocol=protocol, semantic=None)
    out = ei.value.render()
    assert "no eval-split rows" in out
    assert "train" in out
    assert "never eval rows" in out


def test_compute_metrics_empty_records_blocks(weights):
    from mlforge.ops.evaluation import build_protocol

    protocol = build_protocol(metric_names=engine.metric_names())
    with pytest.raises(ValidationBlock) as ei:
        engine.compute_metrics(
            entry=ENTRY, weights=weights, dataset=_dataset(records=[]),
            protocol=protocol, semantic=None)
    assert "no prepared records" in ei.value.render()


def test_compute_metrics_empty_text_record_blocks(weights):
    from mlforge.ops.evaluation import build_protocol

    protocol = build_protocol(metric_names=engine.metric_names())
    records = _records()
    records[1] = {**records[1], "text": "   "}
    with pytest.raises(ValidationBlock) as ei:
        engine.compute_metrics(
            entry=ENTRY, weights=weights, dataset=_dataset(records=records),
            protocol=protocol, semantic=None)
    assert "record 1" in ei.value.render()


# -- export (13 §6.9) ---------------------------------------------------------


def test_export_bytes_real_onnx_and_roundtrip(weights, contract_doc):
    import onnx

    proto, numerical = engine.export_bytes(
        entry=ENTRY, weights=weights, fmt="onnx")
    assert isinstance(proto, (bytes, bytearray)) and len(proto) > 10_000
    assert proto[:2] == b"\x08\x08"  # ir_version = 8 (portable)
    model_onnx = onnx.load_from_string(bytes(proto))
    onnx.checker.check_model(model_onnx)
    # real graph: dynamic batch/sequence, declared ops all present
    dims = [d.dim_param or d.dim_value
            for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]
    assert dims == ["batch", "sequence"]
    graph_ops = {n.op_type for n in model_onnx.graph.node}
    assert set(engine.OPERATORS) <= graph_ops, (
        sorted(set(engine.OPERATORS) - graph_ops))
    # numerical round-trip, shaped exactly like the reference engines
    assert set(numerical) == {"max_error", "tolerance", "result",
                              "harness", "n_probe"}
    assert numerical["result"] == "PASS"
    assert numerical["tolerance"] == engine.ATOL
    assert float(numerical["max_error"]) < float(numerical["tolerance"])
    assert numerical["n_probe"] >= 3
    assert numerical["harness"] == "torch-onnx-roundtrip"
    # the contract the workflow would record matches the binary
    ic = contract_doc["inference_contract"]
    assert model_onnx.opset_import[0].version == 17
    assert ic["numerical_tolerance"]["atol"] == engine.ATOL


def test_export_refuses_non_onnx_format(weights):
    with pytest.raises(PreconditionFailed) as ei:
        engine.export_bytes(entry=ENTRY, weights=weights, fmt="tflite")
    out = ei.value.render()
    assert "tflite" in out
    assert "onnx" in out.lower()  # refusal names the portable target
