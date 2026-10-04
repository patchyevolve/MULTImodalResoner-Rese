"""Text family engine — real byte-level transformer inference (one-shot
prompt scoring), perplexity over prepared `text_corpus` records, and a
genuine torch → ONNX export (trainers/text_lm.py byte-level LM —
03_training_pipeline.md Models 1–5 define no text row, so this family's
architecture is keyed by the trainer's own MODEL_REGISTRY).

Normative: 13_product_specification §6.7 (EVALUATE runs the harness over
the dataset), §6.8 (INFER: validate → contract check → execute), §6.9
(export: operator validation, real binary, numerical round-trip), 12
§12.4 (pluggable real harness), §15.3 (weights alone ≠ model — the
contract declares how the model is called).

Metric truth: 06_benchmarking_plan.md defines NO text-family metric
set (no perplexity row exists there), so the trainer's own quality
signal is the authority — TorchTextTrainer.step records
`ppl = round(math.exp(min(loss, 20.0)), 3)` from byte cross-entropy
(trainers/text_lm.py) → this engine reports `perplexity` with that same
formula (single source of truth).

Family contract (torch/onnx imported only inside functions — the module
is import-safe everywhere):
  * execute    — one-shot UTF-8 text file → the trainer's OWN byte
                 tokenization (UTF-8 bytes, VOCAB_SIZE=256), one
                 no_grad forward: teacher-forced prompt loss +
                 perplexity and the top-k next-byte log-probs;
  * metrics    — perplexity over `text_corpus` records using the SAME
                 corpus join (`"\\n\\n".join`), block windowing and ppl
                 formula the trainer trains/records with, so train-time
                 and eval-time definitions cannot drift;
  * export     — TinyGPT.forward → ONNX opset 17 with dynamic batch and
                 sequence axes, onnx.checker + onnxruntime round-trip
                 against the torch forward on deterministic probes
                 (13 §6.9: a FAIL blocks the export).

Known exporter constraint (torch 2.14): the default (dynamo) ONNX
exporter needs the `onnxscript` package; the legacy TorchScript path
bakes the example sequence length inside nn.MultiheadAttention (measured
— the exported Reshape shape carries a constant `seq`), so a dynamic-seq
graph from it is wrong at runtime. `supports_format` names the missing
package honestly instead of emitting such a graph.
"""

from __future__ import annotations

import math
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash

FAMILY = "text"

#: Models this engine runs — the text trainer's MODEL_REGISTRY keys
#: (trainers/text_lm.py: reasoner_s byte-level transformer).
MODEL_NAMES = frozenset({"reasoner_s"})

#: Numerical round-trip tolerance for the ONNX export (13 §6.9) —
#: MEASURED on this family: torch-vs-onnxruntime max error on the probe
#: set is ~1.0e-6, so 1e-5 is an honest threshold with real margin.
ATOL = 1e-5

#: Required operators of the exported graph (measured from the actual
#: torch → ONNX TinyGPT graph; validate_export checks them against the
#: installed ONNX schemas, 13 §6.9/§7).
OPERATORS: tuple[str, ...] = (
    "Add", "Concat", "Div", "Erf", "Expand", "Gather", "Gemm",
    "LayerNormalization", "MatMul", "Mul", "Range", "Reshape", "Shape",
    "Slice", "Softmax", "Squeeze", "Transpose", "Trilu", "Unsqueeze",
    "Where",
)

#: Deterministic round-trip probes: (batch, sequence) pairs covering both
#: dynamic axes, incl. the degenerate 1-token prompt and a full
#: block_size context (never longer — the positional table has
#: block_size rows). Same seeds ⇒ same bytes ⇒ same validation.
PROBES: tuple[tuple[int, int], ...] = ((1, 1), (2, 8), (3, 17),
                                       (1, 64), (2, 256))
PROBE_SEED = 20240517

#: Next-byte candidates returned by execute.
TOP_K = 5

#: Eval micro-batch (windowed corpus — every window is block_size long,
#: so windows stack into one forward; bounded memory on CPU).
_MICRO_WINDOWS = 16


def dependency_error() -> str | None:
    """None = runnable here (torch present; onnx/onnxruntime/onnxscript
    are checked at export time so inference/metrics work on a
    torch-only machine — same split as the ranker engine)."""
    from mlforge.trainers import text_lm

    return text_lm.dependency_error()


def input_types() -> frozenset[str]:
    return frozenset({"text"})


def metric_names(weights: bytes | None = None) -> tuple[str, ...]:
    """The trainer's own quality metric: `ppl` from byte cross-entropy
    (trainers/text_lm.py step(); 06_benchmarking_plan.md defines no
    text metric set to override it). `weights` is unused here (engine
    contract — the calibrator derives per-model keys from it)."""
    return ("perplexity",)


def _arch(entry: Mapping[str, Any]):
    """MODEL_REGISTRY architecture for this model (fail-closed — an
    unknown family name is never silently run as something else)."""
    from mlforge.trainers.text_lm import MODEL_REGISTRY

    name = str(entry.get("name"))
    arch = MODEL_REGISTRY.get(name)
    if arch is None:
        raise PreconditionFailed(
            f"model {name!r} has no registered text architecture",
            hint=f"text models runnable here: "
                 f"{', '.join(sorted(MODEL_REGISTRY))}",
        )
    return arch


def _model(weights: bytes, entry: Mapping[str, Any]):
    """Checkpoint `model` blob → TinyGPT in eval mode (fail-closed — a
    blob that is not this family's state_dict is refused, never
    re-initialized: no weights ⇒ no run, 12 §15.3)."""
    import io

    import torch

    from mlforge.trainers.text_lm import TinyGPT, torch_available

    if not torch_available():
        raise PreconditionFailed(
            "torch is not installed — the text engine needs it to run "
            "the model",
            hint="pip install torch --index-url "
                 "https://download.pytorch.org/whl/cpu (CPU wheels)",
        )
    arch = _arch(entry)
    name = f"{entry.get('name')}:{entry.get('version')}"
    try:
        state = torch.load(io.BytesIO(weights), weights_only=True,
                           map_location="cpu")
        if not isinstance(state, dict) or not state:
            raise ValueError("payload is not a non-empty state dict")
        model = TinyGPT(arch)
        model.load_state_dict(state)
    except Exception as exc:
        raise ValidationBlock(
            f"model {name}: checkpoint model blob is not a "
            f"{entry.get('name')} state_dict ({exc})",
            hint="the text trainer's `model` component is "
                 "TinyGPT.state_dict() (12 §11.4) — this checkpoint was "
                 "produced by a different trainer",
        ) from exc
    model.eval()
    return model


def _read_prompt(path: Path) -> bytes:
    """One-shot input file → prompt bytes (fail-closed: the contract's
    `text` type means UTF-8, 13 §6.8 — schema invalid ⇒ BLOCK)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationBlock(
            f"text input is not valid UTF-8: {exc}",
            hint="input type `text` requires UTF-8 decodable content "
                 "(13 §6.8 — BLOCK before execution)",
        ) from exc
    data = text.encode("utf-8")
    if not data:
        raise ValidationBlock(
            "text input is empty — the prompt must carry at least one "
            "byte",
            hint="13 §6.8 — an empty input never reaches execution",
        )
    return data


def _check_contract(contract_doc: Mapping[str, Any]) -> None:
    """The declared contract must actually be a text contract before the
    engine runs anything (12 §15.3: the contract is the authority)."""
    schema = ((contract_doc or {}).get("inference_contract") or {}).get(
        "input_schema") or []
    for item in schema:
        if isinstance(item, dict) and item.get("type") == "text":
            return
    raise ValidationBlock(
        "inference contract declares no `text` input — cannot run the "
        "text engine on this model",
        hint="re-export so the contract records its input type "
             "(12 §15.3); the contract at model_spec.json is the "
             "authority (13 §6.8)",
    )


def contract(
    entry: Mapping[str, Any],
    fmt: str,
    *,
    semantic: Mapping[str, Any] | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """First-export `model_spec.json` for the text family (12 §15.3) —
    real input/output schema, the measured operator list, real runtime.

    `semantic` is accepted for the engine-module signature; this
    family's architecture is keyed by the model name in
    MODEL_REGISTRY, not by run semantics."""
    from mlforge.ops.exporting import FORMATS

    ops = list(OPERATORS)
    return {
        "schema_version": 1,
        "name": entry.get("name"),
        "version": entry.get("version"),
        "harness": FAMILY,
        "inference_contract": {
            "input_schema": [
                {"name": "text", "type": "text"},
            ],
            "output_schema": [
                {
                    "name": "score",
                    "type": "structured",
                    "schema": "json_schema://mlforge.text.score.v1",
                    "fields": [
                        {"name": "prompt_bytes", "dtype": "int"},
                        {"name": "window_bytes", "dtype": "int"},
                        {"name": "truncated", "dtype": "bool"},
                        {"name": "loss", "dtype": "float|null"},
                        {"name": "perplexity", "dtype": "float|null"},
                        {"name": "next_byte", "dtype": "int"},
                        {"name": "next_byte_topk", "dtype": "object[]"},
                    ],
                }
            ],
            "operators": ops,
            "operator_set": content_hash({"ops": sorted(ops)}),
            "runtime": {
                "framework": fmt,
                "opset": FORMATS[fmt]["opset"] if fmt in FORMATS else 17,
            },
            "dtype": precision,
            "dynamic_axes": {"batch": [0], "sequence": [1]},
            "minimal_memory": {"vram_mb": 128},
            "numerical_tolerance": {"atol": ATOL},
        },
    }


def supports_format(fmt: str) -> str | None:
    """None = this format can be produced HERE (13 §6.9 gate)."""
    if fmt == "onnx":
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
            import onnxscript  # noqa: F401  # dynamo exporter (torch 2.14)
        except Exception as exc:
            return (f"onnx export needs the onnx + onnxruntime + "
                    f"onnxscript packages ({type(exc).__name__}: {exc}) "
                    "— pip install onnx onnxruntime onnxscript")
        return None
    return (f"the {fmt} converter is not integrated for the text family "
            "(13 §4.1 names ONNX as the portable target for exported "
            "models) — use --format onnx")


def execute(
    *,
    entry: Mapping[str, Any],
    contract_doc: Mapping[str, Any],
    checked: Mapping[str, Any],
    weights: bytes,
) -> dict[str, Any]:
    """Real one-shot execution — the byte-level LM runs on the checked
    text input (13 §6.8).

    Output keys (documented in the contract's output_schema):
      prompt_bytes    total UTF-8 bytes of the prompt
      window_bytes    bytes actually scored (<= block_size)
      truncated       True when older bytes fell outside the context
      loss            teacher-forced mean next-byte NLL over the window
                      (None when the prompt is a single byte)
      perplexity      exp(min(loss, 20.0)) — the trainer's own ppl
                      formula (trainers/text_lm.py step())
      next_byte       argmax next-byte prediction at the last position
      next_byte_topk  [{"byte": int, "logprob": float}, ...] top-5
    """
    import torch
    import torch.nn.functional as F

    from mlforge.trainers.text_lm import VOCAB_SIZE

    _check_contract(contract_doc)
    model = _model(weights, entry)
    block = int(model.arch.block_size)
    data = _read_prompt(Path(str(checked["input"]["path"])))
    truncated = len(data) > block
    window = data[-block:] if truncated else data  # tail = the context
    idx = torch.tensor([list(window)], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(idx, None)  # (1, T, VOCAB_SIZE)
    loss = None
    ppl = None
    if len(window) >= 2:
        # same F.cross_entropy mean the trainer optimizes, scored over
        # the prompt's own next-byte pairs (no byte exists past the
        # window, so the final position carries no target)
        targets = idx[0, 1:]
        loss = float(F.cross_entropy(logits[0, :-1], targets))
        # trainer formula: math.exp(min(loss, 20.0))
        ppl = math.exp(min(loss, 20.0))
    logprobs = F.log_softmax(logits[0, -1], dim=-1)
    k = min(TOP_K, VOCAB_SIZE)
    top = torch.topk(logprobs, k=k)
    return {
        "prompt_bytes": len(data),
        "window_bytes": len(window),
        "truncated": bool(truncated),
        "loss": None if loss is None else round(loss, 6),
        "perplexity": None if ppl is None else round(ppl, 6),
        "next_byte": int(torch.argmax(logprobs).item()),
        "next_byte_topk": [
            {"byte": int(i), "logprob": round(float(v), 6)}
            for v, i in zip(top.values.tolist(), top.indices.tolist())
        ],
    }


def compute_metrics(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Real perplexity over prepared `text_corpus` records (13 §6.7:
    the harness runs over the dataset).

    Single source of truth with the trainer: the eval corpus is built
    exactly like `load_text_corpus` joins texts (`"\\n\\n".join`), the
    windows come from the trainer's stride/block windowing, and the
    reported number is the trainer's own `ppl = exp(min(loss, 20.0))`.

    Row rules mirror the ranker's split discipline (12 §15.4: `split` is
    never "train"): valid/val/validation rows and rows without a split
    column count; train/holdout rows are skipped and a pool with no
    eval-split rows BLOCKs listing what it saw.
    """
    import torch

    from mlforge.trainers.gbdt import _EXCLUDED_SPLITS, _VAL_SPLITS

    transform = dataset.get("transform")
    if transform and str(transform) != "text_corpus":
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} was prepared with transform "
            f"{transform!r} — the text engine measures perplexity over "
            "`text_corpus` records (UTF-8 text)",
            hint="mlforge prepare <MODEL> with the text_corpus "
                 "transform for the eval split (12 §7)",
        )
    records = dataset.get("records")
    if not records:
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} has no prepared records — "
            "the text engine needs `text_corpus` text records",
            hint="mlforge prepare <MODEL> registers the eval split as a "
                 "store artifact first (12 §7: nothing runs on "
                 "unresolved data)",
        )
    texts: list[str] = []
    split_counts: dict[str, int] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValidationBlock(
                f"record {i}: expected a JSON object row, got "
                f"{type(rec).__name__}")
        split = str(rec.get("split") or "").strip().lower()
        if split:
            split_counts[split] = split_counts.get(split, 0) + 1
            if split in _EXCLUDED_SPLITS or split not in _VAL_SPLITS:
                continue  # train/holdout rows are never eval rows
        text = rec.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValidationBlock(
                f"record {i}: `text` must be a non-empty string — "
                "text_corpus records carry extractable text",
                hint="prepare the eval split with transform "
                     "`text_corpus` (12 §7)",
            )
        texts.append(text)
    if not texts:
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} has no eval-split rows — "
            "recorded splits: "
            + (", ".join(f"{k}×{v}" for k, v in sorted(split_counts.items()))
               or "(none)"),
            hint="the protocol evaluates split `val` (12 §15.4) — "
                 "prepare the eval split (split: valid) as its own "
                 "artifact; train/holdout rows are never eval rows",
        )
    subset = protocol.get("sample_subset")
    if subset is not None:
        if isinstance(subset, bool) or not isinstance(subset, int) \
                or subset < 1:
            raise ValidationBlock(
                f"protocol.sample_subset must be a positive integer "
                f"(first N text records), got {subset!r}")
        texts = texts[:subset]

    arch = _arch(entry)
    block = int(arch.block_size)
    corpus = "\n\n".join(texts).encode("utf-8")
    if len(corpus) < block + 1:
        raise ValidationBlock(
            f"eval corpus too small: {len(corpus)} bytes < one sequence "
            f"({block} + 1) — prepare more text for the eval split",
            hint="the trainer's windowing needs at least one full "
                 "block_size window (trainers/text_lm.py)",
        )
    data = torch.frombuffer(bytearray(corpus), dtype=torch.uint8).long()
    n_windows = (len(data) - 1 - block) // block + 1

    model = _model(weights, entry)
    total = 0.0
    scored = 0
    with torch.no_grad():
        for start in range(0, n_windows, _MICRO_WINDOWS):
            xs = torch.stack([data[w * block:(w + 1) * block]
                              for w in range(start,
                                             min(start + _MICRO_WINDOWS,
                                                 n_windows))])
            ys = torch.stack([data[w * block + 1:(w + 1) * block + 1]
                              for w in range(start,
                                             min(start + _MICRO_WINDOWS,
                                                 n_windows))])
            _, loss = model(xs, ys)  # mean over batch*block positions
            positions = xs.shape[0] * block
            total += float(loss) * positions
            scored += positions
    if scored == 0:  # defensive: windowing math above guarantees ≥1
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} yields no scorable windows")
    loss = total / scored
    # trainer's own metric formula (trainers/text_lm.py step())
    return {"perplexity": round(math.exp(min(loss, 20.0)), 6)}


# -- export: TinyGPT → ONNX (opset 17, dynamic batch/sequence) ---------------


def _build_onnx(model, entry: Mapping[str, Any]) -> bytes:
    """The model's forward as a checked ONNX proto (real bytes source).

    torch 2.14's default (dynamo) exporter is used: the legacy
    TorchScript path bakes the example sequence length into
    nn.MultiheadAttention's reshape (measured), which produces a graph
    that is wrong at runtime for any other seq — never exported here."""
    import onnx
    import torch

    arch = _arch(entry)
    example_seq = min(8, int(arch.block_size))

    class _LogitsOnly(torch.nn.Module):
        """TinyGPT.forward with a single tensor output — the loss path
        is trainer-side; the portable artifact scores next-byte logits."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            logits, _ = self.inner(idx, None)
            return logits

    wrapper = _LogitsOnly(model).eval()
    example = torch.zeros(2, example_seq, dtype=torch.long)
    # dynamic batch/sequence via torch.export.Dim — the dynamo
    # exporter's native form (the legacy `dynamic_axes` spelling is a
    # deprecated conversion and would be dropped by a future torch)
    batch_dim = torch.export.Dim("batch")
    seq_dim = torch.export.Dim("sequence")
    program = None
    try:
        program = torch.onnx.export(
            wrapper,
            (example,),
            input_names=["idx"],
            output_names=["logits"],
            opset_version=17,
            dynamic_shapes={"idx": {0: batch_dim, 1: seq_dim}},
            dynamo=True,
        )
        graph = program.model_proto
        graph.ir_version = 8  # portable IR level (onnxruntime-friendly)
        onnx.checker.check_model(graph)
        return graph.SerializeToString()
    except Exception as exc:
        raise ValidationBlock(
            f"text export: torch.onnx.export/checker failed "
            f"({type(exc).__name__}: {exc})\n"
            f"{traceback.format_exc()}",
            hint="no partial export (13 §7) — the exporter's exact "
                 "blocker is reported verbatim, never a fake binary",
        ) from exc
    finally:
        if program is not None:
            try:
                program.release()
            except Exception:  # pragma: no cover — cleanup best-effort
                pass


def _roundtrip(model, proto: bytes) -> dict[str, Any]:
    """torch forward vs onnxruntime on the probe inputs (13 §6.9: max
    error vs tolerance → PASS/FAIL — a FAIL blocks the export)."""
    import numpy as np
    import onnxruntime as ort
    import torch

    block = int(model.arch.block_size)
    shapes = [(b, s) for (b, s) in PROBES if s <= block] or [(1, block)]
    session = ort.InferenceSession(proto, providers=["CPUExecutionProvider"])
    max_error = 0.0
    with torch.no_grad():
        for batch, seq in shapes:
            # (batch, seq) probes derive from the same per-seed stream
            gen = torch.Generator(device="cpu")
            gen.manual_seed(PROBE_SEED + seq)
            idx = torch.randint(0, 256, (batch, seq), generator=gen,
                                dtype=torch.long)
            expected, _ = model(idx, None)
            got = session.run(None, {"idx": idx.numpy()})[0]
            err = float(np.abs(expected.numpy() - np.asarray(got)).max())
            max_error = max(max_error, err)
    return {
        "max_error": round(max_error, 10),
        "tolerance": ATOL,
        "result": "PASS" if max_error < ATOL else "FAIL",
        "harness": "torch-onnx-roundtrip",
        "n_probe": len(shapes),
    }


def export_bytes(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    fmt: str,
) -> tuple[bytes, dict[str, Any]]:
    """Real export: TinyGPT forward → ONNX bytes + round-trip validation.
    Returns (binary, numerical_validation); the workflow blocks on FAIL."""
    if fmt != "onnx":
        raise PreconditionFailed(
            f"cannot export the text model to {fmt} — "
            f"{supports_format(fmt)}",
            hint="13 §6.9: only formats with a real converter produce "
                 "bytes; nothing partial is ever written (13 §7)",
        )
    err = supports_format(fmt)
    if err:
        raise PreconditionFailed(f"cannot export the text model — {err}")
    model = _model(weights, entry)
    proto = _build_onnx(model, entry)
    numerical = _roundtrip(model, proto)
    return proto, numerical


__all__ = [
    "ATOL",
    "FAMILY",
    "MODEL_NAMES",
    "OPERATORS",
    "PROBES",
    "compute_metrics",
    "contract",
    "dependency_error",
    "execute",
    "export_bytes",
    "input_types",
    "metric_names",
    "supports_format",
]
