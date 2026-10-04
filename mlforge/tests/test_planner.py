"""Build step 9 tests — capability planner (12 §13, §8.3, §14, §12.2).

Specs as executable checks:
  * 12 §13.1 capabilities are MEASURED (nvidia-smi), never table-driven;
    absent ⇒ determinate CPU-only set, present-but-broken ⇒ ValidationBlock
  * 12 §13.2 the solver finds micro×accum×world == frozen global_batch
    subject to VRAM/precision constraints; infeasible ⇒ BLOCK with
    migration options (13 §7), never a silent shrug
  * 12 §14 precision fallback (or CPU) ⇒ execution_mode PORTABLE signal
  * 13 §6.1 SHOW PLAN: `micro batch X · accumulation Y · GPUs Z · global
    batch N` before anything is created
  * 12 §12.2 execution segments preserve every hardware context; created
    by the worker only after preflight passes (run stays READY otherwise)
  * plan persistence is identity-gated: unchanged plan ⇒ no journal spam
"""

from __future__ import annotations

import json
import subprocess

import pytest

import mlforge.planner.capabilities as caps_mod
from mlforge.cli.main import main
from mlforge.errors import ValidationBlock
from mlforge.leases import RunLeaseManager, provide_run_lease
from mlforge.planner import (
    Capabilities,
    ExecutionPlan,
    GPUInfo,
    MemoryProfile,
    build_plan,
    detect_capabilities,
    estimate_required_disk_bytes,
    feasible_solutions,
    load_runtime,
    normalize_precision_policy,
    pick,
    solve_or_block,
)
from mlforge.planner.capabilities import _features_from_arch
from mlforge.run_spec import RunSpec
from mlforge.runtime import ScaffoldTrainer
from mlforge.runtime.worker import Worker
from mlforge.states import RunState
from mlforge.validation import (
    RESUME_GATE_STEPS,
    GateContext,
    ValidationGate,
    provide_global_batch,
    provide_pass,
    provide_plan,
    provide_precision,
    provide_topology,
)
from mlforge.workflow import WorkflowAPI

PLANNER_STEPS = ("plan", "global_batch", "precision", "topology")


def _spec(precision="bf16", global_batch=32) -> RunSpec:
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={
            "optimizer": "adamw", "learning_rate": 1e-4, "scheduler": "cosine",
            "loss": "l1", "seed": 42, "global_batch": global_batch,
            "epochs": 50, "precision_policy": precision,
        },
    )


def _cpu(gb: int = 16) -> Capabilities:
    return Capabilities(gpu_count=0, gpus=(), ram_bytes=16 * gb * (1 << 30),
                        cpu_count=8, interconnect="none", source="cpu-only")


def _h100(count: int = 2) -> Capabilities:
    return Capabilities(
        gpu_count=count,
        gpus=tuple(GPUInfo("H100", 81920, "9.0") for _ in range(count)),
        ram_bytes=1 << 40, cpu_count=64, interconnect="nvlink",
        source="nvidia-smi",
    )


def _spec_pass_providers(root, *, real_planner: bool = False):
    """All 16 steps verifiable; planner steps either wired as real
    builtins (omitted from the dict → WorkflowAPI setdefault) or passed."""
    providers = {
        s.id: provide_pass(f"{s.id} verified")
        for s in RESUME_GATE_STEPS
        if s.id not in ("manifest", "schema", "lease", *PLANNER_STEPS)
    }
    providers["lease"] = provide_run_lease(RunLeaseManager(root))
    return providers


# ---------------------------------------------------------------------------
# 12 §13.1 — capabilities are measured
# ---------------------------------------------------------------------------


def test_arch_feature_matrix_is_fail_closed():
    """Volta: fp16/tensor no bf16 · Ampere: bf16 · Hopper: fp8 · unknown: fp32."""
    volta = _features_from_arch("7.5")
    assert volta["fp16"] and volta["tensor_cores"] and not volta["bf16"]
    ampere = _features_from_arch("8.6")
    assert ampere["bf16"] and ampere["fp16"] and not ampere["fp8"]
    hopper = _features_from_arch("9.0")
    assert hopper["fp8"] and hopper["bf16"]
    for unknown in (None, "", "N/A", "garbage", "8.x"):
        feats = _features_from_arch(unknown)
        assert feats["fp32"] and not feats["fp16"] and not feats["bf16"], unknown


def test_detect_without_nvidia_smi_is_cpu_only(monkeypatch):
    monkeypatch.setattr(caps_mod.shutil, "which", lambda _n: None)
    caps = detect_capabilities()
    assert caps.source == "cpu-only"
    assert caps.gpu_count == 0 and caps.gpus == ()
    assert caps.interconnect == "none"
    assert caps.features == {"fp32": True, "fp16": False, "bf16": False,
                             "fp8": False, "tensor_cores": False}
    assert "CPU" in caps.summary


def test_detect_parses_nvidia_smi(monkeypatch):
    def fake_run(cmd, **kw):
        if "nvlink" in cmd:
            out = "GPU 0: active"
        else:
            out = ("NVIDIA H100, 81920, 9.0\nNVIDIA H100, 81920, 9.0\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(caps_mod.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(caps_mod.subprocess, "run", fake_run)
    caps = detect_capabilities()
    assert caps.source == "nvidia-smi" and caps.gpu_count == 2
    assert caps.vram_total_bytes == 81920 * (1 << 20)
    assert caps.supports("bf16") and caps.supports("fp8")
    assert caps.interconnect == "nvlink"


def test_detect_broken_query_blocks(monkeypatch):
    """Present-but-broken ⇒ ValidationBlock (never a CPU guess)."""
    def fake_run(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(caps_mod.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(caps_mod.subprocess, "run", fake_run)
    with pytest.raises(ValidationBlock, match="capability detection failed"):
        detect_capabilities()


def test_detect_unparseable_output_blocks(monkeypatch):
    monkeypatch.setattr(caps_mod.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        caps_mod.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout="totally-unparseable\n", stderr=""),
    )
    with pytest.raises(ValidationBlock, match="unparseable"):
        detect_capabilities()


def test_heterogeneous_gpus_intersect_features_and_min_vram():
    caps = Capabilities(
        gpu_count=2,
        gpus=(GPUInfo("H100", 81920, "9.0"), GPUInfo("T4", 16384, "7.5")),
        ram_bytes=1 << 40, cpu_count=64, interconnect="pcie",
        source="nvidia-smi",
    )
    assert caps.features["fp16"] is True       # both have it
    assert caps.features["bf16"] is False      # T4 decides (intersection)
    assert caps.vram_total_bytes == 16384 * (1 << 20)  # weakest device wins
    assert caps.compute_capability is None     # heterogeneous ⇒ unknown arch


# ---------------------------------------------------------------------------
# 12 §13.2 — the feasibility solver
# ---------------------------------------------------------------------------


def test_solver_finds_spec_example_solutions():
    """12 §13.2: (32,1,1), (8,2,2), (4,4,2) all feasible on 2× H100."""
    labels = [s.label for s in feasible_solutions(32, _h100())]
    assert "32x1x1" in labels and "8x2x2" in labels and "4x4x2" in labels


def test_every_solution_preserves_global_batch():
    for gb in (32, 30, 8):
        for s in feasible_solutions(gb, _h100()):
            assert s.micro_batch * s.grad_accum * s.world_size == gb
            assert gb % (s.micro_batch * s.world_size) == 0


def test_pick_policy_prefers_more_gpus_then_larger_micro():
    chosen = pick(feasible_solutions(32, _h100()))
    assert chosen.world_size == 2          # more GPUs first
    assert chosen.micro_batch == 16        # then the largest micro


def test_infeasible_vram_blocks_with_migration_options():
    tiny = Capabilities(
        gpu_count=1, gpus=(GPUInfo("tiny", 1024, "7.5"),),
        ram_bytes=1, cpu_count=1, interconnect="pcie", source="nvidia-smi",
    )
    with pytest.raises(ValidationBlock) as exc:
        solve_or_block(32, tiny, context="model rf_detr_s")
    msg = str(exc.value)
    assert "global batch 32 is unachievable" in msg
    assert "migration options" in exc.value.render()  # 13 §7: block WITH alternatives


def test_runtime_max_micro_batch_is_respected():
    chosen, _solutions = solve_or_block(32, _h100(), max_micro=2)
    assert chosen.micro_batch <= 2
    assert chosen.micro_batch * chosen.grad_accum * chosen.world_size == 32


def test_cpu_solver_is_world_one():
    chosen, solutions = solve_or_block(32, _cpu())
    assert chosen.world_size == 1
    assert all(s.world_size == 1 for s in solutions)


def test_memory_profile_runtime_override_roundtrip():
    profile = MemoryProfile.from_dict({"base_bytes": 100, "per_sample_bytes": 1})
    assert profile.to_dict() == {"base_bytes": 100, "per_sample_bytes": 1,
                                 "reserve_bytes": profile.reserve_bytes}
    # defaults: 2 GiB base + 96 MiB/sample + 512 MiB reserve
    default = MemoryProfile()
    assert default.base_bytes == 2 * (1 << 30)
    assert default.per_sample_bytes == 96 * (1 << 20)
    assert default.reserve_bytes == 512 * (1 << 20)


# ---------------------------------------------------------------------------
# plan artifact
# ---------------------------------------------------------------------------


def test_cpu_plan_is_portable_with_precision_fallback():
    plan = build_plan(_spec(), _cpu())
    assert plan.product == 32 and plan.world_size == 1
    assert plan.precision_preferred == "bf16" and plan.precision_effective == "fp32"
    assert plan.precision_fallback_used and plan.portable_required
    assert plan.topology["collective_backend"] == "gloo"
    assert plan.topology["node_count"] == 1
    assert plan.topology["world_size"] == 1


def test_gpu_plan_exact_on_h100():
    plan = build_plan(_spec(), _h100())
    assert plan.product == 32
    assert plan.precision_effective == "bf16" and not plan.precision_fallback_used
    assert not plan.portable_required
    assert plan.topology["collective_backend"] == "nccl"
    assert plan.topology["network_fabric"] == "nvlink"
    assert plan.topology["gpus_per_node"] == plan.world_size <= 2


def test_dict_precision_policy_falls_back():
    """fp8 unsupported on Ampere-class ⇒ fp16 fallback recorded."""
    ampere = Capabilities(
        gpu_count=1, gpus=(GPUInfo("A100", 40960, "8.0"),),
        ram_bytes=1 << 40, cpu_count=64, interconnect="pcie",
        source="nvidia-smi",
    )
    plan = build_plan(
        _spec(precision={"preferred": "fp8", "fallback": "fp16"}), ampere
    )
    assert plan.precision_effective == "fp16"
    assert plan.precision_fallback_used and plan.portable_required


def test_both_precisions_unsupported_blocks_with_migration():
    with pytest.raises(ValidationBlock) as exc:
        build_plan(
            _spec(precision={"preferred": "bf16", "fallback": "bf16"}), _cpu()
        )
    assert "precision policy unsupported" in str(exc.value)
    assert "migration options" in exc.value.render()


def test_precision_policy_normalization_and_rejects_junk():
    assert normalize_precision_policy("bf16") == {
        "preferred": "bf16", "fallback": "fp32"}
    assert normalize_precision_policy(
        {"preferred": "bf16", "fallback": "fp16"}
    ) == {"preferred": "bf16", "fallback": "fp16"}
    with pytest.raises(ValidationBlock, match="precision_policy unsupported"):
        normalize_precision_policy(42)
    with pytest.raises(ValidationBlock):
        normalize_precision_policy({})


def test_plan_roundtrip_and_identity(tmp_path):
    plan = build_plan(_spec(), _h100())
    clone = ExecutionPlan.from_dict(plan.to_dict())
    assert clone.identity == plan.identity
    path = tmp_path / "execution_plan.json"
    plan.write(path)
    assert ExecutionPlan.read(path).identity == plan.identity
    # schema change requires migration — never silent reinterpretation
    tampered = dict(plan.to_dict())
    tampered["schema_version"] = 2
    with pytest.raises(ValidationBlock) as exc:
        ExecutionPlan.from_dict(tampered)
    assert "unsupported execution_plan schema_version" in str(exc.value)
    assert "migration" in (exc.value.hint or "")


def test_plan_summary_matches_spec_format():
    plan = build_plan(_spec(), _h100())
    line = plan.summary_line()
    assert line.startswith("micro batch ")
    assert "· GPUs 2 · global batch 32" in line
    cpu_line = build_plan(_spec(), _cpu()).summary_line()
    assert "· CPU · global batch 32" in cpu_line


def test_plan_never_mutates_semantic():
    spec = _spec()
    before = dict(spec.semantic)
    build_plan(spec, _h100())
    build_plan(spec, _cpu())
    assert spec.semantic == before


def test_load_runtime_missing_and_corrupt(tmp_path):
    assert load_runtime(tmp_path) == {}  # missing ⇒ no overrides
    state = tmp_path / "state"
    state.mkdir()
    (state / "runtime.json").write_text("{not json")
    from mlforge.errors import PreconditionFailed

    with pytest.raises(PreconditionFailed, match="runtime config unreadable"):
        load_runtime(tmp_path)


def test_disk_estimate_includes_registered_datasets(tmp_path):
    """12 §7.3 + planner dataset term: 2×4 + 2 + 2 GiB + registry bytes."""
    (tmp_path / "datasets" / "coco_2017").mkdir(parents=True)
    (tmp_path / "datasets" / "coco_2017" / "identity.json").write_text(
        json.dumps({"total_bytes": 12345})
    )
    assert estimate_required_disk_bytes(tmp_path, _spec()) == (
        12 * (1 << 30) + 12345
    )
    # unregistered ref contributes nothing (fail-closed = estimate lower
    # bound still from checkpoints/logs/margin)
    other = _spec()
    assert estimate_required_disk_bytes(tmp_path, other) == 12 * (1 << 30) + 12345
    # execution overrides are honored
    assert estimate_required_disk_bytes(
        tmp_path, _spec(), runtime={"checkpoint_bytes": 1 << 30}
    ) == (2 + 2 + 2) * (1 << 30) + 12345


# ---------------------------------------------------------------------------
# gate steps 10–13
# ---------------------------------------------------------------------------


def _planner_gate() -> ValidationGate:
    providers = {
        s.id: provide_pass(f"{s.id} verified")
        for s in RESUME_GATE_STEPS
        if s.id not in ("manifest", "schema", *PLANNER_STEPS)
    }
    providers.update({
        "plan": provide_plan(),
        "global_batch": provide_global_batch(),
        "precision": provide_precision(),
        "topology": provide_topology(),
    })
    return ValidationGate(providers)


def test_gate_steps_10_to_13_pass_with_builtin_providers(tmp_path):
    wf = WorkflowAPI(tmp_path)
    run_id = wf.create_run(_spec()).run_id
    ctx = GateContext(run_id, tmp_path, _spec())
    ctx.facts["capabilities"] = _h100()
    report = _planner_gate().run(ctx)
    assert not report.blocked, report.render()
    by_id = {c.id: c for c in report.checks}
    for step in PLANNER_STEPS:
        assert by_id[step].verdict == "PASS", by_id[step]
    assert "= 32 preserved" in by_id["global_batch"].detail
    plan = ctx.facts["execution_plan"]
    assert isinstance(plan, ExecutionPlan) and plan.product == 32
    # capabilities measured once, shared across the four steps
    assert ctx.facts["capabilities"] is not None


def test_gate_step_10_emits_portable_signal_on_cpu(tmp_path):
    wf = WorkflowAPI(tmp_path)
    run_id = wf.create_run(_spec()).run_id
    ctx = GateContext(run_id, tmp_path, _spec())
    ctx.facts["capabilities"] = _cpu()
    report = _planner_gate().run(ctx)
    by_id = {c.id: c for c in report.checks}
    assert by_id["plan"].verdict == "PASS"
    assert by_id["plan"].data.get("exact_compatible") is False
    assert "PORTABLE" in by_id["plan"].detail
    assert by_id["precision"].data.get("exact_compatible") is False


def test_gate_step_11_rejects_tampered_plan(tmp_path):
    """global_batch invariant is checked, not assumed (12 §8.3)."""
    wf = WorkflowAPI(tmp_path)
    run_id = wf.create_run(_spec()).run_id
    ctx = GateContext(run_id, tmp_path, _spec())
    ctx.facts["capabilities"] = _h100()
    import dataclasses

    good = build_plan(_spec(), _h100())
    ctx.facts["execution_plan"] = dataclasses.replace(
        good, micro_batch=7, grad_accum=1  # 7×1×2 = 14 != 32
    )
    report = _planner_gate().run(ctx)
    step11 = next(c for c in report.checks if c.id == "global_batch")
    assert step11.verdict == "FAIL"
    assert "!= frozen global_batch 32" in step11.detail
    assert report.blocked and report.failed_step == 11


def test_gate_step_13_rejects_oversized_world(tmp_path):
    wf = WorkflowAPI(tmp_path)
    run_id = wf.create_run(_spec()).run_id
    ctx = GateContext(run_id, tmp_path, _spec())
    import dataclasses

    plan = build_plan(_spec(), _h100())
    caps = _h100(count=1)  # measured: ONE gpu
    ctx.facts["capabilities"] = caps
    # plan built elsewhere claiming 2 GPUs (e.g. stale artifact)
    stale = dataclasses.replace(
        plan, world_size=2,
        topology={**plan.topology, "world_size": 2, "gpus_per_node": 2},
        capabilities=caps.to_dict(),
        capabilities_identity=caps.identity,
    )
    ctx.facts["execution_plan"] = stale
    report = _planner_gate().run(ctx)
    step13 = next(c for c in report.checks if c.id == "topology")
    assert step13.verdict == "FAIL"
    assert "exceeds measured" in step13.detail


def test_gate_planner_steps_fail_closed_without_providers(tmp_path):
    """Unwired planner steps stay FAIL unverifiable — never skip."""
    wf = WorkflowAPI(tmp_path)
    run_id = wf.create_run(_spec()).run_id
    gate = ValidationGate({
        s.id: provide_pass("ok")
        for s in RESUME_GATE_STEPS
        if s.id not in ("manifest", "schema", *PLANNER_STEPS)
    })
    report = gate.run(GateContext(run_id, tmp_path, _spec()))
    assert report.blocked
    assert report.failed_step == 10
    assert "unverifiable" in report.first_failure.detail


# ---------------------------------------------------------------------------
# workflow: persistence + segments
# ---------------------------------------------------------------------------


def test_validate_run_persists_plan_once(tmp_path):
    """13 §6.1: plan lands on the validated run; unchanged identity never
    journals twice (re-validate on READY is quiet)."""
    wf = WorkflowAPI(tmp_path, gate_providers=_spec_pass_providers(tmp_path))
    run_id = wf.create_run(_spec()).run_id
    report = wf.validate_run(run_id)
    assert not report.blocked, report.render()
    assert wf.get_run_state(run_id) == RunState.READY.value

    path = tmp_path / "runs" / run_id / "execution_plan.json"
    assert path.is_file()
    on_disk = ExecutionPlan.read(path)
    assert on_disk.product == 32 and on_disk.schema_version == 1
    assert wf.get_execution_plan(run_id) == on_disk.to_dict()

    events = [e for e in wf.get_run_events(run_id)
              if e["event"] == "plan_generated"]
    assert len(events) == 1
    assert "to" not in events[0]  # plan is not a state transition
    assert events[0]["plan_hash"] == on_disk.identity

    # report-only re-validation on READY: same identity ⇒ still one event.
    # The gate's lease step 15 re-acquires — held by the first validation,
    # so this report-only pass wires lease as plain PASS (the identity of
    # interest here is the PLAN's, not the lease's).
    second_providers = _spec_pass_providers(tmp_path)
    second_providers["lease"] = provide_pass("lease held from validation")
    again = wf.validate_run(run_id, gate=ValidationGate({
        **second_providers,
        "plan": provide_plan(),
        "global_batch": provide_global_batch(),
        "precision": provide_precision(),
        "topology": provide_topology(),
    }))
    assert not again.blocked
    events = [e for e in wf.get_run_events(run_id)
              if e["event"] == "plan_generated"]
    assert len(events) == 1


def test_validate_blocked_writes_no_plan(tmp_path):
    """Gate BLOCK (impossible precision policy) ⇒ no artifact, no state:
    the plan only lands with a validated run (13 §6.1). The solver hits
    the precision constraint while building the plan, so fail-fast reports
    it at step 10 — same refusal, earlier step (12 §18 first-FAIL rule)."""
    wf = WorkflowAPI(tmp_path, gate_providers=_spec_pass_providers(tmp_path))
    spec = _spec(precision={"preferred": "fp8", "fallback": "fp8"})
    run_id = wf.create_run(spec).run_id
    report = wf.validate_run(run_id)
    assert report.blocked
    assert report.failed_step == 10
    assert "precision policy unsupported" in report.first_failure.detail
    assert not (tmp_path / "runs" / run_id / "execution_plan.json").is_file()
    assert wf.get_execution_plan(run_id) is None


def test_create_execution_segment_ordinals_and_journal(tmp_path):
    wf = WorkflowAPI(tmp_path)
    run_id = wf.create_run(_spec()).run_id
    caps = _h100()
    plan = build_plan(_spec(), caps)
    first = wf.create_execution_segment(run_id, capabilities=caps, plan=plan)
    second = wf.create_execution_segment(run_id, capabilities=caps, plan=plan)
    assert (first, second) == (1, 2)
    seg = tmp_path / "runs" / run_id / "segments" / "segment_0002.json"
    payload = json.loads(seg.read_text())
    assert payload["ordinal"] == 2
    assert payload["capabilities_identity"] == caps.identity
    assert payload["plan_hash"] == plan.identity
    assert payload["precision_effective"] == "bf16"
    events = [e for e in wf.get_run_events(run_id)
              if e["event"] == "segment_created"]
    assert [e["ordinal"] for e in events] == [1, 2]
    assert "to" not in events[0]


# ---------------------------------------------------------------------------
# worker: segment created after preflight only
# ---------------------------------------------------------------------------


def _make_ready(tmp_path):
    wf = WorkflowAPI(tmp_path, gate_providers=_spec_pass_providers(tmp_path))
    h = wf.create_run(_spec())
    report = wf.validate_run(h.run_id)
    assert not report.blocked, report.render()
    token = RunLeaseManager(tmp_path).status(h.run_id).session_token
    assert token
    return wf, h.run_id, token


def test_worker_creates_segment_after_preflight(tmp_path):
    wf, run_id, token = _make_ready(tmp_path)
    code = Worker(
        tmp_path, run_id, token,
        trainer=ScaffoldTrainer(max_steps=3, steps_per_epoch=2),
        poll_interval=0.0, heartbeat_interval=60.0,
        checkpoint_interval=2,
    ).run()
    assert code == 0
    seg_dir = tmp_path / "runs" / run_id / "segments"
    files = sorted(seg_dir.glob("segment_*.json"))
    assert [f.name for f in files] == ["segment_0001.json"]
    payload = json.loads(files[0].read_text())
    assert payload["capabilities"]["source"] in ("cpu-only", "nvidia-smi")
    events = [e for e in wf.get_run_events(run_id) if e["event"] == "segment_created"]
    assert len(events) == 1
    # segment follows preflight in the journal (create only after start)
    order = [e["event"] for e in wf.get_run_events(run_id)]
    assert order.index("preflight_passed") < order.index("segment_created")


def test_worker_detection_failure_leaves_run_ready_no_segment(
    tmp_path, monkeypatch
):
    """Capability failure BEFORE preflight ⇒ exit 1, run untouched (READY)."""
    wf, run_id, token = _make_ready(tmp_path)

    def boom():
        raise ValidationBlock("capability detection failed (test)")

    monkeypatch.setattr("mlforge.runtime.worker.detect_capabilities", boom)
    code = Worker(
        tmp_path, run_id, token,
        trainer=ScaffoldTrainer(max_steps=1, steps_per_epoch=1),
        poll_interval=0.0, heartbeat_interval=60.0,
    ).run()
    assert code == 1
    assert wf.get_run_state(run_id) == RunState.READY.value
    assert not (tmp_path / "runs" / run_id / "segments").exists()


# ---------------------------------------------------------------------------
# CLI — 13 §6.1 SHOW PLAN
# ---------------------------------------------------------------------------


def _write_config(tmp_path, *, runtime: dict | None = None,
                  **semantic_extra) -> str:
    data = {
        "schema_version": 1,
        "model": "rf_detr_s",
        "train_datasets": ["coco_2017:v1"],
        "semantic": {
            "optimizer": "adamw", "learning_rate": 1e-4,
            "scheduler": "cosine", "loss": "l1", "seed": 42,
            "global_batch": 32, "epochs": 50, "precision_policy": "bf16",
            **semantic_extra,
        },
    }
    if runtime:
        data["runtime"] = runtime
    p = tmp_path / "run.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_train_shows_execution_plan_then_cancel(tmp_path, capsys, monkeypatch):
    cfg = _write_config(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    code = main(["--root", str(tmp_path), "train", "--config", cfg])
    assert code == 0
    out = capsys.readouterr().out
    assert "Execution plan" in out
    assert "micro batch 32" in out and "global batch 32" in out
    assert "Cancelled — nothing was created" in out
    assert WorkflowAPI(tmp_path).list_runs() == []  # nothing created


def test_train_infeasible_plan_exits_1_before_creation(
    tmp_path, capsys, monkeypatch
):
    cfg = _write_config(tmp_path, precision_policy={"preferred": "fp8",
                                                    "fallback": "fp8"})
    monkeypatch.setattr("builtins.input", lambda _p: "y")
    code = main(["--root", str(tmp_path), "train", "--config", cfg])
    assert code == 1
    err = capsys.readouterr().err
    assert "precision policy unsupported" in err
    assert WorkflowAPI(tmp_path).list_runs() == []  # blocked BEFORE create


def test_train_accepts_plan_and_confirms(tmp_path, capsys, monkeypatch):
    cfg = _write_config(tmp_path, runtime={"max_micro_batch": 4})
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    main(["--root", str(tmp_path), "train", "--config", cfg])
    out = capsys.readouterr().out
    assert "Execution (runtime overrides)" in out
    # solver honored the override: micro batch ≤ 4
    import re as _re

    m = _re.search(r"micro batch (\d+) · accumulation (\d+)", out)
    assert m and int(m.group(1)) <= 4
    assert int(m.group(1)) * int(m.group(2)) == 32  # invariant holds
