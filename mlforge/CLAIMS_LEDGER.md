# MLForge Claims Ledger

Evidence ledger for every verification claim made about MLForge: what was
run, what it proves, what it does **not** prove, and how to reproduce it.
Claims are separated into *verified* (an executable check produced the
number), *reported* (computed, but no normative target exists), and
*gaps* (target exists but this environment cannot run it or the harness
does not expose it). Nothing here is a leaderboard number unless it says
so.

Machine-readable artifacts referenced below live outside the repo (they
are run outputs, not source): coverage report, mutation results, and E2E
logs are captured under `/tmp/opencode/` on the verification host and
are reproducible with the commands given.

---

## 1. Test suite

| Claim | Status | Evidence |
|---|---|---|
| Full unit/integration suite passes | **VERIFIED** | `python -m pytest -q` → exit 0, 777 tests before this ledger, 787 with the golden-metric tests, 808 with the device-placement tests (805 passed + 3 skipped — 92 s), **813 with the evaluate_calibration metric tests (810 passed + 3 skipped)** |
| Device placement contract (`runtime.device`) | **VERIFIED** | `tests/test_device_placement.py` → 19 passed + 2 CUDA-host skips: auto/cpu/cuda resolution, fail-closed refusals (no silent downgrade), preflight `device: cuda` ⇒ GPU probe, CPU-state portable checkpoint bytes |
| Branch coverage of `src/mlforge` | **VERIFIED** | `python -m coverage run -m pytest` → **82 %** total (11 427 stmts, 1 709 missed, 3 712 branches, 670 partial) |
| Coverage hot spots (honest low end) | **VERIFIED** | `ops/bundle.py` 49 %, `trainers/rfdetr.py` 48 %, `trainers/reid.py` 75 %, `engines/calibrator.py` 76 %; `machine.py` 100 %, `workflow.py` 90 % |

Reproduce:

```bash
cd mlforge && TMPDIR=~/.pip-tmp python -m coverage run --branch -m pytest -q
python -m coverage report -m
```

## 2. Golden-value metric tests (`tests/test_metric_golden.py`)

Thirteen tests that pin each metric implementation to (a) hand-worked
arithmetic quoted from the specification and (b) an external oracle
where one exists.

| Metric | Spec anchor | Verified property |
|---|---|---|
| NDCG@3/@5 | `10_training_plan/03_training_pipeline.md:493-494` | hand-worked two-group example (0.946767…, 0.630929…, mean 0.788848…) **and exact agreement with LightGBM's `ndcg@k`** (0.9733838380633502 on a third dataset) |
| ECE (15 bins) | `06_benchmarking_plan.md:348`, §5 | 0.05960146101105884 vs. hand-computed binning; 2-bin boundary case 0.2589931049810458 |
| MCE (15 bins) | `06:317`, `06:349` | worst-bin gap 0.11920292202211769 on the ECE rows; 0.5 exact at `n_bins=2` (boundary row present); `mce ≥ ece` invariant |
| Brier (multiclass) | `06:320`, `06:350` | 0.26420933661861107 hand-worked + numpy one-hot oracle; certain-right → 0.0, certain-wrong → 2.0 |
| Per-α coverage / avg set size | `06:351-353` | 5 hand-derived rows → 0.2/0.6 coverage and 0.2/0.6 set size at α=0.05/0.10; delivered keys == `metric_names(weights)` in order |
| NLL | `06:349` (no target) | formula pinned: t=1 → 0.41003759580145893, t=2 → 0.5032044340390841 |
| Conformal coverage | `06:351-352` | n=99: 95/99 and 90/99 at α=0.05/0.10; n=9 clamps to 1.0 |
| Re-ID rank metrics | `06:301` | golden rank1=0.5, rank5=1.0, mAP=0.75 incl. same-cam exclusion and junk (`0000`) removal |
| COCO mAP / AP50 | `06:251`, `06:150` | perfect boxes → 1.0/1.0; IoU 0.57 → mAP 0.2, AP50 1.0 |

Reproduce: `python -m pytest tests/test_metric_golden.py -q` (13 passed).

## 3. Mutation testing (mutmut 3.8)

Per-file mutation runs with a per-target test selection (fastest set of
tests that covers the file). "no tests" = mutants on lines the selected
tests never execute (not counted in the tested kill rate).

| Target | Mutants | Killed | Timeout | No tests | Survived | Kill % (of tested) |
|---|---|---|---|---|---|---|
| `machine.py` | 117 | 56 | 5 | 0 | 56 | 47.9 % strict, **52.1 % incl. timeouts** |
| `leases/run_lease.py` | 297 | 175 | 0 | 0 | 122 | 58.9 % |
| `runtime/checkpoints.py` | 432 | 280 | 0 | 10 | 142 | 65.7 % |
| `ops/bundle.py` | 204 | 79 | 0 | 48 | 77 | 56.0 % |
| `ops/exporting.py` | 574 | 207 | 0 | 23 | 344 | 37.6 % |

`machine.py` used the expanded selection
(`test_machine + test_states + test_workflow + test_journey_reliability`);
the earlier narrow selection (`test_machine + test_states`) gave 55 killed /
62 survived — the extra tests converted 6 survivors into detections
(1 kill + 5 `StateMachine.fire` timeouts, counted as detected because the
suite caught the mutant by hanging). Survivor triage for `machine.py`:
29 × `_guard_resume`, 9 × `_guard_intentional`, 8 ×
`StateMachine.legal_actions`, 7 × `_guard_no_checkpoint`, 3 ×
`StateMachine.__init__` — mostly guard predicates; no claim is made that
all are real gaps (equivalent mutants exist).

**Aggregate: 1 624 mutants — 797 killed, 5 timeout (detected), 81 no
tests, 741 survived → 52.0 % detected of tested (802/1 543).**

Methodology notes (tooling gotchas that materially affect the numbers):

* `config_fingerprint` deliberately excludes `only_mutate` — switching
  target files with an identical test selection silently reuses the
  previous target's stats and marks **every** mutant "no tests". The run
  script deletes `mutants/mutmut-stats.json` per target to force a full
  recollection (verdicts live in `.meta` and survive).
* Survivors are triaged by function (see the `machine.py` clusters above);
  the expanded selection was used precisely to separate equivalent
  mutants from test gaps.
* Some survivors are equivalent mutants (mutation changes no observable
  behaviour); no claim is made that the survivor list is all real gaps.

Reproduce: `/tmp/opencode/run_mutation.sh <target>` (config template and
evidence under `/tmp/opencode/mutation_evidence/`).

## 4. Live end-to-end runs (real engines, `MLFORGE_HARNESS` absent)

| Journey | Scope | Result |
|---|---|---|
| Tier-3 live E2E | GBDT ranker + calibrator: train → publish → ONNX export (binary + round-trip PASS) → structured infer → evaluate (preview == recorded) → CLI evaluate/infer/export → 2 refusals | **34/34 checks** (`/tmp/opencode/e2e_t3.py`) |
| Family E2E (text) | `reasoner_s`: real byte-LM training via Worker → export → UTF-8 text infer (prompt_bytes/perplexity/next_byte/topk, identity-deterministic) → perplexity eval preview==recorded | **45/45 checks total** (`/tmp/opencode/e2e_families.py`) |
| Family E2E (re-ID) | `osnet_x1_0`: real OSNet training on a registered image tree → ONNX export (round-trip max_error 5.96e-08) → image infer (512-d unit embedding, deterministic) → rank1/rank5/mAP eval preview==recorded | included above |
| Family E2E (refusals) | unknown export format → `ValidationBlock`; missing infer input → `NotFound` | included above |
| RF-DETR smoke | `MLFORGE_RFDETR_SMOKE=1 pytest tests/test_rfdetr_trainer.py` — real Lightning epoch + base download + resume | **16 passed** (`/tmp/opencode/rfdetr_smoke.log`) |
| Fault-injection journeys | 7 journeys in the suite: gate refusal, crash+resume, lease expiry, checkpoint corruption, package weights, finetune root, export commit-marker ordering | in `tests/test_journey_reliability.py` (part of the suite) |

## 5. Benchmarks vs. specification targets

Output of `/tmp/opencode/benchmarks.py` (trains through the real Worker,
evaluates through the real engines, compares to the normative target):

| Metric | Target (§) | Actual | Verdict |
|---|---|---|---|
| NDCG@3 | ≥ 0.85 (`03:493`) | 0.980908 | **PASS** (synthetic tabular) |
| NDCG@5 | ≥ 0.80 (`03:494`) | 0.977196 | **PASS** (synthetic tabular) |
| Model size (ONNX) | < 1 MB / 1 000 000 B (`03:496`) | 1363 B | **PASS** (real export binary) |
| Inference latency (median) | < 1 ms CPU (`03:495`) | 0.0171 ms | **PASS** — onnxruntime CPU EP, 50 warm + 2000 timed runs, input `features [1, 10]`, all-0.5 row (mean 0.0177 ms, p95 0.0196 ms) |
| ECE | < 0.05 (`06:348`) | 0.020657 | **PASS** (synthetic logits) |
| MCE | < 0.10 (`06:349`) | 0.020657 | **PASS** (synthetic logits — now exposed by `evaluate`) |
| Coverage (α=0.05) | ≥ 0.95 (`06:351`) | 1.0 | **PASS** — per-α from `evaluate` (mean proxy retired) |
| Coverage (α=0.10) | ≥ 0.90 (`06:352`) | 1.0 | **PASS** — per-α from `evaluate` |
| Avg set size (α=0.05) | 1–3 (`06:353`) | 2.0 | **PASS** (2-class synthetic sets max out at 2; empty sets count 0) |
| Brier | < 0.15 (`06:350`) | 0.469603 | REPORT — fixture Bayes floor is 0.375 at flip=0.25 (75 %×0.125 + 25 %×1.125 for calibrated p=0.75/0.25), so <0.15 is unreachable **by construction** on this noise; number now computed by `evaluate` |
| Re-ID rank1 | > 0.95 (`06:301`) | 1.0 | PASS on synthetic solid-color tree (**not** Market1501) |
| Re-ID mAP | > 0.85 (`06:301`) | 1.0 | PASS on synthetic tree (**not** Market1501) |
| NLL | no target | 0.662488 | REPORT |
| Text perplexity | no target | 28.996789 | REPORT |

**These are pipeline-validation numbers on synthetic data** — they prove
the targets are computable end-to-end through the real harness, not
that a trained production model would hit them on benchmark corpora.

### Gaps (targets that exist but were NOT run / NOT exposed)

| Metric | Target | Why |
|---|---|---|
| Detection mAP | ≥ 53.0 (`06:251`) | needs COCO val + GPU; verification box has neither. Evidence instead: RF-DETR smoke test + golden COCO mAP test |
| Real Market1501 / COCO leaderboard numbers | `06:251`, `06:301` | datasets not present on this host; requires GPU |

(MCE, Brier, avg set size, inference latency and model size were gaps
until the `evaluate_calibration` metric set and the export
size/latency micro-benchmark landed — they are measured in §5 now;
Brier reports its fixture floor instead of a false FAIL.)

## 6. What this ledger does NOT claim

* No claim of benchmark-state-of-the-art performance anywhere.
* No claim that survivor mutants are all real test gaps (equivalent
  mutants exist and are documented as such).
* No claim of cross-platform CI — everything above ran on one Linux
  host (CPU-only, Python 3.14).
* Coverage %, kill rates, and E2E checks are point-in-time; each
  reproduce command regenerates them.
