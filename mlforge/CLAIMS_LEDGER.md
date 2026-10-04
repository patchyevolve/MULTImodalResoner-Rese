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
| Full unit/integration suite passes | **VERIFIED** | `python -m pytest -q` → exit 0, 777 tests before this ledger, 787 with the golden-metric tests, 808 with the device-placement tests (805 passed + 3 skipped — 92 s), 813 with the evaluate_calibration metric tests (810 passed + 3 skipped), **849 with the coverage phases (846 passed + 3 skipped — 121 s)** |
| Device placement contract (`runtime.device`) | **VERIFIED** | `tests/test_device_placement.py` → 19 passed + 2 CUDA-host skips: auto/cpu/cuda resolution, fail-closed refusals (no silent downgrade), preflight `device: cuda` ⇒ GPU probe, CPU-state portable checkpoint bytes |
| Branch coverage of `src/mlforge` | **VERIFIED** | `python -m coverage run -m pytest` → **85 %** total (11 528 stmts, 1 513 missed, 3 736 branches, 633 partial) |
| Coverage hot spots (honest low end) | **VERIFIED** | `trainers/reid.py` 75 %, `ops/engines/calibrator.py` 78 %; closed since the original audit: `ops/bundle.py` 49 % → **100 %**, `trainers/rfdetr.py` 48 % → **100 %**, `trainers/rfdetr_data.py` → **98 %** (3 unreachable lines, documented); `machine.py` 100 %, `workflow.py` 91 % |
| Lint contract | **VERIFIED** | `ruff check src/ tests/` → **0 findings** under the frozen 413-code policy in `mlforge/pyproject.toml` (codes pinned individually, `ruff==0.16.4` in dev extras; intentional blind catches carry line-level `# noqa` with reasons) |

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
tests never execute (not counted in the tested kill rate). Kill % is
computed as `killed / (mutants − no tests)`; machine.py also reports
timeouts counted as detected (the suite caught the mutant by hanging).

### Baseline (before the lint + coverage phases)

| Target | Mutants | Killed | Timeout | No tests | Survived | Kill % (of tested) |
|---|---|---|---|---|---|---|
| `machine.py` | 117 | 56 | 5 | 0 | 56 | 47.9 % strict, **52.1 % incl. timeouts** |
| `leases/run_lease.py` | 297 | 175 | 0 | 0 | 122 | 58.9 % |
| `runtime/checkpoints.py` | 432 | 280 | 0 | 10 | 142 | 66.4 % |
| `ops/bundle.py` | 204 | 79 | 0 | 48 | 77 | 50.6 % |
| `ops/exporting.py` | 574 | 207 | 0 | 23 | 344 | 37.6 % |

(Two cells in the original baseline table mixed percentages from earlier
intermediate runs of the same targets — checkpoints and bundle are
recomputed here from the counts shown, with the formula above. An earlier
claim that bundle was "the weakest module at 37.6 %" misattributed
exporting's number: baseline bundle was 50.6 %; the weakest baseline
target was `exporting.py` at 37.6 %.)

**Aggregate baseline: 1 624 mutants — 797 killed, 5 timeout (detected),
81 no tests, 741 survived → 51.7 % strict, 52.0 % incl. timeouts of
tested (802/1 543).**

### Re-run after the lint + coverage phases (Phase C, 2026-10-04)

Same five targets, same selections, bundle selection extended with the new
`tests/test_bundle.py`; verdicts regenerated from scratch (`.meta` and
`mutmut-stats.json` removed per target).

| Target | Mutants | Killed | Timeout | No tests | Survived | Kill % (of tested) | Δ vs baseline |
|---|---|---|---|---|---|---|---|
| `machine.py` | 117 | 55 | 5 | 0 | 57 | 47.0 % strict, 51.3 % incl. | −0.9 pp (one flipped equivalent, below) |
| `leases/run_lease.py` | 297 | 175 | 0 | 0 | 122 | 58.9 % | ±0 |
| `runtime/checkpoints.py` | 432 | 280 | 0 | 10 | 142 | 66.4 % | ±0 |
| `ops/bundle.py` | 204 | 124 | 0 | 0 | 80 | **60.8 %** | **+10.2 pp** |
| `ops/exporting.py` | 571 | 205 | 0 | 23 | 343 | 37.4 % | −0.2 pp (population change, below) |

**Aggregate re-run: 1 621 mutants — 839 killed, 5 timeout, 33 no tests,
744 survived → 52.8 % strict, 53.1 % incl. timeouts of tested
(844/1 588). The ≥ 52.0 % no-regression bar PASSES (+1.17 pp).**

Delta explanations (both verified, neither is a lost test):

* `exporting.py` 574 → 571 mutants: the lint autofix (UP012) dropped one
  `encode("utf-8")` → `encode()`, removing that literal's 3 string
  mutants (2 killed + 1 survived in baseline) — pure population change.
* `machine.py` populations are byte-identical across runs (117 mutants,
  per-function counts unchanged; the only edit to the file is a `# noqa`
  comment). The one flipped verdict is `_guard_intentional__mutmut_13`,
  which wraps the literal `"pause/stop/crash must be explicitly initiated"`
  — no test references that text, so the mutation is undetectable by the
  suite and the baseline kill was a transient failure; the re-run's
  "survived" is the accurate verdict. All 5 timeouts are the same 5
  mutants in both runs.

`bundle.py` survivor triage (80 survivors; full per-mutant diffs in
`/tmp/opencode/mutation_evidence/bundle_survivor_diffs.txt`, regenerated
from source through mutmut's own mutation enumeration, id-verified 1:1
against the run's `.meta`): clusters `_build_bundle` 56, `_write_bundle`
12, `_load_bundle` 6, `_list_bundles` 4, `_component_integrity` 2. The
dominant gap is structural: **no test parses what `build_bundle` writes**
(the write test asserts only that `bundle.json` exists; load/list tests
hand-seed fixtures), so every write-side manifest-key mutation survives —
one build→load round-trip test asserting manifest contents is the
highest-yield follow-up. The other classes seen by inspection of all 80:
codec/platform equivalents (`"utf-8"`→`"UTF-8"`, `encoding=None`,
`continue`→`break` on single-corrupt-entry layouts, `.get(None)` where the
fixture lacks the key), formatting-only `json.dumps` indent/sort
mutations, and hash-equality survivors. No claim is made that the
remainder are all real gaps.

`machine.py` baseline triage (unchanged): 29 × `_guard_resume`, 9 ×
`_guard_intentional`, 8 × `StateMachine.legal_actions`, 7 ×
`_guard_no_checkpoint`, 3 × `StateMachine.__init__` — mostly guard
predicates; the expanded selection (`test_machine + test_states +
test_workflow + test_journey_reliability`) converted 6 survivors of the
earlier narrow selection into detections.

Methodology notes (tooling gotchas that materially affect the numbers):

* `config_fingerprint` deliberately excludes `only_mutate` — switching
  target files with an identical test selection silently reuses the
  previous target's stats and marks **every** mutant "no tests". The run
  script deletes `mutants/mutmut-stats.json` per target to force a full
  recollection (verdicts live in `.meta` and survive).
* Survivors are triaged by function (see the clusters above); the
  expanded selection was used precisely to separate equivalent mutants
  from test gaps.
* Some survivors are equivalent mutants (mutation changes no observable
  behaviour); no claim is made that the survivor list is all real gaps.

Reproduce: `/tmp/opencode/run_mutation.sh` (all five targets, ~10 min) or
`/tmp/opencode/run_mutation.sh <tag> <src> <tests…>` for one target;
evidence and the baseline-vs-re-run comparison under
`/tmp/opencode/mutation_evidence/` (baseline copy in
`/tmp/opencode/mutation_evidence_baseline/`).

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
| Inference latency (median) | < 1 ms CPU (`03:495`) | 0.0189 ms | **PASS** — onnxruntime CPU EP, 50 warm + 2000 timed runs, input `features [1, 10]`, all-0.5 row (mean 0.0197 ms, p95 0.0242 ms) |
| ECE | < 0.05 (`06:348`) | 0.020657 | **PASS** (synthetic logits) |
| MCE | < 0.10 (`06:349`) | 0.020657 | **PASS** (synthetic logits — now exposed by `evaluate`) |
| Coverage (α=0.05) | ≥ 0.95 (`06:351`) | 1.0 | **PASS** — per-α from `evaluate` (mean proxy retired) |
| Coverage (α=0.10) | ≥ 0.90 (`06:352`) | 1.0 | **PASS** — per-α from `evaluate` |
| Avg set size (α=0.05) | 1–3 (`06:353`) | 2.0 | **PASS** (2-class synthetic sets max out at 2; empty sets count 0) |
| Brier | < 0.15 (`06:350`) | 0.124444 | **PASS** — LEARNABLE fixture (scale=3, flip=0.05, n=60; Bayes floor 0.095), computed by `evaluate` |
| Brier (noisy fixture, flip=0.25) | floor 0.375 | 0.469603 | REPORT — target unreachable by construction on this noise (75 %×0.125 + 25 %×1.125 for calibrated p=0.75/0.25); number still computed |
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
size/latency micro-benchmark landed — they are measured in §5 now.
Brier gets a real PASS on a learnable fixture (flip=0.05); the
noisy-fixture number stays REPORTed next to its Bayes floor rather
than failing a target the data cannot reach.)

## 6. What this ledger does NOT claim

* No claim of benchmark-state-of-the-art performance anywhere.
* No claim that survivor mutants are all real test gaps (equivalent
  mutants exist and are documented as such).
* No claim of cross-platform CI — everything above ran on one Linux
  host (CPU-only, Python 3.14).
* Coverage %, kill rates, and E2E checks are point-in-time; each
  reproduce command regenerates them.
