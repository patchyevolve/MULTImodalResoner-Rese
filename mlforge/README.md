# MLForge

A hardware-agnostic, project-agnostic ML training system: immutable experiment
identity, cryptographic artifact identities, fail-closed validation, explicit
resume/fork semantics, crash reconciliation, and portable run folders.

## Install

Zero runtime dependencies — the wheel is self-contained and installs
anywhere Python ≥ 3.11 lives:

```bash
# recommended: isolated CLI on your PATH
pipx install /path/to/repo/multimodal_reasoner_research/mlforge

# or into any virtualenv / conda env
pip install /path/to/repo/multimodal_reasoner_research/mlforge

# or build a portable wheel for lab machines (offline-installable)
python -m pip wheel . -w dist --no-deps
pip install dist/mlforge-0.1.0-py3-none-any.whl
```

Then `mlforge` is a single command from any directory (`--root` defaults
to the current directory):

```bash
mlforge                  # opens the command reference (exit 0)
mlforge --version        # mlforge 0.1.0
mlforge status           # overview of runs in this directory's workspace
mlforge watch            # live dashboard (TUI)
mlforge gui              # localhost web dashboard (http://127.0.0.1:8765)
python -m mlforge ...    # same entry point when scripts-dir isn't on PATH
```

`pip install --user` places the command in `~/.local/bin` — add that to
PATH if your shell does not find `mlforge`.

## Quickstart (try-it flow)

```bash
mlforge init myproj && cd myproj          # project scaffold (13 §4.1)
mlforge dataset add coco_2017 /data/coco  # register + explicit path
mlforge dataset verify coco_2017          # re-hash → VERIFIED
mlforge configure datasets                # machine-local paths ($MLFORGE_HOME)
$EDITOR ingestion.yaml                    # scaffolded: model → transform (12 §10.2)
mlforge prepare rf_detr_s                 # derived rf_detr_s_prepared (§6.6)
$EDITOR configs/train.example.json        # train_datasets: [<model>_prepared:v1]
mlforge train --config configs/train.example.json   # gate 16/16 → preflight → run
mlforge watch                             # live dashboard while it trains
```

The gate is fail-closed at every step: raw datasets BLOCK with the
exact `mlforge prepare <MODEL>` command, missing captures BLOCK, disk/
GPU expectations derive from the run's runtime config + execution plan
(12 §7.3/§14), and a passing gate acquires the run lease before launch
(12 §18 step 15).

### Train a text model on your own books/papers (CPU is fine)

```bash
mlforge dataset add books /path/to/your/books --yes   # pdf/docx/md/txt/zip
mlforge configure datasets --set books=/path/to/your/books
$EDITOR ingestion.yaml        # add: reasoner_s / transform: text_corpus
mlforge prepare reasoner_s    # extract → chunk → reasoner_s_prepared:v1
mlforge train --config configs/train.example.json     # model: reasoner_s,
                                    # loss: byte_cross_entropy, fp32
mlforge watch                  # real loss (starts ≈ 5.55 = ln 256)
```

Real gradients on real text (verified end to end: loss 5.56 → 2.68
over 6 epochs on CLRS + OS Concepts + CTCI; kill -9 mid-run →
`lease break` → `resume` continues with optimizer + LR schedule
intact). Detection models (RF-DETR) still require their GPU runtime —
asking to train one here refuses with exactly that reason.

### CPU-only laptops

Nothing in the quickstart needs a GPU — the system measures whatever
host it runs on:

* **GPU expectations follow the plan, never a blind guess.** No
  `nvidia-smi` ⇒ CPU/PORTABLE plan (fp32; `bf16` falls back with a
  warning — 12 §14). `train`/`resume`/`preflight` only demand a driver
  when the execution plan — or, before validation, live host detection —
  actually found GPUs; explicit escape hatches:
  `runtime: {"gpu": false|true}` in the train config, or
  `mlforge preflight RUN --gpu/--no-gpu` for a one-off report.
* **Device placement is a runtime knob, never hard-wired.** Every
  trainer resolves `runtime.device`: `auto` (default) picks cuda when
  THIS host has it, else cpu; explicit `cpu` / `cuda` / `cuda:N` is
  honored literally — requesting `cuda` on a GPU-less host refuses at
  trainer build with exactly that reason, never a silent downgrade.
  Checkpoints serialize CPU-state bytes (a run folder saved on a GPU
  host stays loadable anywhere), and `device: cuda` makes preflight
  demand `nvidia-smi` (the `runtime.gpu` escape hatch still wins).
* **Disk (12 §7.3) defaults are worst-case**: 2×4 GiB checkpoints +
  2 GiB logs + 2 GiB safety margin + dataset bytes. On a small disk,
  put your real numbers in the train config — gate step 16 and
  preflight both read the same `state/runtime.json`:

  ```json
  "runtime": {
    "checkpoint_bytes": 268435456,
    "log_bytes": 134217728,
    "safety_margin_bytes": 134217728,
    "min_ram_bytes": 4294967296
  }
  ```

* **RAM floor** is 8 GiB by default; `runtime.min_ram_bytes` lowers it.
  Every disk/RAM failure message names the exact knob to change.

## Configure

Two layers — **identity is portable, paths are machine-local** (12 §10.1):

| Where | What |
|---|---|
| `datasets.yaml` (workspace) | dataset_id → identity hash; ships with the repo; **never** contains absolute paths |
| `$MLFORGE_HOME/datasets.json` (default `~/.mlforge`) | dataset_id → this machine's directory — what `configure datasets` edits |
| `project.yaml` (workspace) | project name/schema/created/MLForge version — written by `init`, never rewritten |
| `ingestion.yaml` (workspace) | which dataset feeds which model (below) |
| `configs/*.json` (workspace) | train configs (below) |

* `mlforge configure datasets` — list every registration and this machine's
  path for it.
* `mlforge configure datasets --set ID=PATH` — re-point one or more
  (repeatable, non-interactive). `dataset add` already records the path on
  the machine where it ran; use `configure` after cloning the project to a
  new machine, moving data, or re-pointing several datasets at once.
* Every candidate path is **hashed against its registration before being
  written** — a wrong path is an explicit mismatch, never a silent
  acceptance.

A path is **never part of dataset identity** — the same bytes are the same
dataset on every machine; a wrong path fails later at `prepare`/gate
verification (exit 1) instead of corrupting identity.

## The files you write

### `ingestion.yaml` — what feeds what (workspace root)

```yaml
models:
  reasoner_s:
    transform: text_corpus            # required; unknown transform ⇒ exit 1
    train_sources: [corpus_train:train]
    val_sources: [corpus_val:val]     # optional; never the train split
    depends_on: []                    # optional upstream models (must be AVAILABLE)
  calibrator:
    transform: calibration
    train_sources:
      - {generated_from: rf_detr_s, dataset: det_preds}   # another model's output
    val_sources: [det_val:val]
```

Source forms (12 §10.2):

| Form | Meaning |
|---|---|
| `name:split` | a registered dataset (`mlforge dataset add`); the split is source metadata |
| `{generated_from: <model>, dataset: <ID>}` | output of another model — `generated_from` is provenance (that model must be **AVAILABLE** in the registry), `dataset` names the registered bytes |

Fail-closed — the DAG never warns-and-continues:

| Condition | Exit |
|---|---|
| missing/unreadable `ingestion.yaml` | 3 |
| unknown key or unknown transform | 1 |
| unregistered source | 2 |
| content hash mismatch vs registration | 1 |
| unmet `depends_on` / `generated_from` | 3 |

### `configs/<name>.json` — the train config

```json
{
  "schema_version": 1,
  "model": "reasoner_s",
  "train_datasets": ["dataset://reasoner_s_prepared:v1"],
  "val_dataset": "dataset://reasoner_s_prepared:v1",
  "semantic": {
    "optimizer": "adamw", "learning_rate": 0.003, "scheduler": "cosine",
    "loss": "byte_cross_entropy", "seed": 42, "global_batch": 4,
    "epochs": 2, "precision_policy": "fp32"
  },
  "runtime": { "gpu": false, "device": "auto" }
}
```

* **Identity keys** — `model`, `train_datasets`, `val_dataset`,
  `semantic`, `schema_version`. An unknown schema version BLOCKs; it is
  never silently reinterpreted.
* **Execution keys** — everything under `runtime`: `gpu` (bool),
  `device` (`auto`/`cpu`/`cuda[:N]`), `checkpoint_bytes`, `log_bytes`,
  `safety_margin_bytes`, `min_ram_bytes` (see Quickstart). Changing
  `runtime` does **not** change run identity; changing `semantic` does.
* `fork` / `retrain` / `finetune` take a **partial** config —
  `{"semantic": {...}, "runtime": {...}}`, both optional — as base
  overrides; `--set` applies on top (explicit wins).

Validation happens at creation, not mid-run: an unknown key or a missing
required semantic field BLOCKs before anything exists.

## Transforms (the data types)

`mlforge dataset types` prints this catalog. Only registered code runs —
an unknown transform, or one that yields zero records, is a BLOCK, never a
silent skip (12 §6.4):

| Transform | Accepts | Produces | Trains |
|---|---|---|---|
| `text_corpus` | `.pdf .docx .md .txt .rst .zip` | extracted, paragraph-chunked text records | `reasoner_s` |
| `coco_detection` | images + COCO `.json` | file records + rehashed annotations | `rf_detr_s`, `rf_detr_l`, `rf_detr_seg_s` |
| `reid_crops` | `identity_camera*.jpg` folders | identity/camera manifest | `osnet_x1_0` |
| `tabular` | `.csv .tsv .jsonl .xlsx` | strict row records | `hypothesis_ranker` |
| `calibration` | `.jsonl/.json` of `logits\|probs` + `label` | calibrator rows | `calibrator` (usually fed via `generated_from`) |
| `mot_challenge` | MOT `.csv/.txt` + frames | validated per-frame boxes | — data-side only; `train` refuses honestly |
| `video_clips` | `.mp4 .mkv .avi .mov .webm` + COCO json | decoded clip probes | — data-side only |
| `soccernet_events` | `Labels-v2.json` / `video.ini` | 17-class event records | — data-side only |

`audio` is declared not-supported (fail-closed, same as an unknown
transform). The transform is chosen per model in `ingestion.yaml`;
`prepare <model>` materializes `<model>_prepared:v1` from it, which is what
the train config's `train_datasets` points at.

## Semantic fields — what each knob does

Eight fields are required and validated **at run creation** (missing ⇒
BLOCK, exit 1). Together they are the run's frozen identity: after
creation no field can be edited in place — change one and you
`fork`/`retrain` (new lineage edge; the old run stays untouched).

| Field | What it controls | ↑ higher | ↓ lower / watch out |
|---|---|---|---|
| `learning_rate` | step size of the optimizer | faster early progress — too high and the loss diverges; the run is wasted | stable but slower; sane ranges differ per family (text ≈ 3e-3, detection ≈ 1e-4; on the ranker this is LightGBM's shrinkage) |
| `global_batch` | examples per optimizer step — frozen; the planner decomposes `micro × accum × world` to match it, infeasible combination ⇒ BLOCK with migration options | fewer, larger steps ⇒ better throughput, fewer LR updates per epoch (consider raising `learning_rate` with it) | more updates, noisier gradients, more steps per epoch (slower wall-clock) |
| `epochs` | passes over the data | more training; on `hypothesis_ranker` **epochs IS the boosting-round count**; on `calibrator` it must equal the number of fit phases (mismatch ⇒ BLOCK) | undertrained — on books, 6 epochs took the text loss from 5.56 → 2.68 |
| `scheduler` | LR curve after a 5 % warmup: `cosine` decays smoothly to 10 % of the base LR, `linear` decays straight to 0, `constant` never decays | — choice, not magnitude: cosine usually generalizes best for text/vision | a different curve is a different experiment (identity change) |
| `optimizer` | update rule (family allowlist — wrong value ⇒ refusal naming the supported set) | — | torch families: `adamw` (weight-decay-corrected adaptive), `adam`, `sgd` (plain — tune LR with it); ranker: `gbdt`/`boosting`/`none`; calibrator: `search` (closed form) / `none` |
| `loss` | objective (family allowlist) | — | text: `byte_cross_entropy` (default), `cross_entropy`, `next_token`; detection: `l1`; re-ID: `cross_entropy`; ranker: `lambdarank`; calibrator: `nll` |
| `seed` | RNG for weight init and shuffle order | — | same seed ⇒ bit-identical init/sampler on a given host; a different seed is a different run (identity) |
| `precision_policy` | numerics: `fp32` everywhere; `bf16` prefers half precision where the host supports it (CPU-only falls back with a warning) | `bf16` ⇒ faster/less memory on GPU hosts | `fp32` ⇒ portable and always exact; detection also accepts a `{"preferred": ...}` dict form |

Optional keys — their presence is recorded in identity too:

| Key | Family | Effect |
|---|---|---|
| `windows_per_epoch` | text | cap windows drawn per epoch — ↑ sees less data per epoch (faster epochs); `0`/absent = all windows |
| `pretrained` | re-ID | init from the public OSNet ImageNet weights (default: from scratch) |
| `alphas`, `decomposition` | calibrator | conformal coverage levels (arch default if absent); `decomposition` adds a fit phase — it changes the required `epochs` |
| `label_column`, `group_column`, `feature_columns` | ranker | tabular schema mapping (defaults: `label`, `group`, the spec's feature names) |

`--set` on `fork`/`retrain`/`finetune` accepts only the allowlist
(`lineage.py`): `lr`, `batch`, `precision` are recognized aliases for
`learning_rate`, `global_batch`, `precision_policy`, plus
`finetune_strategy`, `pretrained`, `windows_per_epoch`. A typo BLOCKs
**before anything exists**.

## Command reference

Quickstart shows the happy paths; every command also has
`mlforge <cmd> --help`. Common flags: `--root PATH` (default `.`),
`--json`, `--command-id ID` (idempotent retry key), `--attach` (stream a
run; Ctrl+C detaches, training continues), `--yes` (skip confirmation).
Exit codes: `0` ok · `1` validation block · `2` not found · `3`
precondition · `4` runtime (`errors.py`).

| Area | Command | Does |
|---|---|---|
| project | `init NAME` | workspace + portable defaults |
| | `configure datasets [--set ID=PATH]` | machine-local dataset paths |
| data | `dataset add ID PATH [--version V] [--force]` | register + hash + point |
| | `dataset list` / `dataset verify ID` / `dataset types` | inventory / tamper check / transform catalog |
| | `prepare MODEL` | resolve → transform → `MODEL_prepared:v1` |
| train | `train --config CFG` | create → 19-step gate → preflight → spawn supervisor |
| | `resume` / `pause` / `stop RUN` | checkpoint continuation / committed pause / stop |
| | `fork RUN` / `retrain MODEL` / `finetune MODEL` | new run from a lineage edge; `--set k=v`, `--config` base overrides |
| | `validate RUN\|MODEL` / `preflight RUN [--gpu]` / `lease status\|break` | gate replay / capability check / run lease |
| model | `model list` / `model inspect M` / `model import` | registry + lineage / detail / external package |
| | `evaluate MODEL --dataset DS [--protocol F]` | metrics on a registered split (eval split, never `train`) |
| | `compare M1 [M2 ...]` / `infer MODEL INPUT_FILE` | side-by-side metrics / one prediction against the contract |
| | `export MODEL --format FMT` / `package MODEL` / `store gc` | onnx·openvino·tensorrt·coreml·tflite / deploy bundle / artifact GC (dry-run default) |
| observe | `status [-v] [RUN]` / `inspect OBJ` | L1 overview (+L2 detail) / object fact sheet |
| | `events RUN [--follow]` / `watch [RUN]` / `hardware` | journal stream / live dashboard / measured host plan |
| | `gui` / `serve` | local dashboard / honest refusal (exit 4 until packaged) |

## Ground truth (normative)

Implementation follows these two documents exactly; where code and docs
disagree, the docs win until amended:

| Doc | Role |
|---|---|
| `../10_training_plan/12_training_system.md` | **Architecture v1.0** — the machinery (identity model, validation gate, checkpoints, leases, reconciliation, planner) |
| `../10_training_plan/13_product_specification.md` | **Product v1.0** — CLI contract, state machines, sequence diagrams, failure matrix, status plane |
| `../10_training_plan/00_master_plan.md` | What/why, training strategy |

## Layout

```
mlforge/
├── pyproject.toml            # zero runtime deps (stdlib-only core)
├── src/mlforge/
│   ├── errors.py             # exit codes 0-4 + error hierarchy   ✅
│   ├── ids.py                # run_/model_/cmd_ ULID ids          ✅
│   ├── hashing.py            # canonical JSON + sha256 identity   ✅
│   ├── journal.py            # append-only events (authority)     ✅
│   ├── states.py             # lifecycle states + FAILED disposition ✅
│   ├── machine.py            # 4 transition tables (13 §5)        ✅
│   ├── run_spec.py           # immutable semantic identity        ✅
│   ├── lineage.py            # lineage DAG + fork/retrain/finetune plans ✅
│   ├── workflow.py           # Workflow API facade (13 §1)        ✅
│   ├── cli/                  # status/inspect/events/store/validate/preflight/lease/train/resume/pause/stop/watch/hardware/dataset/prepare/fork/retrain/finetune/model ✅
│   ├── yamlmini.py           # strict YAML subset (fail-closed)   ✅
│   ├── ingest/               # identity, paths, DAG, transforms, prepare ✅
│   ├── store/                # CAS + registry + GC                ✅
│   ├── validation/           # 19-step gate + preflight           ✅
│   ├── leases/               # run lease + gate providers 15–16   ✅
│   ├── commands/             # idempotency journal                ✅
│   ├── supervisor.py         # crash detection + spawn queue daemon  ✅
│   ├── runtime/              # checkpoints, worker, control, trainer resolution ✅
│   ├── trainers/             # REAL learning loops — text byte-LM, OSNet re-ID, RF-DETR, calibrator, LightGBM ranker ✅
│   ├── status/               # L1/L2/L3, watch, events --follow   ✅
│   └── planner/              # capability feasibility solver      ✅
└── tests/                    # specs as executable tests
```

## Build order (13 §11) — status

1. ✅ Workflow API + state machines (project/dataset/run/model)
2. ✅ Artifact registry + content store + run_spec canonical hashing
3. ✅ Validation gate + preflight (fail-closed core) — the default
   provider set now covers ALL 16 steps: source/environment captures
   written at run creation (`code/source.snapshot.tar.gz` + tree hash +
   `environment/fingerprint.json`, 12 §6.4), store-backed prepared
   datasets (identity re-hashed from the content store), transform
   (EXACT-required artifact verify), model (from-scratch =
   source-derived; finetune base weights = registry + COMMIT marker),
   driver/hardware (live detection), checkpoint (newest-valid), lease +
   revalidate (TOCTOU close, session token reused by launch), and the
   disk/GPU expectations shared between step 16 and preflight
4. ✅ CLI contract — every §4.1 command implemented (`init`,
   `configure datasets`, train/resume/pause/stop, fork/retrain/finetune,
   dataset/prepare, validate/preflight/lease, evaluate/compare/infer/
   export/package, watch/gui); `serve` alone stays honest exit 4 (no
   build step will ever deliver it — never fake success)
5. ✅ Supervisor daemon + run leases + idempotency journal
6. ✅ Training runtime — **REAL learning loops for every file-11 weight
   target.** text: the `reasoner_s` byte-level transformer trains with
   genuine gradients/AdamW/cosine schedule on prepared text
   (`text_corpus` transform) — loss falls from ≈5.55 (ln 256) into the
   2s on books. re-ID: `osnet_x1_0` on prepared `reid_crops`
   (cross-entropy over string id_map, exact resume with bitwise-equal
   weights). detection: `rf_detr_s/l/seg_s` real Lightning epochs on
   `coco_detection` (published base weights fetched into the RF_HOME
   model cache, never the working directory). calibration: the
   `calibrator` fits temperature (NLL search) + conformal thresholds
   (+ optional decomposition weights) on `calibration` prediction rows —
   closed-form, dependency-free. ranking: `hypothesis_ranker` grows
   LambdaMART rounds one LightGBM boost at a time on grouped `tabular`
   rows (spec params, min_data_in_leaf floor enforced). Transactional
   checkpoints (12 §11 write protocol / newest-valid predicate /
   verify / components) carry REAL model + optimizer + scheduler + RNG +
   sampler state (12 §11.4, all 15 REQUIRED_COMPONENTS), so `resume` —
   including after SIGKILL → INTERRUPTED → reconcile → `lease break` →
   `resume` — CONTINUES the same run (LR schedule and weights intact),
   heartbeat writer, deterministic reconciliation
   (`reconcile_from_disk`), worker process (start contract,
   control-channel pause/stop, checkpoint loop, completion checkpoint so
   the published model = final weights), supervisor spawn queue +
   daemon (`state/pending/` → detached worker, 12 §12.4),
   `train`/`resume`/`pause`/`stop` CLI semantics (gate BLOCK →
   FAILED[FORK_ONLY], `--command-id` dedupe).
   Fail-closed everywhere else: model with no integrated trainer,
   missing framework (torch / lightgbm — honest refusal naming the pip
   command), non-fp32 plan (no AMP), world_size > 1 (no distributed),
   or an unsupported semantic loss/optimizer ⇒ refusal with the
   concrete reason — never a substitute. `ScaffoldTrainer` remains
   system-test harness only (explicit
   `runtime.trainer=harness-scaffold` / `MLFORGE_HARNESS=1`).
7. ✅ Status layer — read-only L1/L2 (`status [RUN] [-v]` incl. stale-
   heartbeat WARNING + FAILED disposition + stage-aware block), L3
   `hardware` (diagnostic telemetry), `watch [RUN]` (viewer only — keys
   write control intents, `q` quits the viewer), `events --follow`;
   worker publishes `state/heartbeat.json` (global_step/state) +
   `state/live.json` (transient metrics) per 13 §9.4
8. ✅ Ingestion/transform DAG — cryptographic dataset identity
   (every file hashed; counts are never identity, 12 §6.2), strict YAML
   config subset (`yamlmini`), machine-local paths at `$MLFORGE_HOME`
   vs portable `datasets.yaml` (12 §6.3), `ingestion.yaml` DAG
   (sources/depends_on, cycle detection), GLOBALLY-OPEN but gated
    transform registry (`register_transform` — only registered code
    runs, unknown ⇒ BLOCK) with code-hash identity + four-field cache
    key (12 §6.4), built-in transforms `coco_detection`,
    `text_corpus` (PDF/DOCX/MD/TXT/zip → extract → paragraph-chunked
   text records; a source yielding nothing BLOCKs, never fabricates) —
   grown to all 8 registered transforms (see *Transforms*),
   `dataset add|list|verify` (tamper → REJECTED, `--force`
   reregister), `prepare` (resolve → cache → transform → derived
   `<model>_prepared` REGISTERED→VERIFIED→PREPARED, `--command-id`
   dedupe), and the gate's step-4 builtin dataset provider
9. ✅ Execution planner — measured capabilities (`nvidia-smi` query,
   never VRAM tables; absent ⇒ determinate CPU-only set, broken ⇒
   BLOCK; unknown arch ⇒ fp32 only, 12 §13.1), feasibility solver
   (micro × accum × world == frozen global_batch s.t. VRAM/precision/
   model-min, deterministic pick: max world then max micro, 12 §13.2),
   `ExecutionPlan` artifact persisted on validation pass (identity-
   gated — unchanged plan never journals twice), gate builtins for
   steps 10–13 (`plan`/`global_batch`/`precision`/`topology`), disk
   estimate for step 16 now includes registered dataset bytes,
   execution segments (`segments/segment_<NNNN>.json` — hardware
   context recorded by the worker only AFTER preflight passes,
   12 §12.2), and `SHOW PLAN` in `train` before confirmation (13 §6.1)
10. ✅ Resume/retrain/finetune flows + lineage DAG — `lineage.json`
    written once per run (train = null parents; fork/retrain/finetune =
    parent set; immutable edges, cycle/missing-parent guards), `fork`
    (semantic change → NEW run, parent's spec never mutates; no-change
    fork BLOCKs), `retrain` (sources the LATEST run of that model,
    fresh init, SHOW DELTA → confirm), `finetune` (resolve
    `model://name:vN` → base weights recorded in lineage, strategy
    menu, NEW run + NEW model, base → `USED_AS_FINE_TUNE_BASE` and
    never mutated), completion publishes the model (`name:vN`,
    provenance, artifact = newest COMMIT marker) with per-name version
    bumps, `model list|inspect` (+ lineage chain), allowlisted
    `--set`/`--config` overrides (typo ⇒ BLOCK), `--command-id` dedupe, `inspect <run>` carries lineage,
    monotonic ULIDs (same-ms ids sort by creation — retrain source is
    deterministic)
11. ✅ Evaluate/compare/infer/export/package + `model import` —
    REAL end to end: five-component evaluation identity
    (model+dataset+code+environment+protocol, 12 §15.4) with SHOW
    PROTOCOL preview that equals the recorded artifact, write-once
    evaluation/export/bundle artifacts, first-consumption model state
    marker, operator/format validation (unknown format or unsupported
    operator ⇒ BLOCK with the closed registry, never a partial export),
    contract reuse + runtime retargeting, package (contract required,
    secrets scan, provenance, no state change), input contract check
    SAFE|BLOCK for `infer` (missing input ⇒ exit 2), descriptive
    `compare` (NOT_COMPARABLE = different protocol, shown never
    averaged, never a winner), `--command-id` dedupe (original result,
    never recomputed), `model import` (declared identity, hashed
    weights copied into the registry with the artifact hash verified,
    duplicate `name:vN` BLOCKs, origin `import`), `validate
    <RUN|MODEL>`, `serve` honest exit 4. **Engines are REAL by
    default** — one `ops/engines` registry behind the SAME
    `MLFORGE_HARNESS=1` gate as `ScaffoldTrainer` (12 §12.4) with five
    family engines (text `reasoner_s` → perplexity; reid `osnet_x1_0`
    → rank1/rank5/mAP; detection `rf_detr_{s,l,seg_s}` → mAP/AP50 via
    COCOeval; calibrator → ece/nll/mce/brier + per-α coverage and
    average set size (`coverage_alpha_05`, `avg_set_size_alpha_10`, … —
    one pair per fitted conformal alpha, 06 §5); `hypothesis_ranker` →
    ndcg@3/ndcg@5 reusing the trainer's own metric): evaluate computes
    the family's metrics over store-backed prepared records with split
    discipline (train/holdout rows are never eval rows, 12 §15.4) and
    transform guards; infer executes one-shot text|image|structured
    inputs over the model's committed weights (newest VALID checkpoint
    COMMIT marker cross-checked against the registry's artifact_hash —
    never random init, 12 §15.3) with identity bound to contract+input
    +model hash; export writes genuine ONNX bytes (ir_version 8,
    opset 17, domain-correct ai.onnx.ml trees for the ranker) gated on
    `onnx.checker` + an onnxruntime numerical round-trip PASS or
    ValidationBlock — no partial export ever (13 §6.9/§7); every
    artifact carries its family harness label, and unknown
    family/unsupported format/missing weights refuse honestly naming
    what is missing; the labeled scaffold survives only behind
    `MLFORGE_HARNESS=1` (12 §12.4).
12. ✅ TUI/GUI over the same Workflow API — `watch` without RUN is the
    multi-screen Live dashboard (overview → run §9.6 frame → events
    §9.7 feed → help; j/k/Enter/e/b/p/s/c/q keys), `watch RUN` keeps the
    single-run frame, `--json` returns the dashboard model; shared
    read-only view model (`mlforge.ui.viewmodel`) feeds both UIs; pure
    `handle_key` state machine + pure `route()` make the whole layer
    testable without a TTY or a socket; `gui` = localhost (127.0.0.1)
    stdlib web dashboard with the §9.6 frame, events, and pause/stop/
    checkpoint INTENT buttons (`requested_by=cli:gui`), 404/400 fail
    closed; every viewer is disposable — closing it never touches a run

## Invariants already enforced in code

* **No scaffold ships as a deliverable** — harness code (fake loss,
  fake metrics, fake exported bytes) is labeled `harness: scaffold` in
  its own output and production refuses it by default (`train`/launch
  exit 3 before anything is created; the worker refuses before
  READY → RUNNING, leaving the run untouched). Explicit opt-ins only:
  `runtime.trainer=harness-scaffold` / `MLFORGE_HARNESS=1`.
  Correction: build steps 6 and 11 were once wrongly checked ✅ while
  their engines were scaffolds — fixed here, never again.

* **FAILED has one meaning** — a recorded `failure.recovery` disposition
  (`RESUME` | `FORK_ONLY`); resume on `FORK_ONLY` exits 3
  (`NO_VALID_CONTINUATION`), never silently restarts (13 §5.3).
* **Automatic continuation is NO** for every non-`RUNNING` state; the system
  never transitions a run on its own.
* **status.json is a projection** — delete it and the state rebuilds from the
  event journal (12 §16.1).
* **run_spec is immutable** — written once (atomic rename); overwrite attempt
  → `VALIDATION_BLOCK`. Semantic change = fork = new run.
* **Crash ≠ pause** — crash from any live state → `INTERRUPTED`;
  reconciliation never starts training (12 §12.3).
* **Illegal transitions change nothing** and list the legal actions (exit 3).
* **GC is reachability-only** — never age/size; run-folder `refs.json` are
  the roots (portable), grace protects fresh orphans, active run leases
  block execution-mode GC outright (12 §6.1).
* **Blobs are atomic and verified** — staging → fsync → rename; identity
  claimed at `put_file` must match (mismatch → BLOCK); corruption detected
  by `verify()`, never trusted.
* **Status is read-only** — it never writes state or transitions a run
  (13 §4.1); a stale RUNNING heartbeat is a WARNING, reconciled only by
  `resume` / supervisor crash detection. Never infer lifecycle from
  telemetry (13 §9.4).
* **Registry registration is transactional** — entry written to temp,
  renamed to commit; crash orphans swept by `recover()`.
* **Unverifiable == failed verification** — a missing provider, a throwing
  check, or a missing probe records FAIL, never SKIP (12 §7.1: "probably
  compatible does not exist"). First FAIL stops the run, so the
  side-effect lease step (15) never executes after an earlier failure.
* **Gate BLOCK has two distinct behaviors** — train path:
  CREATED → VALIDATING → FAILED[FORK_ONLY], never READY; resume path:
  **no state change** ("No changes were made" — 13 §7 failure matrix:
  Stays PAUSED / stays FAILED), journaled as `validation_blocked`.
* **EXACT continuation is opt-in, PORTABLE is the default** — EXACT only
  when every check reports `exact_compatible: True` (12 §8).
* **Disk-space check is worst-case and mandatory** —
  `2×checkpoint + dataset_cache + logs + safety_margin`; free < required
  ⇒ BLOCK, never "train until disk fills" (12 §7.3).
* **Preflight is report-only until the runtime exists** — it never moves
  a run to RUNNING (only after PREFLIGHT PASSED does training proceed).
* **Single-writer run lease** — `runs/<id>/.lease` acquired atomically
  (tmp → fsync → link); second resume → exit 3 `RUN_ALREADY_EXECUTING`;
  stale heartbeat → **SUSPECT, never auto-free** (stale ≠ free); break
  requires `--force --yes` and always journals `LEASE_BROKEN` (12 §23).
* **Command idempotency** — duplicate `command_id` of a success returns
  the original result (never a second run) and journals
  `COMMAND_DEDUPED`; in flight → exit 3; failures are retryable (13 §4.4).
* **Supervisor detects crashes, never continues runs** — expired/missing
  heartbeat → INTERRUPTED; it never resumes, reconciles, or starts
  anything (13 §1 automatic continuation = NO).
* **Dataset identity is cryptographic** — every file's sha256 (sorted,
  POSIX-relative paths only; absolute paths never enter identity); sample
  counts are never identity. Symlinks and empty trees are refused —
  an identity we cannot walk deterministically is a guess (12 §6.2–§6.3).
* **Paths are configuration, never identity** — dataset paths live in
  `$MLFORGE_HOME` (default `~/.mlforge`), verified against registration
  on every gate/preflight/prepare; wrong path = mismatch, never a
  discovery scan (13 §10).
* **Transforms run only from a closed registry** — unknown transform
  ⇒ exit 1; code-hash + config-hash + input identities + env fingerprint
  form the cache key (12 §6.4); a transform that skips a source or
  emits 0 records is BLOCKed, never published.
* **Re-registration is explicit** — changed content requires
  `dataset add --force` (journaled `dataset_reregistered`); a tampered
  verified dataset becomes REJECTED, never silently reinterpreted
  (13 §5.2).
* **global_batch is input AND output** — the solver may only touch
  execution fields (micro/accum/world); `micro × accum × world` must
  equal the frozen semantic `global_batch` or gate step 11 FAILs
  (12 §8.3, §13.3: execution → semantic mutation is forbidden).
* **Capabilities are measured, never assumed** — VRAM comes from the
  device query; unknown compute capability supports nothing beyond
  fp32 (fail-closed); `nvidia-smi` present but broken ⇒ BLOCK, absent
  ⇒ a determinate CPU-only set (12 §13.1, §21).
* **Precision fallback and CPU runs are PORTABLE, never EXACT** —
  bf16→fp32 fallback or a world_size of 1 CPU records
  `exact_compatible: False` (12 §14, §21).
* **An infeasible plan BLOCKs with migration options** — "run on
  another machine / fork with a different global_batch / cancel"
  (13 §7), never a silent shrink of the batch; execution segments are
  created only after preflight passes, one per start (12 §12.2).
* **lineage.json is written once and never rewritten** — edges are
  immutable after run creation; a differing rewrite BLOCKs, a child run
  gets its own file (12 §15.2). The chain walks root-first with cycle
  and missing-parent guards.
* **Semantic change = NEW run with a parent** — `fork`/`retrain`/
  `finetune` all create runs; the parent's `run_spec.json` is never
  touched (frozen identity), and a fork that changes nothing BLOCKs
  (resume continues, retrain repeats — 12 §4, 13 §6.3).
* **An evaluation identity has exactly five components** — model hash,
  re-hashed dataset identity, code hash, environment fingerprint, and
  the evaluation protocol hash (metric definitions + harness code hash
  are derived, never user-overridable — faking them would fake
  comparability). SHOW PROTOCOL shows exactly what gets recorded; a
  different protocol is a DIFFERENT artifact, never an update
  (12 §15.4, 13 §6.7).
* **Comparison is descriptive, never a verdict** — protocol-hash
  equality decides `comparable`; a different protocol renders
  `NOT_COMPARABLE` and the rows are shown side by side but never
  averaged; comparison never picks a winner (13 §6.10).
* **The model's consumption marker fires once** — AVAILABLE fans out to
  exactly one of EVALUATED/EXPORTED/DEPLOYED/USED_AS_FINE_TUNE_BASE;
  later consumptions (second evaluation, export after evaluate) write
  their artifact plus a no-`to` journal event and never flip state back
  (13 §5.4).
* **No partial exports, no invented contracts** — an unknown format or
  an unsupported operator BLOCKs before anything is written, listing the
  closed registry / the offending operators and the formats that can
  host the graph; `package` and `infer` BLOCK without a contract, and an
  existing contract is reused (operators preserved) with only the
  runtime retargeted (13 §6.9, 12 §15.3).
* **Secrets never enter artifacts** — export and package scan the model,
  run, and contract sources; any finding BLOCKs (12 §16, 13 §7).
* **Every UI is a disposable viewer over ONE Workflow API** — CLI, TUI,
  and GUI share `mlforge.ui.viewmodel` reads and write control INTENTS
  through the same channel (`requested_by` is the only difference);
  no screen, page, or key ever transitions a run (13 §1 rules 2–3).
* **STATUS DOWN → TRAINING CONTINUES** — closing the dashboard, killing
  `gui`, or dropping the terminal changes nothing about any run; a run
  that vanishes under the viewer's cursor drops back to the overview
  with a notice, reconstructed from persistence (13 §9.1, §9.9).
* **The GUI binds localhost only and fails closed** — unknown run ⇒ 404,
  unknown/mis-scoped action ⇒ 400/405, bad `--port` ⇒ exit 1, port in
  use ⇒ exit 3; pages HTML-escape everything (13 §9.2 local control
  plane, no remote surface in v1).
* **Run completion publishes its model** — `COMPLETED` ⇒ registry
  entry CREATED → VALIDATED → AVAILABLE with provenance (`name:vN`,
  run, spec hash, artifact = newest COMMIT marker or null) and a
  per-name version bump; weights alone are not the model (13 §5.5,
  12 §15.3).
* **Fine-tuning never mutates the base** — NEW run + NEW model; the
  base only records `USED_AS_FINE_TUNE_BASE` (state, not data), and its
  edges live in the CHILD's lineage (13 §6.4).
* **Override keys are allowlisted** — `--set`/`--config` semantic keys
  accept only identity fields (plus `lr`/`batch`/`precision` aliases);
  a typo BLOCKs before anything exists, never a silent different
  experiment (12 §13.3).
* **Every run records its identity at creation** — source tree hash +
  full `code/source.snapshot.tar.gz` + git provenance and the
  environment fingerprint are written before the run is journaled; a
  run that cannot record identity does not exist (12 §6.4). The gate
  then freezes them: tree changed, snapshot tampered, or environment
  moved ⇒ FAIL (12 §13.3).
* **One disk/GPU story across gate and preflight** — step 16 and the
  preflight disk probe read the same `disk_estimate_components()`
  (runtime overrides from `train --config` honored in both), and the
  preflight GPU expectation follows the validated execution plan
  (CPU/PORTABLE plan never demands `nvidia-smi`; explicit
  `runtime.gpu` still wins — 12 §14).
* **`validate` runs the full 16-step gate, lease included** — a
  passing gate holds the single-writer lease (launch reuses that
  session token), so a second validate while the lease is held is the
  spec'd exit-3 `RUN_ALREADY_EXECUTING` (12 §18 step 15, 13 §7);
  a BLOCKED gate releases whatever it acquired ("no changes" includes
  the lease).

## Develop

```bash
cd mlforge
pip install -e ".[dev]"
pytest
mlforge --version
mlforge --root /path/to/workspace status
```
