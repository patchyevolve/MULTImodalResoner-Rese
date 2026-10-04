"""Torch text trainer — the REAL learning loop (build step 6's missing
half: 12 §11.4 checkpoint contents, §12.4 injected trainer).

Trains a byte-level transformer LM (next-byte prediction) on prepared
text datasets (`text_corpus` transform). Real gradients, real optimizer
state, real LR schedule — every component of REQUIRED_COMPONENTS carries
the actual framework state (or an honest `{"enabled": false}` where the
capability is not integrated: AMP, EMA, early stopping, distributed).

Determinism (Trainer protocol): everything data-order related derives
from (seed, epoch) via a seeded `torch.randperm` — recovery replays the
same order; model/optimizer/RNG come from the checkpoint payload.

Fail-closed rules enforced here:
  * unknown model / missing torch          → PreconditionFailed (resolve)
  * optimizer/scheduler/loss not supported → ValidationBlock (semantic
    identity is honored literally — never silently "close enough")
  * precision other than fp32              → PreconditionFailed (no AMP
    integrated; amp_scaler payload says enabled:false)
  * world_size > 1                         → PreconditionFailed (no
    distributed training integrated)
  * corpus smaller than one sequence       → ValidationBlock
"""

from __future__ import annotations

import io
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS
from mlforge.runtime.trainer import StepResult, TrainState
from mlforge.trainers import cpu_tree, resolve_device

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # ImportError (or broken wheel) — honest refusal
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


def torch_available() -> bool:
    return _TORCH_IMPORT_ERROR is None


def dependency_error() -> str | None:
    """None = this trainer's framework is present here (registry probe)."""
    if torch_available():
        return None
    return (
        "torch is not installed — the real learning loop needs it; "
        "pip install torch (CPU wheel: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu)"
    )


@dataclass(frozen=True)
class TextArch:
    """Architecture constants keyed by MODEL NAME (part of the run's
    semantic identity — `reasoner_s` always means exactly this)."""

    n_layer: int
    n_head: int
    d_model: int
    block_size: int
    dropout: float = 0.0


#: Trainable text models (add a row to onboard another text model).
MODEL_REGISTRY: dict[str, TextArch] = {
    "reasoner_s": TextArch(n_layer=4, n_head=6, d_model=192, block_size=256),
}

#: Byte-level vocabulary: every document maps to tokens with NO UNK.
VOCAB_SIZE = 256

#: Semantic values this trainer honors — anything else BLOCKs (the
#: frozen semantic identity must mean what it says).
_LOSSES = {"byte_cross_entropy", "cross_entropy", "next_token"}
_OPTIMIZERS = {"adamw", "adam", "sgd"}
_SCHEDULERS = {"cosine", "linear", "constant"}

if nn is not None:

    class _Block(nn.Module):
        def __init__(self, arch: TextArch) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(arch.d_model)
            self.attn = nn.MultiheadAttention(
                arch.d_model, arch.n_head, batch_first=True,
                dropout=arch.dropout,
            )
            self.ln2 = nn.LayerNorm(arch.d_model)
            hidden = 4 * arch.d_model
            self.mlp = nn.Sequential(
                nn.Linear(arch.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, arch.d_model),
                nn.Dropout(arch.dropout),
            )

        def forward(self, x: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
            h = self.ln1(x)
            attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + attn
            return x + self.mlp(self.ln2(x))

    class TinyGPT(nn.Module):
        """Minimal pre-LN transformer LM — real gradients end to end."""

        def __init__(self, arch: TextArch) -> None:
            super().__init__()
            self.arch = arch
            self.tok = nn.Embedding(VOCAB_SIZE, arch.d_model)
            self.pos = nn.Embedding(arch.block_size, arch.d_model)
            self.blocks = nn.ModuleList(
                [_Block(arch) for _ in range(arch.n_layer)]
            )
            self.ln_f = nn.LayerNorm(arch.d_model)
            self.head = nn.Linear(arch.d_model, VOCAB_SIZE, bias=False)
            self.head.weight = self.tok.weight  # weight tying
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        def forward(self, idx: torch.Tensor,
                    targets: torch.Tensor | None = None):
            b, t = idx.shape
            pos_ids = torch.arange(t, device=idx.device)
            x = self.tok(idx) + self.pos(pos_ids)[None, :, :]
            causal = torch.triu(
                torch.ones(t, t, device=idx.device, dtype=torch.bool), 1
            )
            for block in self.blocks:
                x = block(x, causal)
            x = self.ln_f(x)
            logits = self.head(x)
            if targets is None:
                return logits, None
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
            )
            return logits, loss

else:  # torch missing — MODEL_REGISTRY still documents what WOULD train
    TinyGPT = None  # type: ignore[assignment,misc]


def load_text_corpus(root: Path, train_datasets: Sequence[str]) -> str:
    """Prepared store artifact(s) → one corpus string (fail-closed).

    Mirrors gate step 4 semantics locally: registered, version-exact,
    store-backed, verified bytes — the trainer never reads a raw path."""
    from mlforge.ingest.identity import parse_ref
    from mlforge.store import ContentStore

    if not train_datasets:
        raise ValidationBlock("run_spec.train_datasets is empty")
    store = ContentStore(root / "store")
    texts: list[str] = []
    for ref in train_datasets:
        name, version = parse_ref(ref)
        reg_path = root / "datasets" / name / "identity.json"
        if not reg_path.is_file():
            raise PreconditionFailed(
                f"dataset {name!r} not registered — mlforge dataset add "
                f"{name} <PATH>",
            )
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        reg_version = reg.get("version")
        if reg_version and str(reg_version) != version:
            raise ValidationBlock(
                f"{ref}: registered version {reg_version!r} != {version!r} "
                "(versions are identity, never reinterpreted)",
            )
        identity = str(reg.get("identity") or "")
        if not identity or not store.contains(identity):
            raise PreconditionFailed(
                f"{ref} is not a prepared store artifact — "
                "run `mlforge prepare <model>` first",
            )
        doc = json.loads(store.get_bytes(identity, verify=True).decode("utf-8"))
        for rec in doc.get("records") or []:
            text = rec.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    if not texts:
        raise ValidationBlock(
            "prepared dataset(s) contain no text records — prepare with "
            "transform `text_corpus` (other transforms produce other "
            "record types, never text)",
        )
    return "\n\n".join(texts)


class TorchTextTrainer:
    """Byte-level LM trainer — the real thing (12 §11.4 state owner)."""

    def __init__(
        self,
        *,
        model_name: str,
        semantic: Mapping[str, Any],
        runtime: Mapping[str, Any],
        train_datasets: Sequence[str],
        root: Path,
        payloads: Mapping[str, bytes] | None = None,
        start_step: int = 0,
        start_epoch: int = 0,
        plan: Any = None,
        run_dir: Path | None = None,        # scratch dir (unused here)
        on_progress: Any = None,            # fast steps: not needed
        init_weights: bytes | None = None,
    ):
        if not torch_available():
            raise PreconditionFailed(
                "torch is not installed — the real learning loop needs it",
                hint="pip install torch --index-url "
                     "https://download.pytorch.org/whl/cpu (CPU wheels)",
            ) from _TORCH_IMPORT_ERROR
        # -- placement (runtime.device; never hard-wired, fail-closed) ----
        self.device = resolve_device(runtime)
        arch = MODEL_REGISTRY.get(model_name)
        if arch is None:
            raise PreconditionFailed(
                f"model {model_name!r} has no registered text architecture",
                hint=f"trainable here: {', '.join(sorted(MODEL_REGISTRY))}",
            )
        # -- semantic identity, honored literally -------------------------
        loss_name = str(semantic.get("loss", "")).strip().lower()
        if loss_name not in _LOSSES:
            raise ValidationBlock(
                f"semantic.loss {loss_name!r} is not implemented by the "
                f"text trainer (supports: {', '.join(sorted(_LOSSES))})",
                hint="semantic identity is frozen — set the loss the "
                     "trainer actually computes, never a silent substitute",
            )
        opt_name = str(semantic.get("optimizer", "")).strip().lower()
        if opt_name not in _OPTIMIZERS:
            raise ValidationBlock(
                f"semantic.optimizer {opt_name!r} not supported "
                f"(supports: {', '.join(sorted(_OPTIMIZERS))})",
            )
        sched_name = str(semantic.get("scheduler", "cosine")).strip().lower()
        if sched_name not in _SCHEDULERS:
            raise ValidationBlock(
                f"semantic.scheduler {sched_name!r} not supported "
                f"(supports: {', '.join(sorted(_SCHEDULERS))})",
            )
        try:
            self.epochs = int(semantic["epochs"])
            self.seed = int(semantic["seed"])
            self.global_batch = int(semantic["global_batch"])
            self.learning_rate = float(semantic["learning_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationBlock(
                f"semantic field unreadable for training: {exc}"
            ) from exc
        if self.epochs < 1 or self.global_batch < 1 or self.learning_rate <= 0:
            raise ValidationBlock(
                "semantic requires epochs >= 1, global_batch >= 1, "
                f"learning_rate > 0 (got epochs={self.epochs}, "
                f"global_batch={self.global_batch}, lr={self.learning_rate})",
            )

        # -- execution plan (micro/accum must reproduce global_batch) ------
        if plan is not None:
            world = int(getattr(plan, "world_size", 1))
            micro = int(getattr(plan, "micro_batch", self.global_batch))
            accum = int(getattr(plan, "grad_accum", 1))
            precision = str(getattr(plan, "precision_effective", "fp32"))
            if micro * accum * world != self.global_batch:
                raise ValidationBlock(
                    f"execution plan breaks the global-batch invariant: "
                    f"micro {micro} × accum {accum} × world {world} != "
                    f"global_batch {self.global_batch}",
                )
            if world != 1:
                raise PreconditionFailed(
                    f"plan wants world_size {world} — distributed training "
                    "is not integrated in this trainer",
                    hint="run single-process (topology world_size 1)",
                )
            if precision != "fp32":
                raise PreconditionFailed(
                    f"plan precision {precision!r} — this trainer has no "
                    "AMP integrated (amp_scaler component would be a lie)",
                    hint="semantic.precision_policy: fp32",
                )
        else:
            micro = min(4, self.global_batch)
            accum = (self.global_batch // micro
                     if self.global_batch % micro == 0 else 1)
            if micro * accum != self.global_batch:
                micro, accum = self.global_batch, 1
        self.micro = micro
        self.accum = accum
        self.scheduler_kind = sched_name
        self.model_name = model_name

        # -- data ----------------------------------------------------------
        self.block = arch.block_size
        corpus_text = load_text_corpus(root, train_datasets)
        corpus = corpus_text.encode("utf-8")
        if len(corpus) < self.block + 1:
            raise ValidationBlock(
                f"corpus too small: {len(corpus)} bytes < one sequence "
                f"({self.block} + 1) — prepare more text or a smaller model",
            )
        self.corpus = torch.frombuffer(
            bytearray(corpus), dtype=torch.uint8
        ).long()
        self.n_windows = (len(corpus) - 1 - self.block) // self.block + 1
        self.steps_per_epoch = max(1, self.n_windows // self.global_batch)
        # optional semantic knob: cap windows drawn per epoch (documented
        # in the dataloader_state payload — part of the run's semantics).
        wpe = int(semantic.get("windows_per_epoch", 0) or 0)
        if wpe > 0:
            steps_cap = max(1, wpe // self.global_batch)
            self.steps_per_epoch = min(self.steps_per_epoch, steps_cap)

        # -- model / optimizer / schedule ----------------------------------
        # Weights init on CPU under the seeded RNG (bit-identical init on
        # every host), then placed on the resolved device.
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.model = TinyGPT(arch).to(self.device)
        params = self.model.parameters()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, lr=self.learning_rate, betas=(0.9, 0.95),
                weight_decay=0.1,
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                params, lr=self.learning_rate, betas=(0.9, 0.95),
            )
        else:
            self.optimizer = torch.optim.SGD(
                params, lr=self.learning_rate, momentum=0.9,
            )
        total_steps = self.epochs * self.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._lr_lambda(total_steps)
        )

        # -- restore (real continuation: weights + optimizer + RNG) --------
        if payloads and init_weights is not None:
            raise ValidationBlock(
                "resume payloads and fine-tune init weights are mutually "
                "exclusive — resume from THIS run's checkpoint or "
                "initialize from the parent's weights, never both",
            )
        if payloads:
            missing = [c for c in REQUIRED_COMPONENTS if c not in payloads]
            if missing:
                raise ValidationBlock(
                    f"checkpoint payload incomplete — missing: "
                    f"{', '.join(missing)} (not resumable, 12 §11.4)",
                )
            self._restore(dict(payloads))
        elif init_weights is not None:
            self._init_from_weights(init_weights)
        self.start_step = int(start_step)
        self.start_epoch = int(start_epoch)

    # -- schedule ----------------------------------------------------------

    def _lr_lambda(self, total: int):
        warmup = max(1, int(total * 0.05))
        kind = self.scheduler_kind

        def fn(step: int) -> float:
            if kind == "constant":
                return 1.0
            if step < warmup:
                return float(step + 1) / float(warmup)
            if kind == "linear":
                return max(
                    0.0,
                    float(total - step) / float(max(1, total - warmup)),
                )
            progress = min(
                1.0, float(step - warmup) / float(max(1, total - warmup))
            )
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return fn

    # -- restore -----------------------------------------------------------

    @staticmethod
    def _load(blob: bytes) -> Any:
        # bytes come from CheckpointStore.load/verify — integrity-checked
        # against the manifest before they reach the trainer.
        return torch.load(io.BytesIO(blob), weights_only=True, map_location="cpu")

    def _init_from_weights(self, blob: bytes) -> None:
        """Fine-tune (13 §6.4): the parent run's `model` component loads
        into a FRESH optimizer/scheduler — weights only, never a resume."""
        try:
            self.model.load_state_dict(self._load(blob))
        except Exception as exc:
            raise ValidationBlock(
                f"fine-tune init weights not loadable by trainer "
                f"{self.model_name!r}: {exc}",
                hint="the parent model must be the same architecture — "
                     "fine-tuning across architectures is never implied",
            ) from exc

    def _restore(self, payloads: dict[str, bytes]) -> None:
        try:
            self.model.load_state_dict(self._load(payloads["model"]))
            self.optimizer.load_state_dict(self._load(payloads["optimizer"]))
            sched_state = json.loads(
                payloads["lr_scheduler"].decode("utf-8")
            )
            if str(sched_state.get("kind")) != self.scheduler_kind:
                raise ValidationBlock(
                    f"checkpoint scheduler {sched_state.get('kind')!r} != "
                    f"semantic.scheduler {self.scheduler_kind!r}",
                )
            self.scheduler.last_epoch = int(sched_state["last_epoch"])
            self.scheduler._last_lr = [
                float(v) for v in sched_state["_last_lr"]
            ]
            rng = self._load(payloads["rng_hierarchy"])
            random.setstate(rng["python"])
            torch.set_rng_state(
                torch.tensor(rng["torch_cpu"], dtype=torch.uint8)
                if not isinstance(rng["torch_cpu"], torch.Tensor)
                else rng["torch_cpu"]
            )
        except ValidationBlock:
            raise
        except Exception as exc:
            raise ValidationBlock(
                f"checkpoint payload not loadable by trainer "
                f"{self.model_name!r}: {exc}",
                hint="a harness checkpoint cannot resume under a real "
                     "trainer (and vice versa) — state is never faked "
                     "into compatibility",
            ) from exc

    # -- data order ---------------------------------------------------------

    def _epoch_order(self, epoch: int) -> torch.Tensor:
        """Deterministic per-epoch window order (recorded protocol)."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed((self.seed * 1_000_003 + epoch) % (2 ** 63))
        base = torch.randperm(self.n_windows, generator=gen)
        need = self.steps_per_epoch * self.global_batch
        if base.numel() < need:
            reps = (need + base.numel() - 1) // base.numel()
            base = base.repeat(reps)
        return base[:need]

    def _gather(self, windows: torch.Tensor):
        starts = windows * self.block
        offsets = torch.arange(self.block)
        idx = starts[:, None] + offsets[None, :]
        x = self.corpus[idx]          # windows live on CPU (the corpus)
        y = self.corpus[idx + 1]
        return x.to(self.device), y.to(self.device)

    # -- protocol -----------------------------------------------------------

    def step(self, state: TrainState) -> StepResult:
        if state.epoch >= self.epochs:  # counters say: already finished
            return StepResult(
                state.global_step, state.epoch, loss=None, done=True,
            )
        pos = state.global_step - state.epoch * self.steps_per_epoch
        if pos >= self.steps_per_epoch:  # defensive: boundary without bump
            nxt_epoch = state.epoch + 1
            return StepResult(
                state.global_step, nxt_epoch, loss=None,
                done=nxt_epoch >= self.epochs,
            )

        order = self._epoch_order(state.epoch)
        batch = order[pos * self.global_batch:(pos + 1) * self.global_batch]
        micro = self.micro
        n_micro = (batch.numel() + micro - 1) // micro

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for i in range(n_micro):
            part = batch[i * micro:(i + 1) * micro]
            if part.numel() == 0:
                break
            x, y = self._gather(part)
            _, loss = self.model(x, y)
            if not torch.isfinite(loss):
                raise ValidationBlock(
                    f"loss diverged at step {state.global_step} "
                    f"({float(loss)}) — refusing to poison checkpoints",
                )
            (loss / n_micro).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        loss_val = sum(losses) / len(losses)
        nxt_step = state.global_step + 1
        nxt_epoch = state.epoch
        if nxt_step % self.steps_per_epoch == 0:
            nxt_epoch += 1
        lr = float(self.scheduler.get_last_lr()[0])
        return StepResult(
            global_step=nxt_step,
            epoch=nxt_epoch,
            loss=round(loss_val, 6),
            metrics={
                "lr": round(lr, 8),
                "ppl": round(math.exp(min(loss_val, 20.0)), 3),
            },
            done=nxt_epoch >= self.epochs,
        )

    def checkpoint_payload(self, state: TrainState) -> dict[str, bytes]:
        """§11.4: every REQUIRED_COMPONENTS entry with REAL state (or an
        honest `enabled: false` for capabilities not integrated)."""
        def _save(obj: Any) -> bytes:
            buf = io.BytesIO()
            torch.save(obj, buf)
            return buf.getvalue()

        def _json(obj: Any) -> bytes:
            return json.dumps(obj, sort_keys=True).encode("utf-8")

        pos = state.global_step - state.epoch * self.steps_per_epoch
        return {
            "model": _save(cpu_tree(self.model.state_dict())),
            "optimizer": _save(cpu_tree(self.optimizer.state_dict())),
            "lr_scheduler": _json({
                "kind": self.scheduler_kind,
                "last_epoch": int(self.scheduler.last_epoch),
                "_last_lr": [float(v) for v in
                             self.scheduler.get_last_lr()],
            }),
            "amp_scaler": _json({"enabled": False}),
            "ema": _json({"enabled": False}),
            "global_step": _json(int(state.global_step)),
            "epoch": _json(int(state.epoch)),
            "batch_position": _json({
                "position_in_epoch": int(pos),
                "steps_per_epoch": int(self.steps_per_epoch),
            }),
            "sampler_state": _json({
                "protocol": "torch.randperm(seed*1000003 + epoch)",
                "seed": int(self.seed),
                "epoch": int(state.epoch),
            }),
            "dataloader_state": _json({
                "corpus_bytes": int(self.corpus.numel()),
                "windows": int(self.n_windows),
                "block_size": int(self.block),
                "stride": int(self.block),
                "global_batch": int(self.global_batch),
                "steps_per_epoch": int(self.steps_per_epoch),
            }),
            "distributed_state": _json({"world_size": 1}),
            "grad_accum_state": _json({
                "micro_batch": int(self.micro),
                "grad_accum": int(self.accum),
                "micro_in_step": 0,  # checkpoints land on optimizer steps
            }),
            "early_stopping_state": _json({
                "enabled": False,
                "reason": "no validation loop integrated",
            }),
            "best_model_state": _json({
                "enabled": False,
                "metric": None,
            }),
            "rng_hierarchy": _save({
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
            }),
        }


#: Registry hook consumed by mlforge.trainers.build_trainer.
TRAINER_CLASS = TorchTextTrainer


__all__ = [
    "MODEL_REGISTRY",
    "TRAINER_CLASS",
    "TextArch",
    "TinyGPT",
    "TorchTextTrainer",
    "dependency_error",
    "load_text_corpus",
    "torch_available",
]
