"""Torch re-ID trainer — the REAL learning loop for OSNet (11_model_weights
§M5/§C2; 12 §10.2 `osnet` + `reid_crops`).

Trains OSNet x1_0 (Zhou et al., ICCV 2019 — "Omni-Scale Feature Learning
for Person Re-Identification", MIT) on prepared `reid_crops` datasets
(Market1501-style identity/camera manifest). The architecture below is a
faithful reimplementation of the official definition (torchreid `osnet.py`)
so the ImageNet-pretrained M5 weights (2.19M backbone params) load
key-for-key — verified by the test suite, never assumed.

Training semantics (frozen into checkpoint components, visible in
`dataloader_state`):
  * input 256x128 (HxW), bilinear resize, horizontal-flip p=0.5,
    ImageNet normalization
  * identity labels remapped to contiguous ids by sorted pid (recorded)
  * loss: cross-entropy over identities; shuffle order derives from
    (seed, epoch) — recovery replays the same order (12 §11.4)

Honest limits (fail-closed, never silently substituted):
  * losses other than cross_entropy      → ValidationBlock (no triplet/
    arcface integrated — `amp_scaler`/`ema` payloads say enabled:false)
  * plan world_size > 1                  → PreconditionFailed
  * plan precision other than fp32       → PreconditionFailed (no AMP
    integrated here — the payload would be a lie)
  * fewer than 2 identities in the data  → ValidationBlock
  * missing image file                   → ValidationBlock (paths are
    explicit; the gate re-verifies dataset identity at preflight)

Pretrained base (M5, file 11 §M5 "auto-download"): semantic
`pretrained` defaults to the ImageNet weights, cached under
`<project>/.cache/weights/` — `pretrained: false` trains from scratch
(that is what the unit tests use; never a hidden download).
"""

from __future__ import annotations

import io
import json
import math
import random
import urllib.request
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

try:
    from PIL import Image

    _PIL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover — pillow ships with torch envs
    Image = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc


def torch_available() -> bool:
    return _TORCH_IMPORT_ERROR is None


def dependency_error() -> str | None:
    """None = this trainer's framework is present here (registry probe)."""
    if not torch_available():
        return (
            "torch is not installed — the real learning loop needs it; "
            "pip install torch (CPU wheel: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu)"
        )
    if _PIL_IMPORT_ERROR is not None:
        return (
            "pillow is not installed — the re-ID trainer decodes images "
            "with it; pip install pillow"
        )
    return None


@dataclass(frozen=True)
class OsNetArch:
    """Architecture constants keyed by MODEL NAME (semantic identity)."""

    channels: tuple[int, ...] = (64, 256, 384, 512)
    layers: tuple[int, ...] = (2, 2, 2)
    feature_dim: int = 512
    input_hw: tuple[int, int] = (256, 128)  # H, W — re-ID standard


#: Trainable re-ID models (add a row to onboard another).
MODEL_REGISTRY: dict[str, OsNetArch] = {
    "osnet_x1_0": OsNetArch(),
}

#: Semantic values this trainer honors — anything else BLOCKs.
_LOSSES = {"cross_entropy"}
_OPTIMIZERS = {"adamw", "adam", "sgd"}
_SCHEDULERS = {"cosine", "linear", "constant"}

#: ImageNet statistics (standard re-ID preprocessing).
_IM_MEAN = (0.485, 0.456, 0.406)
_IM_STD = (0.229, 0.224, 0.225)

#: M5 base weights (file 11 §M5 — HF mirror of the official release).
_PRETRAIN_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_x1_0_imagenet.pth"
)


if nn is not None:

    class ConvLayer(nn.Module):
        """conv + bn + relu (official torchreid building block)."""

        def __init__(self, in_ch: int, out_ch: int, kernel: int,
                     stride: int = 1, padding: int = 0, groups: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                                  padding=padding, bias=False, groups=groups)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                                  bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1Linear(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                                  bias=False)
            self.bn = nn.BatchNorm2d(out_ch)

        def forward(self, x):
            return self.bn(self.conv(x))

    class LightConv3x3(nn.Module):
        """1x1 (linear) + depthwise 3x3 (nonlinear) + bn + relu."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1,
                                   bias=False, groups=out_ch)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv2(self.conv1(x))))

    class ChannelGate(nn.Module):
        """Channel-wise gates conditioned on the input tensor."""

        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, channels // reduction, 1,
                                 bias=True)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(channels // reduction, channels, 1,
                                 bias=True)
            self.gate_activation = nn.Sigmoid()

        def forward(self, x):
            g = self.global_avgpool(x)
            g = self.fc1(g)
            g = self.relu(g)
            g = self.fc2(g)
            g = self.gate_activation(g)
            return x * g

    class OSBlock(nn.Module):
        """Omni-scale feature learning block (official definition)."""

        def __init__(self, in_ch: int, out_ch: int,
                     bottleneck_reduction: int = 4):
            super().__init__()
            mid = out_ch // bottleneck_reduction
            self.conv1 = Conv1x1(in_ch, mid)
            self.conv2a = LightConv3x3(mid, mid)
            self.conv2b = nn.Sequential(
                LightConv3x3(mid, mid), LightConv3x3(mid, mid))
            self.conv2c = nn.Sequential(
                LightConv3x3(mid, mid), LightConv3x3(mid, mid),
                LightConv3x3(mid, mid))
            self.conv2d = nn.Sequential(
                LightConv3x3(mid, mid), LightConv3x3(mid, mid),
                LightConv3x3(mid, mid), LightConv3x3(mid, mid))
            self.gate = ChannelGate(mid)
            self.conv3 = Conv1x1Linear(mid, out_ch)
            self.downsample = (
                Conv1x1Linear(in_ch, out_ch) if in_ch != out_ch else None
            )

        def forward(self, x):
            identity = x
            x1 = self.conv1(x)
            x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
                  + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
            out = self.conv3(x2)
            if self.downsample is not None:
                identity = self.downsample(identity)
            return F.relu(out + identity)

    class OSNet(nn.Module):
        """OSNet x1_0 — feature extractor + identity classifier head.

        In eval mode `forward` returns the 512-d embedding; in train mode
        it returns classification logits (official behavior)."""

        def __init__(self, arch: OsNetArch, num_classes: int):
            super().__init__()
            ch = arch.channels
            self.conv1 = ConvLayer(3, ch[0], 7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = self._make_layer(arch.layers[0], ch[0], ch[1],
                                          reduce=True)
            self.conv3 = self._make_layer(arch.layers[1], ch[1], ch[2],
                                          reduce=True)
            self.conv4 = self._make_layer(arch.layers[2], ch[2], ch[3],
                                          reduce=False)
            self.conv5 = Conv1x1(ch[3], ch[3])
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(arch.feature_dim, arch.feature_dim),
                nn.BatchNorm1d(arch.feature_dim),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Linear(arch.feature_dim, num_classes)
            self._init_params()

        @staticmethod
        def _make_layer(layers: int, in_ch: int, out_ch: int, *,
                        reduce: bool) -> nn.Sequential:
            seq: list[nn.Module] = [OSBlock(in_ch, out_ch)]
            seq += [OSBlock(out_ch, out_ch) for _ in range(1, layers)]
            if reduce:
                seq.append(nn.Sequential(
                    Conv1x1(out_ch, out_ch), nn.AvgPool2d(2, stride=2)))
            return nn.Sequential(*seq)

        def _init_params(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def embed(self, x):
            """512-d features (backbone + fc, no classifier)."""
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            v = self.global_avgpool(x).view(x.size(0), -1)
            return self.fc(v)

        def forward(self, x):
            return self.classifier(self.embed(x))

else:  # torch missing — MODEL_REGISTRY still documents WHAT would train
    OSNet = None  # type: ignore[assignment,misc]


def load_reid_samples(
    root: Path, train_datasets: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Prepared `reid_crops` store artifact(s) → sample list + id map.

    Mirrors gate semantics locally: registered, version-exact, store-backed,
    verified bytes. Image bytes are NOT in the store (they live at the
    registered dataset path the gate re-verifies) — existence is checked
    here, full hashing stays the gate's job (documented, never implied)."""
    from mlforge.ingest import config as ingest_config
    from mlforge.ingest.identity import parse_ref
    from mlforge.store import ContentStore

    if not train_datasets:
        raise ValidationBlock("run_spec.train_datasets is empty")
    store = ContentStore(root / "store")
    rows: list[dict[str, Any]] = []
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
        schema = str((doc.get("transform") or {}).get("name") or "")
        if schema and schema != "reid_crops":
            raise ValidationBlock(
                f"{ref} was prepared with transform {schema!r} — the "
                "re-ID trainer needs `reid_crops` (other transforms produce "
                "other record types, never identity labels)",
            )
        for rec in doc.get("records") or []:
            if not isinstance(rec, dict):
                continue
            if rec.get("identity") is None or not rec.get("relative_path"):
                continue
            rows.append(rec)
    if not rows:
        raise ValidationBlock(
            "prepared dataset(s) contain no re-ID records — prepare with "
            "transform `reid_crops` (other transforms produce other record "
            "types, never identity labels)",
        )

    # train split: prefer explicit `train` records (Market1501 keeps
    # query/test alongside); flat layouts without split fields use all.
    train_rows = [r for r in rows if r.get("split") in (None, "train")]
    by_source = {str(r.get("source") or "") for r in train_rows}
    if any(str(r.get("split")) == "train" for r in rows):
        train_rows = [r for r in rows if r.get("split") == "train"]
    if not train_rows:
        raise ValidationBlock(
            "no train-split re-ID records — reid_crops expected a "
            "bounding_box_train/ (or train/) folder (02 §5)",
        )

    paths = ingest_config.load_paths(root)
    pids = sorted({str(r["identity"]) for r in train_rows},
                  key=lambda p: int(p) if str(p).lstrip("-").isdigit() else 0)
    if len(pids) < 2:
        raise ValidationBlock(
            f"need at least 2 identities to train (found {len(pids)}) — "
            "cross-entropy over one class teaches nothing",
        )
    id_map = {pid: str(i) for i, pid in enumerate(pids)}
    samples: list[dict[str, Any]] = []
    missing: list[str] = []
    for rec in train_rows:
        src = str(rec.get("source") or "")
        ds_name = src.split(":", 1)[0]
        base = paths.get(ds_name)
        if not base:
            raise PreconditionFailed(
                f"dataset {ds_name!r}: no machine-local path configured — "
                f"mlforge dataset add {ds_name} <PATH>",
            )
        path = Path(base) / str(rec["relative_path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        samples.append({
            "path": path,
            "pid": str(rec["identity"]),
            "label": int(id_map[str(rec["identity"])]),
            "camera": int(rec.get("camera") or 0),
        })
    if missing:
        raise ValidationBlock(
            f"{len(missing)} re-ID image(s) missing at the registered path "
            f"(first: {missing[0]}) — the dataset content changed since "
            "registration; re-register with `mlforge dataset add --force`",
        )
    return samples, id_map


def _ensure_pretrained(root: Path) -> Path:
    """Fetch (once) the M5 ImageNet base into the project cache."""
    cache = root / ".cache" / "weights" / "osnet_x1_0_imagenet.pth"
    if cache.is_file() and cache.stat().st_size > 0:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    try:
        with urllib.request.urlopen(_PRETRAIN_URL, timeout=60) as resp, \
                open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(cache)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise PreconditionFailed(
            f"could not download the OSNet ImageNet base weights: {exc}",
            hint=f"network fetch from {_PRETRAIN_URL} failed — train from "
                 "scratch with `pretrained: false`, or place the file at "
                 f"{cache}",
        ) from exc
    return cache


class ReidTrainer:
    """OSNet re-ID trainer — real gradients, real optimizer, real state."""

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
        run_dir: Path | None = None,   # scratch dir (unused here)
        on_progress: Any = None,       # fast steps: worker heartbeats itself
        init_weights: bytes | None = None,
    ):
        if dependency_error() is not None:
            raise PreconditionFailed(
                dependency_error() or "trainer dependencies missing",
                hint="pip install torch pillow",
            ) from _TORCH_IMPORT_ERROR
        # -- placement (runtime.device; never hard-wired, fail-closed) ----
        self.device = resolve_device(runtime)
        arch = MODEL_REGISTRY.get(model_name)
        if arch is None:
            raise PreconditionFailed(
                f"model {model_name!r} has no registered re-ID architecture",
                hint=f"trainable here: {', '.join(sorted(MODEL_REGISTRY))}",
            )
        # -- semantic identity, honored literally -------------------------
        loss_name = str(semantic.get("loss", "")).strip().lower()
        if loss_name not in _LOSSES:
            raise ValidationBlock(
                f"semantic.loss {loss_name!r} is not implemented by the "
                f"re-ID trainer (supports: {', '.join(sorted(_LOSSES))})",
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
        if micro < 2:
            # OSNet's fc sits behind BatchNorm1d — a train-mode forward
            # with one sample per micro-batch dies ("Expected more than 1
            # value per channel"). Refuse rather than ship a config that
            # cannot complete an epoch (architecture identity is frozen:
            # switching to GroupNorm would break every published weight).
            raise ValidationBlock(
                f"micro_batch {micro} < 2 — this architecture's BatchNorm "
                "needs >= 2 samples per micro-batch in train mode"
                + ("" if plan is None else " (from the execution plan)"),
                hint="set global_batch >= 2, or plan micro_batch >= 2 with "
                     "grad_accum raised so micro × accum still equals "
                     "global_batch",
            )
        self.micro = micro
        self.accum = accum
        self.scheduler_kind = sched_name
        self.model_name = model_name

        # -- data ----------------------------------------------------------
        self.samples, self.id_map = load_reid_samples(root, train_datasets)
        self.n = len(self.samples)
        self.steps_per_epoch = max(1, self.n // self.global_batch)
        self.num_classes = len(self.id_map)
        self.arch = arch

        # -- model / optimizer / schedule ----------------------------------
        # Weights init on CPU under the seeded RNG (bit-identical init on
        # every host), then placed on the resolved device.
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.model = OSNet(arch, self.num_classes).to(self.device)
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

        # -- init: resume (full) > fine-tune (weights) > base pretrained ---
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
            self._init_from_weights(init_weights, strict_backbone=True)
        else:
            pretrained = semantic.get("pretrained", True)
            if isinstance(pretrained, str):
                flag = pretrained.strip().lower()
                if flag in ("false", "none", "scratch", "0", ""):
                    pretrained = False
                elif flag in ("true", "imagenet", "1"):
                    pretrained = True
                else:
                    raise ValidationBlock(
                        f"semantic.pretrained {pretrained!r} is not "
                        "understood (use `imagenet` or `false`)",
                    )
            if pretrained:
                base = _ensure_pretrained(Path(root))
                self._init_from_weights(
                    base.read_bytes(), strict_backbone=True,
                    allow_head_mismatch=True,
                )
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

    # -- restore / init ------------------------------------------------------

    @staticmethod
    def _load(blob: bytes) -> Any:
        return torch.load(io.BytesIO(blob), weights_only=True, map_location="cpu")

    def _init_from_weights(self, blob: bytes, *, strict_backbone: bool,
                           allow_head_mismatch: bool = False) -> None:
        """Fine-tune / base init: load matching weights into a FRESH
        optimizer. Backbone keys must all match (architecture identity);
        the classifier head may differ when the label space differs (its
        fresh init is recorded — never a silent partial load)."""
        try:
            state = self._load(blob)
            if not isinstance(state, dict):
                raise TypeError(f"expected a state dict, got {type(state)}")
            own = self.model.state_dict()
            matched: list[str] = []
            skipped: list[str] = []
            for key, value in state.items():
                key = key.removeprefix("module.")
                if key in own and own[key].shape == value.shape:
                    own[key] = value
                    matched.append(key)
                else:
                    skipped.append(key)
            backbone_missing = [
                k for k in own
                if k not in matched and not k.startswith("classifier")
            ]
            if strict_backbone and backbone_missing:
                raise KeyError(
                    f"{len(backbone_missing)} backbone parameter(s) not "
                    f"provided (first: {backbone_missing[0]})"
                )
            head_skipped = [k for k in skipped
                            if k.startswith("classifier")]
            if head_skipped and not allow_head_mismatch:
                raise KeyError(
                    f"classifier shape mismatch: {head_skipped} — "
                    "fine-tuning into a different identity space needs an "
                    "explicit fresh head (base pretrained init allows it)"
                )
            self.model.load_state_dict(own)
        except ValidationBlock:
            raise
        except Exception as exc:
            raise ValidationBlock(
                f"weights not loadable by trainer {self.model_name!r}: "
                f"{exc}",
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
            self.scheduler._last_lr = [float(v) for v in
                                       sched_state["_last_lr"]]
            ids_state = json.loads(
                payloads.get("dataloader_state", b"{}").decode("utf-8")
            )
            recorded = ids_state.get("id_map")
            if recorded is not None and recorded != self.id_map:
                raise ValidationBlock(
                    "checkpoint label space differs from the prepared "
                    "dataset (identity map changed) — a resume continues "
                    "the SAME experiment, never a new label space",
                )
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

    # -- data -----------------------------------------------------------------

    def _epoch_order(self, epoch: int) -> torch.Tensor:
        """Deterministic per-epoch sample order (recorded protocol)."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed((self.seed * 1_000_003 + epoch) % (2 ** 63))
        base = torch.randperm(self.n, generator=gen)
        need = self.steps_per_epoch * self.global_batch
        if base.numel() < need:
            reps = (need + base.numel() - 1) // base.numel()
            base = base.repeat(reps)
        return base[:need]

    def _gather(self, indices: torch.Tensor):
        """Indices → batch tensor (256x128, flip-augmented, normalized)."""
        h, w = self.arch.input_hw
        imgs: list[torch.Tensor] = []
        for i in indices.tolist():
            sample = self.samples[int(i)]
            try:
                with Image.open(sample["path"]) as im:
                    im = im.convert("RGB").resize((w, h), Image.BILINEAR)
                    if random.random() < 0.5:  # horizontal flip (recorded)
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    buf.seek(0)
                    with Image.open(buf) as rgb:
                        data = rgb.tobytes()
            except OSError as exc:
                raise ValidationBlock(
                    f"cannot read image {sample['path']}: {exc}",
                    hint="the dataset content changed since registration — "
                         "re-register with `mlforge dataset add --force`",
                ) from exc
            t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            t = t.view(h, w, 3).permute(2, 0, 1).float().div(255.0)
            imgs.append(t)
        x = torch.stack(imgs)
        mean = torch.tensor(_IM_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IM_STD).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = torch.tensor(
            [self.samples[int(i)]["label"] for i in indices.tolist()],
            dtype=torch.long,
        )
        return x.to(self.device), y.to(self.device)

    # -- protocol ------------------------------------------------------------

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
        correct = 0
        seen = 0
        for i in range(n_micro):
            part = batch[i * micro:(i + 1) * micro]
            if part.numel() == 0:
                break
            x, y = self._gather(part)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise ValidationBlock(
                    f"loss diverged at step {state.global_step} "
                    f"({float(loss)}) — refusing to poison checkpoints",
                )
            (loss / n_micro).backward()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(dim=1) == y).sum())
            seen += int(y.numel())
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
                "batch_top1": round(correct / max(1, seen), 4),
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
                "samples": int(self.n),
                "num_classes": int(self.num_classes),
                "id_map": self.id_map,
                "input_hw": list(self.arch.input_hw),
                "augmentation": "resize_256x128 + hflip(0.5) + "
                                "imagenet_norm",
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
TRAINER_CLASS = ReidTrainer


__all__ = [
    "MODEL_REGISTRY",
    "TRAINER_CLASS",
    "OSNet",
    "OsNetArch",
    "ReidTrainer",
    "dependency_error",
    "load_reid_samples",
    "torch_available",
]
