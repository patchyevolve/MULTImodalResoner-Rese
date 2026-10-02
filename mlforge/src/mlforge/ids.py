"""Identifier generation.

Run/model/evaluation/command ids are prefixed, time-sortable, and unique
(12_training_system.md §4, 13 §4.3 object reference syntax):

    run_01JABC...   model://rf_detr_s:v2   dataset://coco_2017:v1   eval_001

Format: <prefix>_<ULID> — 48-bit millisecond timestamp + 80 bits of
randomness, Crockford base32 (26 chars, lexicographically sortable).

Monotonic within a process: ids generated in the same millisecond
increment the random field instead of redrawing it (ULID "monotonic
mode"). Without this, two runs created in the same millisecond would
sort by luck — and "the latest run of model X" (retrain's source, 13
§6.3) would be a coin flip. Cross-process ordering within one
millisecond remains timestamp-only (ms resolution); callers needing
exact recency should compare journal `ts` events.
"""

from __future__ import annotations

import os
import secrets
import threading
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32 (no I, L, O, U)
_TIME_LEN = 10
_RAND_LEN = 16
_RAND_MAX = (1 << (_RAND_LEN * 5)) - 1  # 80 bits, one per base32 char

_lock = threading.Lock()
_last_ms = -1
_last_rand = 0


def _b32(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_ulid() -> str:
    global _last_ms, _last_rand
    with _lock:
        ms = int(time.time() * 1000)
        if ms > _last_ms:
            _last_ms, _last_rand = ms, secrets.randbits(_RAND_LEN * 5)
        else:
            # same millisecond (or clock stepped back): increment —
            # strictly increasing, still random-based across restarts
            _last_rand += 1
            if _last_rand > _RAND_MAX:
                _last_ms += 1
                _last_rand = 0
        return _b32(_last_ms, _TIME_LEN) + _b32(_last_rand, _RAND_LEN)


def new_run_id() -> str:
    return f"run_{new_ulid()}"


def new_model_id() -> str:
    return f"model_{new_ulid()}"


def new_eval_id() -> str:
    return f"eval_{new_ulid()}"


def new_export_id() -> str:
    """Export artifact identity (13 §4.1 `export` — "own artifact
    identity", §6.9)."""
    return f"exp_{new_ulid()}"


def new_bundle_id() -> str:
    """Inference bundle identity (13 §4.1 `package`)."""
    return f"bdl_{new_ulid()}"


def new_output_id() -> str:
    """One-shot inference output identity (13 §6.8 `infer`)."""
    return f"out_{new_ulid()}"


def new_command_id() -> str:
    """Client-supplied idempotency key (13 §4.4). CLI generates one per
    invocation when the user does not supply --command-id."""
    return f"cmd_{new_ulid()}"


def new_session_token() -> str:
    """Lease ownership token (12 §23.2) — proves the holder owns the lease."""
    return secrets.token_hex(16)
