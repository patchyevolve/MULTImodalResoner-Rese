"""Minimal YAML-subset loader/dumper — stdlib-only core (12 §6.5).

Why this exists: the specs author human config as YAML (`ingestion.yaml`,
`datasets.yaml`, `~/.mlforge/datasets_<project>.yaml` — 12 §10.2/§6.3)
while the core stays zero-dependency. This module supports EXACTLY the
shapes the specs use:

  * block mappings (nested by indentation)
  * block sequences of scalars (`- item`)
  * flow collections: ``[a, b]`` and ``{k: v}`` (nestable)
  * scalars: single/double-quoted strings, int, float, bool, null/~
  * ``#`` comments and blank lines

Fail-closed (13 §1): anything outside this subset — anchors/aliases,
block scalars (``|``/``>``), tabs for indentation, multiple documents,
list-of-mappings — raises ``YamlError``. We never guess what a config
we cannot fully parse was supposed to mean. The system only reads files
in this subset (ours are produced by :func:`dump`).
"""

from __future__ import annotations

from typing import Any


class YamlError(ValueError):
    """Unparseable / unsupported YAML — callers map it to exit 3 (config)."""


def _strip_comment(line: str) -> str:
    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            out.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
        else:
            if ch in "'\"":
                quote = ch
                out.append(ch)
            elif ch == "#" and (not out or out[-1] in " \t"):
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _scalar(tok: str) -> Any:
    tok = tok.strip()
    if tok == "" or tok in ("~", "null", "Null", "NULL"):
        return None
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "'\"":
        body = tok[1:-1]
        if tok[0] == '"':
            return body.encode().decode("unicode_escape")
        return body.replace("''", "'")
    low = tok.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(tok, 10)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    if tok.startswith(("[", "{")):
        raise YamlError(f"unbalanced flow collection: {tok!r}")
    if any(tok.startswith(p) for p in ("&", "*", "!", "|", ">")):
        raise YamlError(f"unsupported YAML construct: {tok!r}")
    return tok


def _parse_flow(text: str) -> Any:
    """Parse a flow collection starting at text[0] in [ { ."""
    pos = 0

    def skip_ws() -> None:
        nonlocal pos
        while pos < len(text) and text[pos] in " \t":
            pos += 1

    def parse_value() -> Any:
        nonlocal pos
        skip_ws()
        if pos >= len(text):
            raise YamlError(f"truncated flow value: {text!r}")
        ch = text[pos]
        if ch in "[{":
            return parse_collection()
        # scalar / quoted / nested-through-commas
        start = pos
        quote: str | None = None
        depth = 0
        while pos < len(text):
            c = text[pos]
            if quote:
                if c == "\\" and quote == '"' and pos + 1 < len(text):
                    pos += 2
                    continue
                if c == quote:
                    quote = None
            elif c in "'\"":
                quote = c
            elif c in "[{":
                depth += 1
            elif c in "]}":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            pos += 1
        return _scalar(text[start:pos])

    def parse_collection() -> Any:
        nonlocal pos
        opener = text[pos]
        closer = "]" if opener == "[" else "}"
        pos += 1
        items: list[Any] = []
        pairs: dict[str, Any] = {}
        skip_ws()
        if pos < len(text) and text[pos] == closer:
            pos += 1
            return [] if opener == "[" else {}
        while True:
            skip_ws()
            if opener == "{":
                key = _flow_key()
                if key is None:
                    raise YamlError(f"flow map expects 'key: value': {text!r}")
                skip_ws()
                if pos >= len(text) or text[pos] != ":":
                    raise YamlError(f"flow map expects ':': {text!r}")
                pos += 1
                pairs[key] = parse_value()
            else:
                items.append(parse_value())
            skip_ws()
            if pos >= len(text):
                raise YamlError(f"unterminated flow collection: {text!r}")
            if text[pos] == ",":
                pos += 1
                continue
            if text[pos] == closer:
                pos += 1
                break
            raise YamlError(f"unexpected {text[pos]!r} in flow: {text!r}")
        return items if opener == "[" else pairs

    def _flow_key() -> str | None:
        nonlocal pos
        if pos >= len(text):
            return None
        if text[pos] in "'\"":
            q = text[pos]
            end = text.find(q, pos + 1)
            if end < 0:
                raise YamlError(f"unterminated key quote: {text!r}")
            key = text[pos + 1 : end]
            pos = end + 1
        else:
            start = pos
            while pos < len(text) and text[pos] not in ":,{}[] \t":
                pos += 1
            key = text[start:pos]
        save = pos
        skip_ws()
        if pos < len(text) and text[pos] == ":":
            return key
        pos = save
        return None

    result = parse_collection()
    skip_ws()
    if pos != len(text):
        raise YamlError(f"trailing content in flow: {text[pos:]!r}")
    return result


def _split_key_value(line: str) -> tuple[str, Any] | None:
    """`key: value` / `key:` — returns None when the line is not a mapping."""
    if ":" not in line:
        return None
    if line.lstrip().startswith("- "):
        return None
    # keys never contain ": " (YAML rule) — split on first ": " or trailing ":"
    in_q: str | None = None
    for i, ch in enumerate(line):
        if in_q:
            if ch == in_q:
                in_q = None
            continue
        if ch in "'\"":
            in_q = ch
            continue
        if ch == ":" and (i + 1 == len(line) or line[i + 1] == " "):
            key = line[:i].strip()
            if not key:
                raise YamlError(f"empty key in line: {line!r}")
            if key[0] in "'\"":
                key = str(_scalar(key))
            rest = line[i + 1 :].strip()
            if rest.startswith(("|", ">", "&", "*")):
                raise YamlError(f"unsupported YAML construct: {line!r}")
            return key, (_parse_flow(rest) if rest.startswith(("[", "{")) else _scalar(rest))
    return None


def loads(text: str) -> Any:
    """Parse the supported YAML subset. Empty document → {}."""
    raw = text.splitlines()
    # pre-validate: tabs for indentation are unsupported (YAML forbids them
    # anyway) and multi-doc markers are out of scope
    for n, line in enumerate(raw, 1):
        stripped = _strip_comment(line)
        if not stripped.strip():
            continue
        if stripped.lstrip().startswith("---") or stripped.lstrip().startswith("..."):
            raise YamlError(f"line {n}: multi-document YAML unsupported")
        indent_part = line[: len(line) - len(line.lstrip())]
        if "\t" in indent_part:
            raise YamlError(f"line {n}: tabs are not valid indentation")
    lines = [(n, _strip_comment(line)) for n, line in enumerate(raw, 1)]
    lines = [(n, ln) for n, ln in lines if ln.strip()]
    if not lines:
        return {}
    value, idx = _parse_block(lines, 0, len(lines[0][1]) - len(lines[0][1].lstrip()))
    if idx != len(lines):
        raise YamlError(f"line {lines[idx][0]}: unexpected content {lines[idx][1].strip()!r}")
    return value


def _parse_block(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[Any, int]:
    _n, ln = lines[i]
    body = ln[indent:] if len(ln) >= indent else ln.lstrip()
    if body.startswith("- ") or body == "-":
        return _parse_seq(lines, i, indent)
    if body.startswith(("[", "{")):
        # flow collection on its own line (block level)
        return _parse_flow(body), i + 1
    return _parse_map(lines, i, indent)


def _parse_seq(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    while i < len(lines):
        n, ln = lines[i]
        cur = len(ln) - len(ln.lstrip())
        if cur < indent:
            break
        body = ln[cur:]
        if cur == indent and (body.startswith("- ") or body == "-"):
            rest = body[1:].strip()
            if not rest:
                raise YamlError(f"line {n}: empty sequence item (nested blocks unsupported)")
            if rest.startswith(("[", "{")):
                items.append(_parse_flow(rest))
                i += 1
                continue
            kv = _split_key_value(rest)
            if kv is not None:
                raise YamlError(f"line {n}: sequence of mappings unsupported")
            items.append(_scalar(rest))
            i += 1
            continue
        if cur > indent:
            raise YamlError(f"line {n}: unexpected indentation")
        break
    return items, i


def _parse_map(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while i < len(lines):
        n, ln = lines[i]
        cur = len(ln) - len(ln.lstrip())
        if cur < indent:
            break
        if cur > indent:
            raise YamlError(f"line {n}: unexpected indentation")
        body = ln[cur:]
        if body.startswith("- ") or body == "-":
            raise YamlError(f"line {n}: sequence where mapping expected")
        kv = _split_key_value(body)
        if kv is None:
            raise YamlError(f"line {n}: expected 'key: value', got {body.strip()!r}")
        key, value = kv
        if key in out:
            raise YamlError(f"line {n}: duplicate key {key!r}")
        i += 1
        if value is None and i < len(lines):
            _nxt_n, nxt = lines[i]
            nxt_indent = len(nxt) - len(nxt.lstrip())
            if nxt_indent > indent:
                value, i = _parse_block(lines, i, nxt_indent)
        out[key] = value
    return out, i


def dump(data: Any, indent: int = 0) -> str:
    """Emit the same subset (stable, sorted keys) — round-trips via loads."""
    pad = "  " * indent
    if isinstance(data, dict):
        if not data:
            return pad + "{}\n" if indent else "{}\n"
        out = []
        for k in sorted(data):
            v = data[k]
            if isinstance(v, dict) and v:
                out.append(f"{pad}{k}:\n{dump(v, indent + 1)}")
            elif isinstance(v, list) and v:
                if any(isinstance(x, (dict, list)) for x in v):
                    # nested collections only parse inline (flow form)
                    out.append(f"{pad}{k}: ["
                               + ", ".join(_dump_flow_item(x) for x in v) + "]\n")
                else:
                    out.append(f"{pad}{k}:\n{dump(v, indent + 1)}")
            elif isinstance(v, dict):
                out.append(f"{pad}{k}: {{}}\n")
            elif isinstance(v, list):
                out.append(f"{pad}{k}: []\n")
            else:
                out.append(f"{pad}{k}: {_dump_scalar(v)}\n")
        return "".join(out)
    if isinstance(data, list):
        if not data:
            return pad + "[]\n"
        if any(isinstance(v, (dict, list)) for v in data):
            # our parser accepts nested collections only in flow form
            return pad + "[" + ", ".join(_dump_flow_item(v) for v in data) + "]\n"
        return "".join(f"{pad}- {_dump_scalar(v)}\n" for v in data)
    return f"{pad}{_dump_scalar(data)}\n"


def _dump_flow_item(v: Any) -> str:
    if isinstance(v, dict):
        inner = ", ".join(f"{k}: {_dump_flow_item(x)}" for k, x in sorted(v.items()))
        return "{" + inner + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_dump_flow_item(x) for x in v) + "]"
    return _dump_scalar(v)


def _dump_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if s != s.strip() or s[:1] in "&*!|>{['\"" or ": " in s or s.endswith(":") \
            or s.split("#")[0] != s or any(c in s for c in "\n\t"):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return s


def load_file(path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise YamlError(f"cannot read {path}: {exc}") from exc
    return loads(text)


__all__ = ["YamlError", "dump", "load_file", "loads"]
