"""Ingestion subsystem (12 §6.2–§6.4, §10) — datasets in, prepared
artifacts out, identity everywhere.

    identity  — cryptographic dataset manifests (files → sha256)
    config    — machine-local paths vs portable registrations
    dag       — ingestion.yaml model plans (sources/transforms/DAG)
    transforms — closed registry, deterministic, code-hashed
    prepare   — resolve → cache → transform → REGISTERED→PREPARED
"""

from mlforge.ingest.config import load_paths, mlforge_home, paths_file, set_path
from mlforge.ingest.dag import ModelPlan, Source, load_ingestion, resolve_order
from mlforge.ingest.identity import (
    DEFAULT_VERSION,
    DatasetManifest,
    FileEntry,
    build_manifest,
    full_ref,
    parse_ref,
    recompute_identity,
    scan_files,
)
from mlforge.ingest.prepare import PreparedResult, derived_dataset_id, prepare
from mlforge.ingest.transforms import (
    ResolvedSource,
    get_transform,
    registry_names,
    transform_identity,
)

__all__ = [
    "DEFAULT_VERSION",
    "DatasetManifest",
    "FileEntry",
    "ModelPlan",
    "PreparedResult",
    "ResolvedSource",
    "Source",
    "build_manifest",
    "derived_dataset_id",
    "full_ref",
    "get_transform",
    "load_ingestion",
    "load_paths",
    "mlforge_home",
    "parse_ref",
    "paths_file",
    "prepare",
    "recompute_identity",
    "registry_names",
    "resolve_order",
    "scan_files",
    "set_path",
    "transform_identity",
]
