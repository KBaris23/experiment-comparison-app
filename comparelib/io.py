from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


TABLE_KEYS = (
    "signal_processing_inputs",
    "results",
    "titration_steps",
    "langmuir_fit_summary",
)


@dataclass(frozen=True)
class ExperimentBundle:
    bundle_dir: Path
    manifest_path: Path
    manifest: dict
    tables: Dict[str, pd.DataFrame]

    @property
    def experiment_name(self) -> str:
        return str(self.manifest.get("experiment_name") or self.bundle_dir.name)

    @property
    def analysis_mode(self) -> str:
        return str(self.manifest.get("analysis_mode") or "")

    @property
    def created_at(self) -> str:
        return str(self.manifest.get("created_at") or "")

    @property
    def schema_version(self) -> str:
        return str(self.manifest.get("schema_version") or "")


def parse_paths(text: str) -> List[Path]:
    paths = []
    for raw_line in text.splitlines():
        token = raw_line.strip().strip('"')
        if token:
            paths.append(Path(token).expanduser())
    return paths


def discover_manifests(paths: Iterable[Path]) -> List[Path]:
    manifests = set()
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.name == "manifest.json":
            manifests.add(path.resolve())
            continue
        if path.is_dir():
            direct = path / "manifest.json"
            if direct.exists():
                manifests.add(direct.resolve())
            outputs_dir = path / "outputs"
            if outputs_dir.exists():
                manifests.update(p.resolve() for p in outputs_dir.glob("*/manifest.json"))
            manifests.update(p.resolve() for p in path.glob("outputs/*/manifest.json"))
    return sorted(manifests, key=lambda p: str(p).lower())


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bundle(manifest_path: Path) -> ExperimentBundle:
    manifest = load_manifest(manifest_path)
    bundle_dir = manifest_path.parent
    file_map = manifest.get("files") or {}
    tables: Dict[str, pd.DataFrame] = {}
    for key in TABLE_KEYS:
        rel_path = file_map.get(key)
        if not rel_path:
            continue
        table_path = bundle_dir / rel_path
        if table_path.exists():
            tables[key] = pd.read_csv(table_path)
    return ExperimentBundle(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        tables=tables,
    )


def bundle_label(bundle: ExperimentBundle) -> str:
    created = bundle.created_at[:19].replace("T", " ") if bundle.created_at else "unknown date"
    return f"{bundle.experiment_name} | {created}"


def combine_table(
    bundles: Iterable[ExperimentBundle],
    table_key: str,
    selected_labels: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    selected_set = set(selected_labels) if selected_labels is not None else None
    frames = []
    for bundle in bundles:
        label = bundle_label(bundle)
        if selected_set is not None and label not in selected_set:
            continue
        table = bundle.tables.get(table_key)
        if table is None or table.empty:
            continue
        df = table.copy()
        df.insert(0, "experiment_label", label)
        df.insert(1, "experiment_name", bundle.experiment_name)
        df.insert(2, "bundle_dir", str(bundle.bundle_dir))
        df.insert(3, "analysis_mode", bundle.analysis_mode)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def manifest_summary(bundles: Iterable[ExperimentBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        files = bundle.manifest.get("files") or {}
        metadata = bundle.manifest.get("metadata") or {}
        row = {
            "experiment_label": bundle_label(bundle),
            "experiment_name": bundle.experiment_name,
            "analysis_mode": bundle.analysis_mode,
            "created_at": bundle.created_at,
            "schema_version": bundle.schema_version,
            "bundle_dir": str(bundle.bundle_dir),
            "source_folder_count": len(bundle.manifest.get("source_folders") or []),
            "has_results": "results" in files,
            "has_titration_steps": "titration_steps" in files,
            "has_langmuir_fit_summary": "langmuir_fit_summary" in files,
        }
        for key, value in metadata.items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    exclude_set = set(exclude or [])
    cols = []
    for col in df.columns:
        if col in exclude_set:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            cols.append(col)
    return cols
