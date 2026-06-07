from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_CURRENT_METRICS = (
    "peak_current_selected",
    "peak_height",
    "peak_current",
    "current_peak_height",
    "peak_current_height",
)

DEFAULT_VOLTAGE_METRICS = (
    "peak_voltage_selected",
    "peak_voltage",
    "peak_potential_selected",
    "peak_potential",
    "voltage_at_peak",
)

BASE_COLUMNS = (
    "experiment_label",
    "experiment_name",
    "bundle_dir",
    "analysis_mode",
)

HANDLED_INPUT_COLUMNS = (
    "aptamer_type",
    "aptamer",
    "target_aptamer",
    "thiol_count",
    "thiol_type",
    "thiol",
    "analysis_vlines_json",
)

PARAMETER_KEYWORDS = (
    "swv",
    "step",
    "start",
    "end",
    "timestamp",
    "time",
    "pulse",
    "amplitude",
    "increment",
    "frequency",
    "quiet",
    "window",
    "baseline",
)


def metric_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def slugify(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "experiment"


def plot_path(experiment_name: object, plot_name: str) -> str:
    return str(Path("plots") / f"{slugify(experiment_name)}_{plot_name}.png").replace("\\", "/")


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _first_value(df: pd.DataFrame, columns: Iterable[str]) -> str:
    if df.empty:
        return ""
    for col in columns:
        if col not in df.columns:
            continue
        values = [_clean_text(value) for value in df[col].dropna().tolist()]
        values = [value for value in values if value]
        if values:
            return values[0]
    return ""


def _unique_value_from_columns(df: pd.DataFrame, columns: Iterable[str]) -> str:
    if df.empty:
        return ""
    for col in columns:
        if col in df.columns:
            return _unique_join(df[col])
    return ""


def _compact_json(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except Exception:
        return text


def _input_value(df: pd.DataFrame, col: str) -> str:
    if df.empty or col not in df.columns:
        return ""
    if col == "analysis_vlines_json":
        values = [_compact_json(value) for value in df[col].dropna().tolist()]
        values = [value for value in values if value]
        return values[0] if values else ""
    return _unique_join(df[col])


def parameter_columns(inputs: pd.DataFrame) -> list[str]:
    if inputs.empty:
        return []
    excluded = set(BASE_COLUMNS) | set(HANDLED_INPUT_COLUMNS)
    candidates = []
    for col in inputs.columns:
        lower = col.lower()
        if col in excluded or lower in excluded:
            continue
        if any(keyword in lower for keyword in PARAMETER_KEYWORDS):
            candidates.append(col)
    return candidates


def parameter_label(col: str) -> str:
    if col.lower() == "analysis_vlines_json":
        return "Vlines"
    return col.replace("_", " ").strip().title()


def _unique_join(series: pd.Series, max_items: int = 8) -> str:
    values = []
    for value in series.dropna().tolist():
        text = _clean_text(value)
        if text and text not in values:
            values.append(text)
    if not values:
        return ""
    shown = values[:max_items]
    suffix = f" (+{len(values) - max_items} more)" if len(values) > max_items else ""
    return ", ".join(shown) + suffix


def _format_number(value: object, precision: int = 6) -> object:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return float(f"{float(numeric):.{precision}g}")


def _infer_aptamer_type(experiment_name: str) -> str:
    tokens = [token for token in re.split(r"[_\-\s]+", experiment_name.lower()) if token]
    tokens = [token for token in tokens if not re.fullmatch(r"\d{4,8}", token)]
    for token in tokens:
        if token in {"mono", "monothiol", "di", "dithiol", "tri", "trithiol", "trithic", "tetra", "tetrathiol"}:
            continue
        if "thiol" in token or "thio" in token or token == "trithic":
            continue
        return token
    return ""


def _infer_thiol_count(experiment_name: str) -> str:
    haystack = experiment_name.lower()
    explicit = {
        "monothiol": "monothiol",
        "mono-thiol": "monothiol",
        "mono_thiol": "monothiol",
        "dithiol": "dithiol",
        "di-thiol": "dithiol",
        "di_thiol": "dithiol",
        "trithiol": "trithiol",
        "tri-thiol": "trithiol",
        "tri_thiol": "trithiol",
        "tetrathiol": "tetrathiol",
        "tetra-thiol": "tetrathiol",
        "tetra_thiol": "tetrathiol",
    }
    for needle, label in explicit.items():
        if needle in haystack:
            return label
    for token in re.split(r"[_\-\s]+", haystack):
        if token.startswith("mono"):
            return "monothiol"
        if token.startswith("di"):
            return "dithiol"
        if token.startswith("tri"):
            return "trithiol"
        if token.startswith("tetra"):
            return "tetrathiol"
    return ""


def _channel_list(df: pd.DataFrame) -> str:
    if df.empty or "channel" not in df.columns:
        return ""
    channels = pd.to_numeric(df["channel"], errors="coerce").dropna().astype(int).drop_duplicates().sort_values()
    return ", ".join(str(value) for value in channels.tolist())


def _successful_channel_count(df: pd.DataFrame, metric_col: Optional[str]) -> object:
    if df.empty or "channel" not in df.columns:
        return ""
    if metric_col and metric_col in df.columns:
        metric = pd.to_numeric(df[metric_col], errors="coerce")
        channels = pd.to_numeric(df.loc[metric.notna(), "channel"], errors="coerce").dropna().astype(int).unique()
        return int(len(channels))
    channels = pd.to_numeric(df["channel"], errors="coerce").dropna().astype(int).unique()
    return int(len(channels))


def _range(df: pd.DataFrame, col: Optional[str]) -> object:
    if df.empty or not col or col not in df.columns:
        return ""
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return ""
    return _format_number(values.max() - values.min())


def _max(df: pd.DataFrame, col: Optional[str]) -> object:
    if df.empty or not col or col not in df.columns:
        return ""
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return ""
    return _format_number(values.max())


def build_comparison_summary(
    manifest_summary: pd.DataFrame,
    results: pd.DataFrame,
    inputs: pd.DataFrame,
    langmuir_summary: pd.DataFrame,
    current_metric: Optional[str] = None,
    voltage_metric: Optional[str] = None,
) -> pd.DataFrame:
    current_col = current_metric or metric_column(results, DEFAULT_CURRENT_METRICS)
    voltage_col = voltage_metric or metric_column(results, DEFAULT_VOLTAGE_METRICS)
    input_parameter_cols = parameter_columns(inputs)
    _ = langmuir_summary

    rows = []
    for _, manifest_row in manifest_summary.iterrows():
        label = manifest_row.get("experiment_label", "")
        experiment_name = _clean_text(manifest_row.get("experiment_name")) or _clean_text(label)
        plot_key = _clean_text(label) or experiment_name
        result_rows = results[results.get("experiment_label") == label] if "experiment_label" in results.columns else pd.DataFrame()
        input_rows = inputs[inputs.get("experiment_label") == label] if "experiment_label" in inputs.columns else pd.DataFrame()
        fit_rows = (
            langmuir_summary[langmuir_summary.get("experiment_label") == label]
            if "experiment_label" in langmuir_summary.columns
            else pd.DataFrame()
        )

        frequency = _unique_value_from_columns(result_rows, ["frequency_hz", "frequency", "frequency_Hz"])
        aptamer_type = _first_value(input_rows, ["aptamer_type", "aptamer", "target_aptamer"]) or _infer_aptamer_type(experiment_name)
        thiol_count = _first_value(input_rows, ["thiol_count", "thiol_type", "thiol"]) or _infer_thiol_count(experiment_name)

        row = {
            "Experiment ID": experiment_name,
            "Frequency": frequency,
            "Aptamer Type": aptamer_type,
            "Thiol Count": thiol_count,
            "Successful Channels": _successful_channel_count(result_rows, current_col),
            "Channel List": _channel_list(result_rows),
            "Vlines": _clean_text(manifest_row.get("analysis_vlines_json")) or _input_value(input_rows, "analysis_vlines_json"),
        }
        for col in input_parameter_cols:
            row[parameter_label(col)] = _input_value(input_rows, col)
        row.update(
            {
                "Current Peak Height Plot": plot_path(plot_key, "current_peak_height"),
                "Peak Voltage Drift Plot": plot_path(plot_key, "peak_voltage_drift") if voltage_col else "",
                "Max Peak Height": _max(result_rows, current_col),
                "Peak Height Range": _range(result_rows, current_col),
                "Peak Voltage Drift Range": _range(result_rows, voltage_col),
            }
        )
        rows.append(row)

    columns = [
        "Experiment ID",
        "Frequency",
        "Aptamer Type",
        "Thiol Count",
        "Successful Channels",
        "Channel List",
        "Vlines",
        *[parameter_label(col) for col in input_parameter_cols],
        "Current Peak Height Plot",
        "Peak Voltage Drift Plot",
        "Max Peak Height",
        "Peak Height Range",
        "Peak Voltage Drift Range",
    ]
    return pd.DataFrame(rows, columns=columns)
