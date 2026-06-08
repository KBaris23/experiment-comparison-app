from __future__ import annotations

import json
import re
from typing import Iterable, Optional

import pandas as pd


ChannelMap = dict[str, set[int]]

INPUT_CHANNEL_COLUMNS = (
    "successful_channels",
    "success_channels",
    "selected_successful_channels",
    "analysis_channels",
    "analyzed_channels",
    "selected_channels",
    "channels_selected",
    "channels_analyzed",
    "channels_used",
    "included_channels",
    "processed_channels",
    "fit_channels",
    "kd_channels",
    "channel_list",
    "channels",
    "channel",
)


def _normal_channel(value: object) -> Optional[int]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


def _parse_channel_values(value: object) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, bool):
        return set()
    if isinstance(value, (list, tuple, set)):
        channels: set[int] = set()
        for item in value:
            channels.update(_parse_channel_values(item))
        return channels
    if isinstance(value, dict):
        channels: set[int] = set()
        for key, item in value.items():
            if "channel" in str(key).lower():
                channels.update(_parse_channel_values(item))
        return channels
    if isinstance(value, (int, float)):
        channel = _normal_channel(value)
        return {channel} if channel is not None else set()

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return set()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if parsed is not None and parsed != text:
        parsed_channels = _parse_channel_values(parsed)
        if parsed_channels:
            return parsed_channels

    channels = set()
    for match in re.finditer(r"(?i)(?:ch(?:annel)?\s*)?(\d+)", text):
        channel = _normal_channel(match.group(1))
        if channel is not None:
            channels.add(channel)
    return channels


def _candidate_input_columns(inputs: pd.DataFrame) -> list[str]:
    if inputs.empty:
        return []
    exact = {col.lower(): col for col in inputs.columns}
    candidates = [exact[name] for name in INPUT_CHANNEL_COLUMNS if name in exact]
    for col in inputs.columns:
        lower = col.lower()
        if col in candidates:
            continue
        if "channel" in lower and any(token in lower for token in ("success", "selected", "analysis", "analyzed", "used", "processed", "fit", "kd")):
            candidates.append(col)
    return candidates


def _channels_from_inputs(inputs: pd.DataFrame) -> set[int]:
    channels: set[int] = set()
    for col in _candidate_input_columns(inputs):
        for value in inputs[col].dropna().tolist():
            channels.update(_parse_channel_values(value))
    return channels


def _channels_from_kd(langmuir_summary: pd.DataFrame) -> set[int]:
    if langmuir_summary.empty or "channel" not in langmuir_summary.columns or "langmuir_kd" not in langmuir_summary.columns:
        return set()
    kd = pd.to_numeric(langmuir_summary["langmuir_kd"], errors="coerce")
    return {
        int(channel)
        for channel in pd.to_numeric(langmuir_summary.loc[kd.notna(), "channel"], errors="coerce").dropna().tolist()
    }


def _channels_from_metric(results: pd.DataFrame, metric_col: Optional[str]) -> set[int]:
    if results.empty or "channel" not in results.columns:
        return set()
    if metric_col and metric_col in results.columns:
        metric = pd.to_numeric(results[metric_col], errors="coerce")
        source = results.loc[metric.notna(), "channel"]
    else:
        source = results["channel"]
    return {int(channel) for channel in pd.to_numeric(source, errors="coerce").dropna().tolist()}


def successful_channel_map(
    inputs: pd.DataFrame,
    langmuir_summary: pd.DataFrame,
    results: pd.DataFrame,
    metric_col: Optional[str] = None,
    include_metric_fallback: bool = True,
) -> ChannelMap:
    labels: set[str] = set()
    for df in (inputs, langmuir_summary, results):
        if not df.empty and "experiment_label" in df.columns:
            labels.update(str(value) for value in df["experiment_label"].dropna().unique())

    channel_map: ChannelMap = {}
    for label in sorted(labels):
        input_rows = inputs[inputs["experiment_label"].astype(str) == label] if "experiment_label" in inputs.columns else pd.DataFrame()
        fit_rows = langmuir_summary[langmuir_summary["experiment_label"].astype(str) == label] if "experiment_label" in langmuir_summary.columns else pd.DataFrame()
        result_rows = results[results["experiment_label"].astype(str) == label] if "experiment_label" in results.columns else pd.DataFrame()

        channels = _channels_from_inputs(input_rows)
        if not channels:
            channels = _channels_from_kd(fit_rows)
        if not channels and include_metric_fallback:
            channels = _channels_from_metric(result_rows, metric_col)
        if channels:
            channel_map[label] = channels
    return channel_map


def channel_union(channel_map: ChannelMap) -> list[int]:
    channels: set[int] = set()
    for values in channel_map.values():
        channels.update(values)
    return sorted(channels)


def filter_successful_channels(
    df: pd.DataFrame,
    channel_map: ChannelMap,
    channels: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    if df.empty or "channel" not in df.columns:
        return df

    selected = {int(channel) for channel in channels or []}
    out = df.copy()
    numeric_channel = pd.to_numeric(out["channel"], errors="coerce")

    if not channel_map:
        if selected:
            return out[numeric_channel.isin(selected)].copy()
        return out

    if "experiment_label" not in out.columns:
        allowed = channel_union(channel_map)
        if selected:
            allowed = [channel for channel in allowed if channel in selected]
        return out[numeric_channel.isin(allowed)].copy()

    mask = pd.Series(False, index=out.index)
    labels = out["experiment_label"].astype(str)
    for label, allowed_channels in channel_map.items():
        allowed = set(allowed_channels)
        if selected:
            allowed &= selected
        if allowed:
            mask |= labels.eq(str(label)) & numeric_channel.isin(allowed)
    return out[mask].copy()
