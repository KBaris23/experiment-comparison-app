from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def langmuir_isotherm(x, baseline, amplitude, kd):
    return baseline + amplitude * (x / (kd + x))


def _filtered(df: pd.DataFrame, experiments: Iterable[str], channels: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    exp_set = set(experiments or [])
    ch_set = set(channels or [])
    if exp_set:
        out = out[out["experiment_label"].isin(exp_set)]
    if ch_set and "channel" in out.columns:
        out = out[out["channel"].isin(ch_set)]
    return out


def plot_metric_vs_scan(
    results: pd.DataFrame,
    metric: str,
    experiments: Iterable[str],
    channels: Iterable[int],
    x_col: str = "scan_number",
):
    df = _filtered(results, experiments, channels)
    if df.empty or metric not in df.columns or x_col not in df.columns:
        return None

    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[x_col, metric])
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for (experiment, channel), group in df.groupby(["experiment_label", "channel"], dropna=False):
        group = group.sort_values(x_col)
        ax.plot(
            group[x_col],
            group[metric],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"{experiment} | Ch{channel}",
        )
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {x_col}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_titration_plateaus(
    titration_steps: pd.DataFrame,
    metric_label: str,
    experiments: Iterable[str],
    channels: Iterable[int],
):
    df = _filtered(titration_steps, experiments, channels)
    if df.empty:
        return None
    if "metric_label" in df.columns:
        df = df[df["metric_label"] == metric_label]
    if df.empty:
        return None

    x_col = "step_concentration" if "step_concentration" in df.columns and pd.to_numeric(df["step_concentration"], errors="coerce").notna().any() else "step_index"
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df["plateau_value"] = pd.to_numeric(df["plateau_value"], errors="coerce")
    df = df.dropna(subset=[x_col, "plateau_value"])
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for (experiment, channel), group in df.groupby(["experiment_label", "channel"], dropna=False):
        group = group.sort_values(x_col)
        ax.plot(
            group[x_col],
            group["plateau_value"],
            marker="D",
            markersize=4,
            linewidth=1.4,
            label=f"{experiment} | Ch{channel}",
        )
    unit = ""
    if x_col == "step_concentration" and "step_concentration_unit" in df.columns:
        units = [str(v) for v in df["step_concentration_unit"].dropna().unique() if str(v)]
        if units:
            unit = f" ({units[0]})"
    ax.set_xlabel("Concentration" + unit if x_col == "step_concentration" else "Titration step")
    ax.set_ylabel("Plateau value")
    ax.set_title(f"Titration plateaus | {metric_label}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_langmuir_curves(
    titration_steps: pd.DataFrame,
    langmuir_summary: pd.DataFrame,
    metric_label: str,
    experiments: Iterable[str],
    channels: Iterable[int],
):
    steps = _filtered(titration_steps, experiments, channels)
    fits = _filtered(langmuir_summary, experiments, channels)
    if "metric_label" in steps.columns:
        steps = steps[steps["metric_label"] == metric_label]
    if "metric_label" in fits.columns:
        fits = fits[fits["metric_label"] == metric_label]
    if steps.empty or fits.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    grouped_steps = {
        (experiment, channel): group
        for (experiment, channel), group in steps.groupby(["experiment_label", "channel"], dropna=False)
    }
    for _, fit in fits.iterrows():
        experiment = fit.get("experiment_label")
        channel = fit.get("channel")
        kd = pd.to_numeric(pd.Series([fit.get("langmuir_kd")]), errors="coerce").iloc[0]
        baseline = pd.to_numeric(pd.Series([fit.get("langmuir_baseline")]), errors="coerce").iloc[0]
        amplitude = pd.to_numeric(pd.Series([fit.get("langmuir_amplitude")]), errors="coerce").iloc[0]
        if not np.isfinite(kd) or not np.isfinite(baseline) or not np.isfinite(amplitude):
            continue
        group = grouped_steps.get((experiment, channel))
        if group is None or group.empty:
            continue
        x = pd.to_numeric(group.get("step_concentration"), errors="coerce")
        y = pd.to_numeric(group.get("plateau_value"), errors="coerce")
        keep = x.notna() & y.notna()
        x = x[keep]
        y = y[keep]
        if x.empty:
            continue
        order = np.argsort(x.to_numpy())
        x_sorted = x.to_numpy()[order]
        y_sorted = y.to_numpy()[order]
        label = f"{experiment} | Ch{channel}"
        ax.scatter(x_sorted, y_sorted, s=26, label=label)
        x_dense = np.linspace(float(np.nanmin(x_sorted)), float(np.nanmax(x_sorted)), 300)
        ax.plot(x_dense, langmuir_isotherm(x_dense, baseline, amplitude, kd), linewidth=1.7)
        kd_y = float(langmuir_isotherm(kd, baseline, amplitude, kd))
        ax.axvline(kd, linestyle="--", linewidth=1.0, alpha=0.45)
        unit = str(fit.get("langmuir_kd_unit") or "")
        ax.annotate(f"Kd {kd:.3g} {unit}".strip(), xy=(kd, kd_y), xytext=(7, 8), textcoords="offset points", fontsize=8)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    unit = ""
    if "step_concentration_unit" in steps.columns:
        units = [str(v) for v in steps["step_concentration_unit"].dropna().unique() if str(v)]
        if units:
            unit = f" ({units[0]})"
    ax.set_xlabel("Concentration" + unit)
    ax.set_ylabel("Plateau value")
    ax.set_title(f"Langmuir reconstruction | {metric_label}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_kd_comparison(langmuir_summary: pd.DataFrame, metric_label: Optional[str] = None):
    df = langmuir_summary.copy()
    if metric_label and "metric_label" in df.columns:
        df = df[df["metric_label"] == metric_label]
    if df.empty or "langmuir_kd" not in df.columns:
        return None
    df["langmuir_kd"] = pd.to_numeric(df["langmuir_kd"], errors="coerce")
    df = df.dropna(subset=["langmuir_kd"])
    if df.empty:
        return None
    df["label"] = df["experiment_name"].astype(str) + " | Ch" + df["channel"].astype(str)
    df = df.sort_values("langmuir_kd")

    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.38 * len(df))))
    ax.barh(df["label"], df["langmuir_kd"])
    unit = ""
    units = [str(v) for v in df.get("langmuir_kd_unit", pd.Series(dtype=str)).dropna().unique() if str(v)]
    if units:
        unit = f" ({units[0]})"
    ax.set_xlabel("Kd" + unit)
    ax.set_title("Kd comparison")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig

