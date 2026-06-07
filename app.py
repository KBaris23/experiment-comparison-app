"""
Experiment Comparison App

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from comparelib.io import (
    bundle_label,
    combine_table,
    discover_manifests,
    load_bundle,
    manifest_summary,
    numeric_columns,
    parse_paths,
)
from comparelib.plots import (
    plot_kd_comparison,
    plot_langmuir_curves,
    plot_metric_vs_scan,
    plot_titration_plateaus,
)
from comparelib.summary import (
    DEFAULT_CURRENT_METRICS,
    DEFAULT_VOLTAGE_METRICS,
    build_comparison_summary,
    metric_column,
    plot_path,
)


st.set_page_config(
    page_title="Experiment Comparison",
    page_icon="EC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 3.2rem; }
    div[data-testid="stSidebarContent"] { font-size: 0.86rem; }
</style>
""",
    unsafe_allow_html=True,
)


def _pick_folder_windows() -> str:
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root=tk.Tk()\n"
        "root.withdraw()\n"
        "root.wm_attributes('-topmost', True)\n"
        "p=filedialog.askdirectory(title='Select experiment folder or outputs folder')\n"
        "root.destroy()\n"
        "print(p or '')\n"
    )
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()


def append_unique(lines: str, path: str) -> str:
    existing = [line.strip() for line in lines.splitlines() if line.strip()]
    if path and path not in existing:
        existing.append(path)
    return "\n".join(existing)


def filter_by_metric_label(df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    if df.empty or "metric_label" not in df.columns:
        return df
    return df[df["metric_label"].astype(str) == str(metric_label)]


def figure_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    return buffer.getvalue()


def png_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def close_figure(fig) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def default_x_column(df: pd.DataFrame) -> str:
    for col in ["scan_number", "filtered_source_scan_number", "original_scan_number", "measurement_index"]:
        if col in df.columns:
            return col
    return ""


SCAN_METRIC_EXCLUDE_KEYWORDS = (
    "timestamp",
    "created",
    "date",
    "time_s",
    "time_sec",
    "elapsed",
    "swv_",
    "method_",
    "start",
    "end",
    "step_size",
    "step_interval",
    "sample_interval",
    "quiet",
    "duration",
    "nscans",
    "parameter",
    "setting",
)


def is_scan_metric_column(col: str) -> bool:
    lower = col.lower()
    if any(keyword in lower for keyword in SCAN_METRIC_EXCLUDE_KEYWORDS):
        return False
    return True


def is_voltage_metric(col: str) -> bool:
    lower = col.lower()
    return "voltage" in lower or "potential" in lower


def _parse_vline_payload(value) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                return [{"scan": float(text), "label": ""}]
            except ValueError:
                return []
        return _parse_vline_payload(parsed)
    if isinstance(value, dict):
        scan = value.get("scan", value.get("x", value.get("scan_number")))
        numeric = pd.to_numeric(pd.Series([scan]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return []
        return [{"scan": float(numeric), "label": str(value.get("label") or "").strip()}]
    if isinstance(value, (list, tuple)):
        markers = []
        for item in value:
            markers.extend(_parse_vline_payload(item))
        return markers
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [{"scan": float(value), "label": ""}]
    return []


def vline_markers_for_labels(summary_df: pd.DataFrame, inputs_df: pd.DataFrame, labels: List[str]) -> list[dict]:
    markers = []
    for df in [summary_df, inputs_df]:
        if df.empty or "analysis_vlines_json" not in df.columns:
            continue
        selected = df[df["experiment_label"].isin(labels)] if "experiment_label" in df.columns else df
        for raw in selected["analysis_vlines_json"].dropna().tolist():
            markers.extend(_parse_vline_payload(raw))

    deduped = []
    seen = set()
    for marker in sorted(markers, key=lambda item: (item["scan"], item.get("label", ""))):
        key = (round(marker["scan"], 10), marker.get("label", ""))
        if key not in seen:
            deduped.append(marker)
            seen.add(key)
    return deduped


def build_plot_assets(
    selected_summary: pd.DataFrame,
    results_df: pd.DataFrame,
    inputs_df: pd.DataFrame,
    channel_filter: List[int],
    current_metric: str,
    voltage_metric: str,
) -> dict[str, bytes]:
    assets = {}
    x_col = default_x_column(results_df)
    if not x_col:
        return assets

    for _, row in selected_summary.iterrows():
        label = row.get("experiment_label")
        plot_key = label or row.get("experiment_name")
        vlines = vline_markers_for_labels(selected_summary, inputs_df, [label])

        if current_metric:
            fig = plot_metric_vs_scan(
                results_df,
                current_metric,
                [label],
                channel_filter,
                x_col=x_col,
                legend_style="channel",
                reference_x_values=vlines,
            )
            if fig:
                assets[plot_path(plot_key, "current_peak_height")] = figure_png_bytes(fig)
                close_figure(fig)

        if voltage_metric:
            fig = plot_metric_vs_scan(
                results_df,
                voltage_metric,
                [label],
                channel_filter,
                x_col=x_col,
                legend_style="channel",
                reference_x_values=vlines,
            )
            if fig:
                assets[plot_path(plot_key, "peak_voltage_drift")] = figure_png_bytes(fig)
                close_figure(fig)

    return assets


def display_plot_markers(comparison_df: pd.DataFrame, plot_assets: dict[str, bytes]) -> pd.DataFrame:
    out = comparison_df.copy()
    for col in ["Current Peak Height Plot", "Peak Voltage Drift Plot"]:
        if col not in out.columns:
            continue
        out[col] = out[col].apply(lambda value: "[embedded PNG]" if value in plot_assets else "")
    return out


def build_navigation_pack(comparison_df: pd.DataFrame, plot_assets: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("comparison_summary.csv", comparison_df.to_csv(index=False))
        for path, png in plot_assets.items():
            zf.writestr(path, png)
    return buffer.getvalue()


def build_html_report(comparison_df: pd.DataFrame, plot_assets: dict[str, bytes]) -> bytes:
    plot_cols = {"Current Peak Height Plot", "Peak Voltage Drift Plot"}
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Experiment Comparison</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;color:#1f2933}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "th,td{border:1px solid #cbd5e1;padding:6px;vertical-align:top}",
        "th{background:#276749;color:white;position:sticky;top:0}",
        "img{max-width:420px;height:auto}",
        "</style></head><body>",
        "<h1>Experiment Comparison</h1>",
        "<table><thead><tr>",
    ]
    for col in comparison_df.columns:
        parts.append(f"<th>{html.escape(str(col))}</th>")
    parts.append("</tr></thead><tbody>")

    for _, row in comparison_df.iterrows():
        parts.append("<tr>")
        for col in comparison_df.columns:
            value = row.get(col, "")
            if col in plot_cols and value in plot_assets:
                parts.append(f"<td><img src='{png_data_uri(plot_assets[value])}' alt='{html.escape(str(col))}'></td>")
            else:
                parts.append(f"<td>{html.escape(str(value))}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></body></html>")
    return "\n".join(parts).encode("utf-8")


@st.cache_data(show_spinner=False)
def load_bundles_cached(manifest_paths: List[str]):
    return [load_bundle(Path(path)) for path in manifest_paths]


if "input_paths" not in st.session_state:
    st.session_state.input_paths = ""


with st.sidebar:
    st.title("Experiment Comparison")
    st.caption("Load exported experiment output bundles and compare reconstructed metrics.")

    if st.button("Browse folder", disabled=not sys.platform.startswith("win"), use_container_width=True):
        try:
            picked = _pick_folder_windows()
            if picked:
                st.session_state.input_paths = append_unique(st.session_state.input_paths, picked)
        except Exception as e:
            st.error(f"Folder picker failed: {e}")

    input_paths = st.text_area(
        "Experiment/output folders",
        value=st.session_state.input_paths,
        height=150,
        help=(
            "Paste one path per line. You can use an experiment folder, an outputs folder, "
            "a specific output bundle folder, or a manifest.json file."
        ),
    )
    st.session_state.input_paths = input_paths
    search_paths = parse_paths(input_paths)
    missing_paths = [str(path) for path in search_paths if not path.exists()]
    for path in missing_paths:
        st.warning(f"Not found: {path}")

    manifest_paths = discover_manifests(search_paths)
    st.caption(f"Discovered {len(manifest_paths)} output bundle(s).")

    if st.button("Clear paths", use_container_width=True):
        st.session_state.input_paths = ""
        st.rerun()


if not manifest_paths:
    st.info("Add one or more exported experiment folders to begin.")
    st.stop()

with st.spinner("Loading experiment bundles..."):
    bundles = load_bundles_cached([str(path) for path in manifest_paths])

labels = [bundle_label(bundle) for bundle in bundles]
if not labels:
    st.warning("No readable experiment bundles were found.")
    st.stop()

output_scope = st.selectbox(
    "Output selection",
    options=["Compare all discovered outputs", "Choose specific outputs"],
    help="Use all loaded output bundles for the comparison summary, or narrow the active set.",
)
if output_scope == "Choose specific outputs":
    selected_labels = st.multiselect(
        "Outputs",
        options=labels,
        default=labels,
        help="Choose which loaded output bundles are included in plots and tables.",
    )
else:
    selected_labels = labels
    st.caption("Comparing all discovered output bundles. Plot filenames use the full output label so matching experiment names do not overwrite each other.")

if not selected_labels:
    st.info("Select at least one output bundle.")
    st.stop()

summary_df = manifest_summary(bundles)
selected_summary = summary_df[summary_df["experiment_label"].isin(selected_labels)]

results_df = combine_table(bundles, "results", selected_labels)
titration_df = combine_table(bundles, "titration_steps", selected_labels)
langmuir_df = combine_table(bundles, "langmuir_fit_summary", selected_labels)
inputs_df = combine_table(bundles, "signal_processing_inputs", selected_labels)

analysis_modes = sorted(selected_summary["analysis_mode"].dropna().astype(str).unique())
channels = []
if not results_df.empty and "channel" in results_df.columns:
    channels = sorted(pd.to_numeric(results_df["channel"], errors="coerce").dropna().astype(int).unique())

top_cols = st.columns(4)
top_cols[0].metric("Experiments", len(selected_labels))
top_cols[1].metric("Rows", len(results_df))
top_cols[2].metric("Channels", len(channels))
top_cols[3].metric("Modes", ", ".join(analysis_modes) if analysis_modes else "-")

if channels:
    channel_filter = st.multiselect("Channels", options=channels, default=channels)
else:
    channel_filter = []

view = st.radio(
    "View",
    ["Overview", "Comparison Index", "Processing Inputs", "Scan Metrics", "Titration", "Langmuir / Kd", "Tables"],
    horizontal=True,
)


if view == "Overview":
    st.subheader("Loaded Bundles")
    st.dataframe(selected_summary, use_container_width=True, height=260)

    if not langmuir_df.empty and "langmuir_kd" in langmuir_df.columns:
        st.markdown("#### Kd Snapshot")
        fig = plot_kd_comparison(langmuir_df)
        if fig:
            st.pyplot(fig)


elif view == "Comparison Index":
    st.subheader("Comparison Index")
    if results_df.empty:
        st.info("No results table was found, so the comparison index cannot calculate peak summaries yet.")
    else:
        current_options = [
            col for col in numeric_columns(results_df)
            if "peak" in col.lower() or "current" in col.lower()
        ]
        detected_current = metric_column(results_df, DEFAULT_CURRENT_METRICS)
        if detected_current and detected_current not in current_options:
            current_options.insert(0, detected_current)

        voltage_options = [
            col for col in numeric_columns(results_df)
            if "voltage" in col.lower() or "potential" in col.lower()
        ]
        detected_voltage = metric_column(results_df, DEFAULT_VOLTAGE_METRICS)
        if detected_voltage and detected_voltage not in voltage_options:
            voltage_options.insert(0, detected_voltage)

        if not current_options:
            st.warning("No numeric peak-current style columns were found in the results table.")
            current_metric = ""
        else:
            default_index = current_options.index(detected_current) if detected_current in current_options else 0
            current_metric = st.selectbox("Peak height metric", current_options, index=default_index)

        voltage_choices = [""] + voltage_options
        default_voltage_index = voltage_choices.index(detected_voltage) if detected_voltage in voltage_choices else 0
        voltage_metric = st.selectbox(
            "Peak voltage drift metric",
            voltage_choices,
            index=default_voltage_index,
            format_func=lambda value: "None available" if value == "" else value,
        )

        comparison_df = build_comparison_summary(
            selected_summary,
            results_df,
            inputs_df,
            langmuir_df,
            current_metric=current_metric or None,
            voltage_metric=voltage_metric or None,
        )
        plot_assets = build_plot_assets(
            selected_summary,
            results_df,
            inputs_df,
            channel_filter,
            current_metric,
            voltage_metric,
        )
        st.dataframe(display_plot_markers(comparison_df, plot_assets), use_container_width=True, height=360)

        if not comparison_df.empty:
            csv_bytes = comparison_df.to_csv(index=False).encode()
            cols = st.columns(3)
            cols[0].download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="comparison_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            cols[1].download_button(
                "Download ZIP + plots",
                data=build_navigation_pack(comparison_df, plot_assets),
                file_name="comparison_navigation_pack.zip",
                mime="application/zip",
                use_container_width=True,
            )
            cols[2].download_button(
                "Download HTML report",
                data=build_html_report(comparison_df, plot_assets),
                file_name="comparison_report.html",
                mime="text/html",
                use_container_width=True,
            )


elif view == "Processing Inputs":
    st.subheader("Signal Processing Inputs")
    if inputs_df.empty:
        st.info("No signal_processing_inputs table was found.")
    else:
        st.dataframe(inputs_df, use_container_width=True, height=300)
        compare_cols = [
            col for col in inputs_df.columns
            if col not in {"bundle_dir", "analysis_vlines_json"}
            and inputs_df[col].astype(str).nunique(dropna=False) > 1
        ]
        if compare_cols:
            st.markdown("#### Settings that differ")
            st.dataframe(inputs_df[["experiment_label", *compare_cols]], use_container_width=True, height=260)
        else:
            st.caption("No differing settings were detected among the selected experiments.")


elif view == "Scan Metrics":
    st.subheader("Metric vs Scan")
    if results_df.empty:
        st.info("No results table was found.")
    else:
        metric_options = numeric_columns(
            results_df,
            exclude={
                "channel", "scan_number", "original_scan_number", "filtered_source_scan_number",
                "frequency_hz", "measurement_index", "cycle_count_in_file", "method_nscans",
            },
        )
        metric_options = [col for col in metric_options if is_scan_metric_column(col)]
        default_metric = "peak_current_selected" if "peak_current_selected" in metric_options else (metric_options[0] if metric_options else None)
        x_options = [col for col in ["scan_number", "filtered_source_scan_number", "original_scan_number", "measurement_index"] if col in results_df.columns]
        if not metric_options:
            st.info("No scan-varying metric columns were found after filtering out scalar signal parameters.")
        else:
            metric_col = st.selectbox("Metric", metric_options, index=metric_options.index(default_metric) if default_metric in metric_options else 0)
            x_col = st.selectbox("X axis", x_options, index=0) if x_options else "scan_number"
            legend_style = "channel" if len(selected_labels) == 1 else "experiment_channel"
            reference_x_values = vline_markers_for_labels(selected_summary, inputs_df, selected_labels)
            fig = plot_metric_vs_scan(
                results_df,
                metric_col,
                selected_labels,
                channel_filter,
                x_col=x_col,
                legend_style=legend_style,
                reference_x_values=reference_x_values,
            )
            if fig:
                st.pyplot(fig)
            else:
                st.info("No plottable rows match the selected metric/channel filters.")


elif view == "Titration":
    st.subheader("Titration Plateaus")
    if titration_df.empty:
        st.info("No titration_steps table was found for the selected experiments.")
    else:
        metric_labels = sorted(titration_df.get("metric_label", pd.Series(["plateau"])).dropna().astype(str).unique())
        metric_label = st.selectbox("Titration metric", metric_labels)
        fig = plot_titration_plateaus(titration_df, metric_label, selected_labels, channel_filter)
        if fig:
            st.pyplot(fig)
        st.dataframe(filter_by_metric_label(titration_df, metric_label), use_container_width=True, height=260)


elif view == "Langmuir / Kd":
    st.subheader("Langmuir Fits and Kd")
    if langmuir_df.empty:
        st.info("No langmuir_fit_summary table was found for the selected experiments.")
    else:
        metric_labels = sorted(langmuir_df.get("metric_label", pd.Series(["fit"])).dropna().astype(str).unique())
        metric_label = st.selectbox("Langmuir metric", metric_labels)
        kd_fig = plot_kd_comparison(langmuir_df, metric_label=metric_label)
        if kd_fig:
            st.pyplot(kd_fig)
        if not titration_df.empty:
            curve_fig = plot_langmuir_curves(titration_df, langmuir_df, metric_label, selected_labels, channel_filter)
            if curve_fig:
                st.pyplot(curve_fig)
        filtered = filter_by_metric_label(langmuir_df, metric_label)
        st.dataframe(filtered, use_container_width=True, height=300)


elif view == "Tables":
    st.subheader("Raw Loaded Tables")
    table_name = st.selectbox(
        "Table",
        ["manifest_summary", "signal_processing_inputs", "results", "titration_steps", "langmuir_fit_summary"],
    )
    table_map = {
        "manifest_summary": selected_summary,
        "signal_processing_inputs": inputs_df,
        "results": results_df,
        "titration_steps": titration_df,
        "langmuir_fit_summary": langmuir_df,
    }
    table = table_map[table_name]
    st.dataframe(table, use_container_width=True, height=420)
    if not table.empty:
        st.download_button(
            f"Download combined_{table_name}.csv",
            data=table.to_csv(index=False).encode(),
            file_name=f"combined_{table_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )
