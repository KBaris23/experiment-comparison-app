"""
Experiment Comparison App

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import subprocess
import sys
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

selected_labels = st.multiselect(
    "Experiments",
    options=labels,
    default=labels,
    help="Choose which loaded experiments are included in plots and tables.",
)
if not selected_labels:
    st.info("Select at least one experiment.")
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
    ["Overview", "Processing Inputs", "Scan Metrics", "Titration", "Langmuir / Kd", "Tables"],
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
        default_metric = "peak_current_selected" if "peak_current_selected" in metric_options else (metric_options[0] if metric_options else None)
        x_options = [col for col in ["scan_number", "filtered_source_scan_number", "original_scan_number", "measurement_index"] if col in results_df.columns]
        metric_col = st.selectbox("Metric", metric_options, index=metric_options.index(default_metric) if default_metric in metric_options else 0)
        x_col = st.selectbox("X axis", x_options, index=0) if x_options else "scan_number"
        fig = plot_metric_vs_scan(results_df, metric_col, selected_labels, channel_filter, x_col=x_col)
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
