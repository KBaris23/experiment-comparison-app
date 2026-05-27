# Experiment Comparison App

Streamlit app for comparing exported electrochemistry analysis bundles from `swv_app`.

This app does not process raw SWV/CV files. It reads exported `manifest.json` bundles and reconstructs comparisons from CSV files.

## Expected Input

Use the main analysis app's Export tab to save an experiment output bundle:

```text
MyExperiment/
  outputs/
    20260527_143200_MyExperiment/
      manifest.json
      swv_signal_processing_inputs.csv
      swv_results.csv
      swv_titration_steps.csv
      swv_langmuir_fit_summary.csv
```

You can paste any of these paths into the sidebar:

- an experiment folder containing `outputs/`
- an `outputs/` folder
- a specific bundle folder
- a direct path to `manifest.json`

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Views

| View | Purpose |
| --- | --- |
| Overview | Loaded bundle summary and quick Kd snapshot |
| Processing Inputs | Compare signal-processing settings and detect differences |
| Scan Metrics | Reconstruct metric-vs-scan plots from `results.csv` |
| Titration | Reconstruct plateau-vs-concentration plots from `titration_steps.csv` |
| Langmuir / Kd | Compare Kd values and reconstructed Langmuir fits |
| Tables | Inspect and download combined CSV tables |

## Design Notes

- Figures are reconstructed from CSVs instead of stored images.
- `manifest.json` is the table of contents for each bundle.
- Bundle schema version is read from `manifest.json` so future schema changes can be handled explicitly.
- The app keeps loaded tables labeled with experiment name, bundle path, analysis mode, and source metadata so comparisons remain traceable.

## Repository Layout

```text
experiment_compare_app/
  app.py
  requirements.txt
  comparelib/
    io.py
    plots.py
```

