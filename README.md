# RoLA v2 — Lightweight Real-Time Online Anomaly Detection (Live Demo)

This repository contains an improved [RoLA variant](https://github.com/freelansire/RoLA-Anomaly-Detection/tree/v1-maintenance) (“RoLA v2”) for **real-time multivariate anomaly detection** in non-stationary sensor streams, with a **live streaming dashboard** (Streamlit) for reproducible experiments.

---

## Overview (RoLA v2)

- Introduced an improved RoLA variant with **uncertainty-aware Bayesian online thresholding**, **adaptive correlation gates**, and **temporal recalibration (forgetting)** for non-stationary streams.
- Demonstrated robust performance on **multivariate marine sensor data** *(temperature, turbidity, oxygen, salinity)*, emphasizing **latency–accuracy trade-offs** under resource constraints.
- Delivered a **live streaming demo** with explainable anomaly scores and confidence intervals; designed for **edge-friendly deployment** and **reproducible benchmarking** for low-power inference experiments.
- **Independent Study (Reviewed by NTNU Faculty – Highest Ranked).** Formed basis for ongoing manuscript on empirical hybrid extensions.

---

## Live Demo (Streamlit)

### Features

- **Three data modes**
  - **Synthetic marine stream** (reproducible anomalies — kept unchanged by design)
  - **USGS realtime (US)** (river telemetry)
  - **UK Hydrology (EA/Defra)** (UK stations via measures API)
- **Online detection components**
  - Bayesian predictive scoring (online updates)
  - Conformal thresholding (adaptive threshold, streaming-safe)
  - Drift detection (Page–Hinkley)
  - Correlation gating (cross-sensor agreement check)
- **Exports**
  - Download full run **history CSV**
  - Download flagged **anomalies CSV**
  - Optional autosave to `outputs/<run_id>/`

---
## Install dependencies
pip install -r requirements.txt

## Dashboard Usage
## How to use the dashboard

### Synthetic mode (recommended first)
1. Select **Source → Synthetic**
2. Choose **stream length** + **seed**
3. Click **Start / Resume**
4. Export results using the **download buttons**

### USGS realtime (US)
1. Select **Source → USGS**
2. Enter a **USGS Site ID**
3. Pick **Lookback** (**P1D / P7D / P30D**)
4. Choose parameter codes *(temperature / flow / conductivity / oxygen, etc.)*
5. Use the **coverage report** + **feature auto-select** when some variables are missing

### UK Hydrology (EA/Defra)
1. Select **Source → UK Hydrology**
2. Search and pick a **station**
3. To discover richer stations, adjust:
   - **Station filter:** `waterLevel`, `waterFlow`, `temperature`, `turbidity`, `conductivity`, `dissolved-oxygen`
   - **Measures periodName:** set to **Any** *(important)*
4. Choose available **measures** *(the UI only shows what the station actually provides)*
5. Use **Align to grid** to resample to a common interval if timestamps differ

> **Note:** Many UK stations are **level/flow-only**. For water-quality signals *(temp/turbidity/DO/conductivity)*, use station filters and broaden **periodName**.

## Outputs

You can export directly from the UI, or enable autosave:

- `rola_v2_history.csv` — streaming step-by-step records *(scores, thresholds, drift flags, sensor values)*
- `rola_v2_anomalies.csv` — only flagged events + context

If autosave is enabled, the app writes:

outputs/<run_id>/history.csv 
outputs/<run_id>/anomalies.csv

## Quickstart

### 1) Create environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate


## Citation / Attribution

```bibtex
@software{orokpo_rola_v2,
  author  = {Samuel Moses Orokpo},
  title   = {RoLA v2: Lightweight Real-Time Online Anomaly Detection (Live Streaming Demo)},
  year    = {2025},
  url     = {https://github.com/freelansire/<YOUR_REPO_NAME>}
}

