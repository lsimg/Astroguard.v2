# AstroGuard

**Universal autonomous AI for satellite reaction wheel prognostics.**

<img src="logo/actroguardlogo.png" alt="logo" width="600">

AstroGuard is an edge AI system that predicts bearing failures in satellite reaction wheels before they happen. It auto-detects the input data format, builds its own neural architecture, trains on healthy data only, and forecasts remaining useful life — all without manual configuration.

Validated on two NASA datasets: IMS bearings (7-day advance warning) and SMAP/MSL satellite telemetry (real-time anomaly detection).

## What's New in v2 (vs v1)

v1 was a proof of concept. v2 is a fundamentally different system.

| | v1 | v2 |
|---|---|---|
| **Framework** | scikit-learn MLPRegressor (not a real autoencoder) | PyTorch symmetric autoencoder |
| **Architecture** | Hardcoded `(8, 4)` layers | AutoBuilder searches 54 candidates and picks the best |
| **Data formats** | NASA IMS only | Any format: IMS, SMAP/MSL, CSV, NumPy (auto-detected) |
| **Features** | RMS only (1 feature per bearing) | RMS + Peak + Kurtosis + Std (4 features per channel) |
| **Anomaly threshold** | `max × 2` (heuristic) | μ + 3σ (statistically grounded, 99.7% confidence) |
| **RUL prediction** | Linear extrapolation (inaccurate) | Exponential degradation curve fitting |
| **FFT verification** | Basic spectrum plot | BPFO + BPFI fault frequency detection with 5× median threshold |
| **Validation** | 1 dataset (IMS) | 2 datasets (IMS + SMAP/MSL satellite telemetry) |
| **Adaptability** | Requires manual code changes for new data | Zero-configuration: point to any folder and run |

The core innovation of v2 is **universality**: the system adapts to the data, not the other way around.

## Key Features

- **Universal data adapter** — automatically detects IMS, SMAP/MSL, CSV, and NumPy formats
- **AutoBuilder (NAS)** — searches up to 54 neural architectures and selects the optimal one for each dataset
- **Unsupervised learning** — trains only on healthy data, no labeled failures required
- **FFT fault verification** — spectral analysis confirms physical defect frequencies (BPFO/BPFI)
- **Exponential RUL prediction** — fits degradation curves to forecast exact failure time
- **Edge-native** — lightweight enough for onboard satellite processors (<20ms inference)

## How It Works

```
Any telemetry data → Auto-detect format → AutoBuilder selects architecture
→ Train on healthy baseline → Monitor → Detect anomalies → Predict RUL
```

1. Point the system to any telemetry folder
2. AstroGuard detects the format and extracts features automatically
3. AutoBuilder evaluates candidate neural architectures and picks the best
4. The model trains on healthy data and learns what "normal" looks like
5. During monitoring, deviations trigger alerts with FFT physical verification
6. An exponential degradation model predicts when failure will occur

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter the path to your data folder in the sidebar and press **INITIATE MISSION**.

## Supported Data Formats

| Format | Structure | Auto-detected by |
|--------|-----------|------------------|
| NASA IMS | Folder of timestamped tab-separated files | Filename pattern `YYYY.MM.DD.HH.MM.SS` |
| NASA SMAP/MSL | `train/` + `test/` folders with `.npy` files | Directory structure |
| Generic CSV | Folder of `.csv` files with numeric columns | File extension |
| Generic NumPy | Folder of `.npy` arrays | File extension |

## Results

| Method | Lead Time | AUROC | Edge-Compatible | Universal |
|--------|-----------|-------|-----------------|-----------|
| Kurtosis (classical) | 4.2 days | 0.85 | Yes | No |
| SVM (supervised) | 5.1 days | 0.92 | No | No |
| GCN-LSTM (deep) | ~5 days | 0.94 | No | No |
| **AstroGuard** | **7.0 days** | **0.97** | **Yes** | **Yes** |

## Tech Stack

- **PyTorch** — autoencoder model
- **Streamlit** — real-time monitoring dashboard
- **SciPy** — FFT analysis and exponential curve fitting
- **scikit-learn** — data preprocessing

## Project Structure

```
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── IMS/                   # NASA IMS dataset (not included, download separately)
└── data/                  # SMAP/MSL dataset (not included, download separately)
    ├── train/
    ├── test/
    └── labeled_anomalies.csv
```

## Datasets

- **NASA IMS Bearing Dataset**: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- **NASA SMAP/MSL**: [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

## Author

**Mukhtar Gulsim** — NIS Taldykorgan, 2026

## License

MIT
