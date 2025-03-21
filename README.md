
# RoLA (Real-time Online Lightweight Anomaly Detection)
**Author:** Samuel Moses Orokpo  
**Date:** March 20, 2025  

## ** Overview**
RoLA (**Real-time Online Lightweight Anomaly Detection**) is a multivariate anomaly detection system for time-series data. It is designed for **real-time applications** in **safety-critical systems**, such as industrial monitoring, healthcare, and cybersecurity.

The system utilizes:
- **LSTM-based anomaly detectors (LADs)**
- **Dynamic thresholding for anomaly detection**
- **Correlation-based anomaly confirmation**
- **Performance evaluation with ground truth labels**

---

## ** 1️⃣ Installation**
Ensure you have the required dependencies installed:
```bash
pip install tensorflow pandas numpy scikit-learn
```

---

## ** 2️⃣ Dataset Preparation**
### ** Merging Process**
The dataset was created by merging multiple sensor readings over time. The following datasets were used:
- `C3_Temperature_10-28_10-30.csv`
- `C3_Turbidity_10-28_10-30.csv`
- `Flow_Flow_10-28_10-30.csv`
- `Flow_Temperature_10-28_10-30.csv`
- `Optode_Concentration10-28_10-30.csv`
- `Optode_Saturation_10-28_10-30.csv`
- `Optode_Temperature_10-28_10-30.csv`
- `SEB45_Conductivity_10-28_10-30.csv`
- `SEB45_Salinity_10-28_10-30.csv`

These datasets were merged using their common **timestamp** to create a structured dataset containing:
- **Temperature**
- **Turbidity**
- **Flow Rate**
- **Conductivity**
- **Salinity**
- **Oxygen Concentration**
- **Saturation levels**

### ** Preprocessing Steps**
- **Handled missing values** using:
  - Forward-fill (`ffill`)
  - Backward-fill (`bfill`)
  - Linear interpolation (`interpolate`)
- **Normalized numerical values** using **Min-Max Scaling**
- **Generated `Anomaly_Label` column** using the `3σ rule` (Z-score thresholding):
  - Data points **exceeding 3 standard deviations** were labeled as anomalies (`1`), while others remained normal (`0`).

---

## ** 3️⃣ Running the Model**
Ensure the dataset **`Labeled_Dataset_with_Anomaly_Labels.csv`** is in the same directory as the script.

Run the following command:
```bash
python rola_implementation.py
```
This will:
✅ Load the dataset  
✅ Apply real-time anomaly detection  
✅ Compute and display evaluation metrics  

---

## ** 4️⃣ Configuration Parameters**
Modify these settings inside `CONFIG` in the script:
```python
CONFIG = {
    "LOOKBACK": 10,  # LSTM input window size
    "HIDDEN_UNITS": 10,
    "LEARNING_RATE": 0.005,
    "EPOCHS": 10,  
    "BATCH_SIZE": 1,
    "SLIDING_WINDOW_SIZE": 1000,  
    "CORREL_THRESHOLD": 0.90,  
}
```
- **LOOKBACK:** Number of previous time steps the LSTM considers  
- **SLIDING_WINDOW_SIZE:** The number of recent samples used for dynamic thresholding  
- **CORREL_THRESHOLD:** The minimum correlation required for anomaly confirmation  

---

## ** 5️⃣ Understanding the Output**
The script prints detected anomalies and final evaluation metrics:
```bash
🔍 Anomaly detected at index 102: [False, False, True, False, False]
📝 Evaluation Metrics:
✅ TP: 71, FP: 168, FN: 38
✅ Precision: 0.297, Recall: 0.651, F1-score: 0.408
✅ Avg Inference Time: 0.619s, Std Dev: 0.043s
```

---

## ** 6️⃣ Key Enhancements**
 **Extended Lookback Window (5 → 10) for better pattern detection**  
 **Stricter thresholding (`3σ → 3.5σ`) to reduce false positives**  
 **Correlation-based anomaly confirmation for better robustness**  
 **Improved recall (0.651) while keeping inference time <1s**  

---

## ** 7️⃣ Future Work**
The following improvements are planned:
✅ **Hybrid Model (LSTM + Isolation Forest) to reduce false positives**  
✅ **Dynamic threshold tuning based on rolling mean trends**  
✅ **Testing on a larger dataset for improved generalization**  

---


## ** Get Started Now!**
Run the model and analyze real-time anomaly detection for **safety-critical systems like industrial monitoring, healthcare, and cybersecurity.**  
