"""
RoLA (Real-time Online Lightweight Anomaly Detection) - With Ground Truth Evaluation
Author: Samuel Moses Orokpo
Date: 20th March, 2025

Features:
- Used LSTM-based anomaly detectors (LADs) for time series
- Implemented real-time dynamic thresholding
- Simplified Correlation-based anomaly confirmation
- Evaluated Precision, Recall, F-score, and Inference Time Statistics using `Anomaly_Label`
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


### Configurations

CONFIG = {
    "LOOKBACK": 10,  # LSTM input window size (10 to better understand pattern overtime)
    "HIDDEN_UNITS": 10,
    "LEARNING_RATE": 0.005,
    "EPOCHS": 10,  # Optimized for efficiency
    "BATCH_SIZE": 1,
    "SLIDING_WINDOW_SIZE": 1000,  # Used for adaptive thresholding
    "CORREL_THRESHOLD": 0.90,  # Pearson correlation threshold for anomaly confirmation
}


### LSTM-Based Lightweight Anomaly Detector (LAD)

class OnlineLSTMDetector:
    """ LSTM-based Anomaly Detector """

    def __init__(self, lookback=5, hidden_units=10, lr=0.005):
        self.lookback = lookback
        self.hidden_units = hidden_units
        self.lr = lr
        self.model = self._build_model()
        self.aare_history = []
        self.train_data = []

    def _build_model(self):
        """ Build LSTM Model """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.lookback, 1)),
            tf.keras.layers.LSTM(self.hidden_units, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(self.lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def _compute_aare(self, y_true, y_pred):
        """ Compute Average Absolute Relative Error (AARE) """
        return np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-9)

    def update_and_detect(self, new_data):
        """ Process new data, detect anomalies """
        self.train_data.append(new_data)
        anomaly_detected = False

        if len(self.train_data) >= self.lookback:
            X = np.array(self.train_data[-self.lookback:]).reshape(1, self.lookback, 1)
            prediction = self.model.predict(X, verbose=0)[0][0]
            aare_value = self._compute_aare(new_data, prediction)
            self.aare_history.append(aare_value)

            ### Dynamic thresholding using a rolling window
            if len(self.aare_history) > CONFIG["LOOKBACK"]:
                recent_errors = self.aare_history[-CONFIG["SLIDING_WINDOW_SIZE"]:]
                threshold = np.mean(recent_errors) + 3.5 * np.std(recent_errors)#(Strict to reduce FP)
                if aare_value > threshold:
                    anomaly_detected = True

        return anomaly_detected

### RoLA System

class RoLA:
    """ RoLA Multivariate Anomaly Detection """

    def __init__(self, num_variables):
        self.num_variables = num_variables
        self.lads = [OnlineLSTMDetector(lookback=CONFIG["LOOKBACK"]) for _ in range(num_variables)]
        self.data_history = [[] for _ in range(num_variables)]

    def process_data_point(self, data_point, index):
        anomalies = [self.lads[i].update_and_detect(data_point[i]) for i in range(self.num_variables)]

        ### Correlation-based confirmation
        confirmed_anomalies = anomalies.copy()
        if sum(anomalies) > 0:
            for i in range(self.num_variables):
                for j in range(self.num_variables):
                    if i != j:
                        corr = np.corrcoef(self.data_history[i][-100:], self.data_history[j][-100:])[0, 1]
                        if abs(corr) > CONFIG["CORREL_THRESHOLD"]:
                            confirmed_anomalies[i] = confirmed_anomalies[i] or anomalies[j]

        ### Logging detected anomalies
        if any(confirmed_anomalies):
            print(f"🔍Anomaly detected at index {index}: {confirmed_anomalies}")

        return confirmed_anomalies


### Evaluation Function

def evaluate_results(ground_truth, predictions, inference_times):
    """ Computing Precision, Recall, F1-score, and timing statistics """
    tp = sum((ground_truth == 1) & (predictions == 1))
    fp = sum((ground_truth == 0) & (predictions == 1))
    fn = sum((ground_truth == 1) & (predictions == 0))

    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)

    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)

    print(f"\n📝 Evaluation Metrics:")
    print(f"➡️ TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"➡️ Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
    print(f"➡️ Avg Inference Time: {avg_time:.6f}s, Std Dev: {std_time:.6f}s")


### Main Execution

def main():
    print("\n➡️ Running RoLA Anomaly Detection...")

    # Loading the dataset
    df = pd.read_csv("Labeled_Dataset_with_Anomaly_Labels.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)

    # Extracting anomaly labels
    ground_truth = df["Anomaly_Label"].values
    df.drop(columns=["Anomaly_Label"], inplace=True)  # Remove labels from input data

    # Selecting numeric data only
    data_array = df.values.astype(float)

    # Initializing RoLA system
    rola_system = RoLA(num_variables=data_array.shape[1])

    predictions = []
    inference_times = []

    # Processing each time step
    for i, row in enumerate(data_array):
        start_time = time.time()
        anomalies = rola_system.process_data_point(row, i)
        end_time = time.time()

        predictions.append(1 if any(anomalies) else 0)
        inference_times.append(end_time - start_time)

    # Evaluating performance
    evaluate_results(ground_truth, np.array(predictions), inference_times)

if __name__ == "__main__":
    main()