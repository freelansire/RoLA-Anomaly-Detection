import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support

from marine_sim import make_stream
from mv_bayes_niw import OnlineNIW
from conformal import ConformalThreshold
from corr_gate import CorrGate

def eval_all(alpha=0.01, forget=0.995):
    df = make_stream(n=1800)
    features = ["temperature", "turbidity", "oxygen", "salinity"]
    X = df[features].values
    y = df["label"].values.astype(int)
    d = X.shape[1]

    # Baseline
    iso = IsolationForest(contamination=0.02, random_state=42)
    y_iso = (iso.fit_predict(X) == -1).astype(int)

    # RoLA V2
    model = OnlineNIW(d=d, mu0=X[0], forget=forget)
    gate = CorrGate(d=d, gate_corr=0.6)
    calib = ConformalThreshold(alpha=alpha, window=320, warmup=80)

    y_pred = []
    for i in range(len(X)):
        x = X[i]
        gate.update(x)

        nll, _ = model.score(x)
        th = calib.threshold()
        is_raw_anom = (nll > th) if calib.ready() else False

        dfree, mu, Sigma = model.predictive_params()
        resid = np.abs(x - mu)
        scale = np.sqrt(np.maximum(np.diag(Sigma), 1e-9))
        sensor_flags = resid > (3.2 * scale)
        gated_any, _ = gate.gate(sensor_flags)

        is_anom = int(is_raw_anom and (gated_any or sensor_flags.sum() > 1))
        y_pred.append(is_anom)

        model.update(x)
        if not is_anom:
            calib.update(nll)

    y_pred = np.array(y_pred, dtype=int)

    for name, yp in [("IsolationForest", y_iso), ("RoLA-V2", y_pred)]:
        p, r, f1, _ = precision_recall_fscore_support(y, yp, average="binary", zero_division=0)
        print(f"{name:14s} | Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

if __name__ == "__main__":
    eval_all()
