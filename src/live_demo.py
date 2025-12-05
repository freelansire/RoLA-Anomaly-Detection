import numpy as np
import matplotlib.pyplot as plt

from marine_sim import make_stream
from mv_bayes_niw import OnlineNIW
from conformal import ConformalThreshold
from drift import PageHinkley
from corr_gate import CorrGate

def run_live(alpha=0.01, forget=0.995):
    df = make_stream(n=1800)
    features = ["temperature", "turbidity", "oxygen", "salinity"]
    X = df[features].values
    y = df["label"].values

    d = X.shape[1]
    model = OnlineNIW(d=d, mu0=X[0], forget=forget)
    calib = ConformalThreshold(alpha=alpha, window=320, warmup=80)
    drift = PageHinkley(delta=0.01, lamb=6.0)
    gate = CorrGate(d=d, gate_corr=0.6)

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))

    xs, nlls, ths = [], [], []
    anom_x, anom_y = [], []
    drift_x = []

    for i in range(len(X)):
        x = X[i]
        gate.update(x)

        # Per-sensor quick flags (cheap): z-score-ish via running corr state isn't per-sensor here;
        # We'll approximate per-sensor flags via deviations from model mean after update params are stable.
        # For demo: use simple absolute residual vs predicted mean thresholds.
        # The real decision uses multivariate NLL + conformal threshold + correlation gate.
        nll, _ = model.score(x)

        # Update online calibration BEFORE classification? In practice, you calibrate on presumed-normal region.
        # Here: update calibration when not clearly anomalous (robust-ish).
        # We'll check against last threshold if ready.
        th = calib.threshold()
        is_raw_anom = (nll > th) if calib.ready() else False

        # gate: if only one sensor "fires" (approx), reject glitches
        # approximate per-sensor: compare abs residual to running scale proxy using model predictive Sigma diag
        dfree, mu, Sigma = model.predictive_params()
        resid = np.abs(x - mu)
        scale = np.sqrt(np.maximum(np.diag(Sigma), 1e-9))
        sensor_flags = resid > (3.2 * scale)
        gated_any, reason = gate.gate(sensor_flags)

        # final decision combines: (a) conformal NLL rule, (b) correlation gate sanity check
        is_anom = bool(is_raw_anom and (gated_any or sensor_flags.sum() > 1))

        # Drift detection on NLL
        if drift.update(nll):
            drift_x.append(i)

        # Update model always (online learning)
        model.update(x)

        # Update calibrator mainly on presumed normal points
        if not is_anom:
            calib.update(nll)

        xs.append(i)
        nlls.append(nll)
        ths.append(calib.threshold())

        ax.clear()
        ax.plot(xs, nlls, label="NLL (Bayesian multivariate t)")
        ax.plot(xs, ths, label=f"Conformal threshold (alpha={alpha})")

        if is_anom:
            anom_x.append(i)
            anom_y.append(nll)
            ax.scatter(anom_x, anom_y, marker="x", label="Anomaly")

        if drift_x:
            ax.scatter(drift_x, [max(nlls)*0.95]*len(drift_x), marker="|", label="Drift (PH)")

        ax.set_title(f"RoLA V2 Demo: Bayesian+Conformal+Drift | forget={forget} | gate={reason}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Anomaly score (NLL)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_live()
