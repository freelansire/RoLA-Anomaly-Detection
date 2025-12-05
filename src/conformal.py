import numpy as np
from collections import deque

class ConformalThreshold:
    """
    Rolling conformal calibrator on anomaly scores (NLL).
    Threshold = quantile(scores, 1-alpha).
    """
    def __init__(self, alpha=0.01, window=256, warmup=64):
        self.alpha = float(alpha)
        self.window = int(window)
        self.warmup = int(warmup)
        self.buf = deque(maxlen=self.window)

    def update(self, score):
        self.buf.append(float(score))

    def ready(self):
        return len(self.buf) >= self.warmup

    def threshold(self):
        if not self.ready():
            return float("inf")
        arr = np.asarray(self.buf, dtype=float)
        q = np.quantile(arr, 1.0 - self.alpha)
        return float(q)
