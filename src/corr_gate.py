import numpy as np

class CorrGate:
    def __init__(self, d, alpha=0.03, gate_corr=0.6):
        self.d = d
        self.alpha = alpha
        self.gate_corr = gate_corr
        self.mean = np.zeros(d)
        self.var = np.ones(d) * 1e-3
        self.corr = np.eye(d)

    def update(self, x):
        x = np.asarray(x, dtype=float)
        a = self.alpha
        self.mean = (1 - a) * self.mean + a * x
        r = x - self.mean
        self.var = (1 - a) * self.var + a * (r * r)
        z = r / np.sqrt(np.maximum(self.var, 1e-9))
        C = np.outer(z, z)
        self.corr = (1 - a) * self.corr + a * C
        d = np.sqrt(np.maximum(np.diag(self.corr), 1e-9))
        self.corr = self.corr / np.outer(d, d)

    def gate(self, per_sensor_flags):
        flags = np.asarray(per_sensor_flags, dtype=bool)
        k = flags.sum()
        if k == 1:
            i = int(np.where(flags)[0][0])
            if np.max(np.abs(np.delete(self.corr[i], i))) >= self.gate_corr:
                return False, "rejected_single_glitch"
        return k > 0, "ok"
