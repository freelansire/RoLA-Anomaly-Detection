class PageHinkley:
    """
    Detects mean shifts in a streaming statistic (here: NLL).
    """
    def __init__(self, delta=0.005, lamb=5.0):
        self.delta = float(delta)
        self.lamb = float(lamb)
        self.mean = 0.0
        self.t = 0
        self.ph = 0.0
        self.min_ph = 0.0
        self.triggered = False

    def update(self, x):
        self.t += 1
        # running mean
        self.mean += (x - self.mean) / self.t
        # cumulative deviation
        self.ph += (x - self.mean - self.delta)
        self.min_ph = min(self.min_ph, self.ph)
        # change if deviation exceeds threshold
        if (self.ph - self.min_ph) > self.lamb:
            self.triggered = True
        return self.triggered
