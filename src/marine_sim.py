import numpy as np
import pandas as pd

def make_stream(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    latent = np.sin(t / 60.0) + 0.6 * np.sin(t / 19.0)

    temp = 15 + 2.0 * latent + rng.normal(0, 0.25, n)
    turb = 5 + 1.1 * latent + rng.normal(0, 0.20, n)
    oxy  = 90 + 4.0 * latent + rng.normal(0, 0.50, n)
    sal  = 33 + 0.8 * latent + rng.normal(0, 0.15, n)

    df = pd.DataFrame({
        "t": t,
        "temperature": temp,
        "turbidity": turb,
        "oxygen": oxy,
        "salinity": sal,
    })

    y = np.zeros(n, dtype=int)

    # True multivariate events (apply only if inside length)
    for idx in [450, 980, 1500]:
        if idx < n:
            end = min(idx + 12, n - 1)
            df.loc[idx:end, ["temperature","turbidity"]] += rng.normal(2.4, 0.5)
            df.loc[idx:end, ["oxygen"]] -= rng.normal(4.2, 1.0)
            y[idx:end+1] = 1

    # Single-sensor glitches (apply only if inside length)
    for idx in [650, 1280]:
        if idx < n:
            df.loc[idx, "turbidity"] += rng.normal(8.0, 1.0)
            y[idx] = 1

    df["label"] = y
    return df
