import numpy as np
from dataclasses import dataclass
from numpy.linalg import cholesky, solve

def _mv_t_logpdf(x, mu, Sigma, df):
    """
    Multivariate Student-t logpdf with location mu, scale Sigma, dof df.
    Sigma must be SPD. Uses Cholesky for stability.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    d = x.shape[0]
    L = cholesky(Sigma)  # Sigma = L L^T
    y = solve(L, (x - mu))          # L y = (x-mu)
    quad = float(y.T @ y)
    logdet = 2.0 * float(np.log(np.diag(L)).sum())

    # log normalization constant
    from scipy.special import gammaln
    logC = (
        gammaln((df + d) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * (d * np.log(df * np.pi) + logdet)
    )
    return logC - 0.5 * (df + d) * np.log1p(quad / df)

@dataclass
class OnlineNIW:
    """
    Online Normal-Inverse-Wishart for multivariate Gaussian with unknown mean/cov,
    with exponential forgetting (temporal recalibration).

    Predictive: multivariate Student-t.
    """
    d: int
    mu0: np.ndarray
    kappa0: float = 1.0
    nu0: float = None             # must be > d-1; set default d+2
    Psi0: np.ndarray = None       # scale matrix
    forget: float = 0.995         # 0.98..0.999 typical

    def __post_init__(self):
        self.mu = self.mu0.astype(float).copy()
        self.kappa = float(self.kappa0)
        self.nu = float(self.nu0 if self.nu0 is not None else (self.d + 2.0))
        if self.Psi0 is None:
            self.Psi = np.eye(self.d) * 1.0
        else:
            self.Psi = np.asarray(self.Psi0, dtype=float).copy()

    def predictive_params(self):
        # Student-t predictive parameters for NIW
        df = self.nu - self.d + 1.0
        df = max(df, 3.0)
        Sigma = (self.Psi * (self.kappa + 1.0)) / (self.kappa * df)
        return df, self.mu, Sigma

    def score(self, x):
        df, mu, Sigma = self.predictive_params()
        ll = _mv_t_logpdf(np.asarray(x), mu, Sigma, df)
        nll = -float(ll)
        return nll, float(df)

    def update(self, x):
        x = np.asarray(x, dtype=float)
        f = self.forget

        # Apply forgetting to "strength" of prior/posterior
        kappa = f * self.kappa
        nu = f * self.nu
        Psi = f * self.Psi
        mu = self.mu

        # NIW posterior update with one observation
        kappa_new = kappa + 1.0
        nu_new = nu + 1.0
        mu_new = (kappa * mu + x) / kappa_new

        # rank-1 update on Psi
        dx = (x - mu).reshape(-1, 1)
        Psi_new = Psi + (kappa / kappa_new) * (dx @ dx.T)

        self.kappa, self.nu, self.mu, self.Psi = kappa_new, nu_new, mu_new, Psi_new
