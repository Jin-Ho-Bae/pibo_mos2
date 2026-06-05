"""
GP surrogate for PIBO.

Two features the PIBO prompt explicitly asks for:

  1. **Sparse approximation** for >100 observations (FITC-lite via M inducing
     points). Activated automatically when `n > sparse_threshold`.
  2. **Hyperparameter priors** on length-scales / signal variance / noise
     variance, with maximum-a-posteriori (MAP) fit via SciPy `minimize`. The
     priors are **lognormal** on positive hyperparameters — this is the
     standard weakly-informative prior used in GPy / GPyTorch / BoTorch and is
     what makes the GP robust on the sparse-data regime ReaxFF datasets sit in.

The two features compose: sparse mode uses the same MAP-fitted hyperparameters
to define Kmm/Knm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _matern_kernel(X1: np.ndarray, X2: np.ndarray, ls: np.ndarray,
                   sigma2: float, nu: float = 2.5) -> np.ndarray:
    """ARD Matern kernel (nu in {0.5, 1.5, 2.5}; defaults to 2.5)."""
    diff = (X1[:, None, :] - X2[None, :, :]) / ls
    d = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    if nu == 0.5:
        return sigma2 * np.exp(-d)
    if nu == 1.5:
        s3 = np.sqrt(3.0) * d
        return sigma2 * (1 + s3) * np.exp(-s3)
    s5 = np.sqrt(5.0) * d
    return sigma2 * (1 + s5 + (5.0 / 3.0) * d * d) * np.exp(-s5)


@dataclass
class GPModel:
    X: np.ndarray
    y: np.ndarray
    length_scales: np.ndarray
    signal_var: float
    noise_var: float
    L: np.ndarray
    alpha: np.ndarray
    inducing: np.ndarray | None = None


@dataclass
class GPPriors:
    """Weakly-informative log-normal priors on positive hyperparameters.

    For a parameter t > 0, log t ~ Normal(mu, sigma^2). Larger sigma means a
    flatter, less informative prior; the defaults below were chosen to be
    consistent with BoTorch's default `MaternKernel` priors.
    """
    ls_mu: float = 0.0       # log length-scale mean
    ls_sigma: float = 1.0
    sig_mu: float = 0.0
    sig_sigma: float = 1.0
    noise_mu: float = -4.0   # encourage small noise variance
    noise_sigma: float = 1.0

    def neg_log_prior(self, log_ls: np.ndarray,
                      log_sig2: float, log_noise: float) -> float:
        def nlp(x, mu, s):
            return 0.5 * ((x - mu) / s) ** 2 + np.log(s)
        ls_term = np.sum(nlp(log_ls, self.ls_mu, self.ls_sigma))
        return float(ls_term
                     + nlp(log_sig2, self.sig_mu, self.sig_sigma)
                     + nlp(log_noise, self.noise_mu, self.noise_sigma))


class SparseGP:
    """ARD Matern GP with hyperparameter priors and optional FITC sparse mode."""

    def __init__(self,
                 nu: float = 2.5,
                 ard: bool = True,
                 sparse_threshold: int = 100,
                 n_inducing: int = 50,
                 noise: float = 1e-3,
                 priors: GPPriors | None = None,
                 optimize_hyperparams: bool = True,
                 n_restarts: int = 2):
        self.nu = nu
        self.ard = ard
        self.sparse_threshold = sparse_threshold
        self.n_inducing = n_inducing
        self.noise = noise
        self.priors = priors or GPPriors()
        self.optimize_hyperparams = optimize_hyperparams
        self.n_restarts = n_restarts
        self._model: GPModel | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    # ----- MAP fit --------------------------------------------------------

    def _neg_log_marginal(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray
                          ) -> float:
        """MAP objective: NLL + neg-log-prior. `theta` packs log-hyperparameters."""
        d = X.shape[1]
        if self.ard:
            log_ls = theta[:d]
            log_sig2 = float(theta[d])
            log_noise = float(theta[d + 1])
        else:
            log_ls = np.full(d, float(theta[0]))
            log_sig2 = float(theta[1])
            log_noise = float(theta[2])

        ls = np.exp(log_ls)
        sig2 = float(np.exp(log_sig2))
        noise = float(np.exp(log_noise))

        K = _matern_kernel(X, X, ls, sig2, self.nu)
        K += (noise + 1e-6) * np.eye(len(X))
        try:
            c, low = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = cho_solve((c, low), y)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        nll = 0.5 * (y @ alpha) + 0.5 * logdet + 0.5 * len(y) * np.log(2 * np.pi)
        prior_nll = self.priors.neg_log_prior(log_ls, log_sig2, log_noise)
        return float(nll + prior_nll)

    def _map_fit(self, X: np.ndarray, y: np.ndarray
                 ) -> Tuple[np.ndarray, float, float]:
        d = X.shape[1]
        rng = np.random.default_rng(0)
        best = None
        for _ in range(max(1, self.n_restarts)):
            if self.ard:
                init = np.concatenate([
                    rng.normal(0.0, 0.5, size=d),
                    [rng.normal(0.0, 0.3), rng.normal(-4.0, 0.3)],
                ])
            else:
                init = rng.normal([0.0, 0.0, -4.0], 0.3)
            try:
                res = minimize(self._neg_log_marginal, init, args=(X, y),
                               method="L-BFGS-B", options={"maxiter": 80})
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue
        if best is None:
            return np.ones(d), 1.0, self.noise
        theta = best.x
        if self.ard:
            ls = np.exp(theta[:d])
            sig2 = float(np.exp(theta[d]))
            noise = float(np.exp(theta[d + 1]))
        else:
            ls = np.full(d, float(np.exp(theta[0])))
            sig2 = float(np.exp(theta[1]))
            noise = float(np.exp(theta[2]))
        return ls, sig2, noise

    # ----- training -------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparseGP":
        X = np.atleast_2d(X).astype(float)
        y = np.asarray(y, dtype=float).ravel()
        self._y_mean, self._y_std = float(np.mean(y)), float(np.std(y) + 1e-9)
        y_norm = (y - self._y_mean) / self._y_std

        d = X.shape[1]
        if self.optimize_hyperparams and len(X) >= 4:
            ls, sigma2, noise = self._map_fit(X, y_norm)
        else:
            ls = np.full(d, 1.0 if self.ard else float(np.std(X) + 1e-3))
            sigma2 = 1.0
            noise = self.noise

        sparse = len(X) > self.sparse_threshold
        if sparse:
            idx = np.random.default_rng(0).choice(len(X), self.n_inducing,
                                                  replace=False)
            inducing = X[idx]
            Kmm = _matern_kernel(inducing, inducing, ls, sigma2, self.nu)
            Knm = _matern_kernel(X, inducing, ls, sigma2, self.nu)
            Kmm += 1e-6 * np.eye(len(inducing))
            L = np.linalg.cholesky(Kmm)
            A = np.linalg.solve(L, Knm.T)
            Qnn_diag = np.sum(A * A, axis=0)
            Lam = np.maximum(noise + sigma2 - Qnn_diag, 1e-6)
            B = A @ np.diag(1.0 / Lam) @ A.T + np.eye(len(inducing))
            Lb = np.linalg.cholesky(B + 1e-6 * np.eye(len(inducing)))
            rhs = A @ (y_norm / Lam)
            alpha = np.linalg.solve(Lb.T, np.linalg.solve(Lb, rhs))
            self._model = GPModel(X, y_norm, ls, sigma2, noise, Lb, alpha,
                                  inducing)
        else:
            K = _matern_kernel(X, X, ls, sigma2, self.nu)
            K += (noise + 1e-6) * np.eye(len(X))
            c, low = cho_factor(K, lower=True)
            alpha = cho_solve((c, low), y_norm)
            self._model = GPModel(X, y_norm, ls, sigma2, noise, c, alpha)
        return self

    # ----- prediction -----------------------------------------------------

    def predict(self, Xs: np.ndarray, return_std: bool = True
                ) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        m = self._model
        Xs = np.atleast_2d(Xs).astype(float)

        if m.inducing is not None:
            Kms = _matern_kernel(m.inducing, Xs, m.length_scales,
                                 m.signal_var, self.nu)
            mean = Kms.T @ m.alpha
            v = np.linalg.solve(m.L, np.linalg.solve(m.L.T, Kms))
            var = m.signal_var - np.sum(Kms * v, axis=0)
            var = np.maximum(var, 1e-8) + m.noise_var
        else:
            Ks = _matern_kernel(m.X, Xs, m.length_scales, m.signal_var, self.nu)
            mean = Ks.T @ m.alpha
            v = cho_solve((m.L, True), Ks)
            var = m.signal_var + m.noise_var - np.sum(Ks * v, axis=0)
            var = np.maximum(var, 1e-8)

        mean = mean * self._y_std + self._y_mean
        std = np.sqrt(var) * self._y_std
        if return_std:
            return mean, std
        return mean, None  # type: ignore[return-value]

    # ----- acquisitions ---------------------------------------------------

    def expected_improvement(self, Xs: np.ndarray,
                             y_best: float, xi: float = 0.01) -> np.ndarray:
        """EI for *minimization* of the loss (Manuscript stage 30–70%)."""
        from scipy.stats import norm
        mu, std = self.predict(Xs)
        std = np.maximum(std, 1e-9)
        z = (y_best - mu - xi) / std
        return (y_best - mu - xi) * norm.cdf(z) + std * norm.pdf(z)

    def upper_confidence_bound(self, Xs: np.ndarray,
                               beta: float = 2.0) -> np.ndarray:
        """LCB for *minimization*: score = -(μ − √β · σ); higher = better.

        Manuscript exploration stage (0–30% of iterations) uses this with a
        decaying β to broaden the surrogate's support.
        """
        mu, std = self.predict(Xs)
        return -(mu - np.sqrt(max(beta, 1e-9)) * np.maximum(std, 1e-9))

    def probability_of_improvement(self, Xs: np.ndarray,
                                   y_best: float,
                                   xi: float = 0.0) -> np.ndarray:
        """PI for *minimization* (Manuscript exploitation stage > 70%)."""
        from scipy.stats import norm
        mu, std = self.predict(Xs)
        std = np.maximum(std, 1e-9)
        z = (y_best - mu - xi) / std
        return norm.cdf(z)

    def thompson_sample(self, Xs: np.ndarray,
                        rng: np.random.Generator | None = None) -> np.ndarray:
        """Single Thompson draw at each candidate (used as the 0.15-prob
        injection in the manuscript's staged acquisition).
        """
        rng = rng or np.random.default_rng()
        mu, std = self.predict(Xs)
        # Negate so larger = better (matches EI/UCB/PI orientation in PIBO).
        return -rng.normal(loc=mu, scale=np.maximum(std, 1e-9))

    # ----- posterior samples for parameter-posterior plots ---------------

    def posterior_samples(self, Xs: np.ndarray, n: int = 100,
                          rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        mu, std = self.predict(Xs)
        return rng.normal(loc=mu, scale=std, size=(n, len(mu)))