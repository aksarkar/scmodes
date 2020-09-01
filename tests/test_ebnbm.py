import numpy as np
import scipy.stats as st
import scmodes.ebnbm
from ebpm.fixtures import *

def test_ebnbm_gamma_em(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  # Monte Carlo integral
  n, p = x.shape
  l0 = st.poisson(s * st.gamma(a=np.exp(-log_phi), scale=np.exp(log_mu + log_phi)).rvs(size=(10, n, p))).logpmf(x).mean(axis=0).sum()
  log_mu_hat, log_phi_hat, log_alpha_hat, l1 = scmodes.ebnbm.ebnbm_gamma(x, s, tol=5e-2, alpha=1e3, max_iters=10000, extrapolate=False)
  assert log_mu_hat.shape == log_mu.shape
  assert log_phi_hat.shape == log_phi.shape
  assert log_alpha_hat.shape == ()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(log_phi_hat).all()
  assert np.isfinite(log_alpha_hat)
  assert l1 >= l0

def test_ebnbm_gamma_squarem(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  # Monte Carlo integral
  n, p = x.shape
  l0 = st.poisson(s * st.gamma(a=np.exp(-log_phi), scale=np.exp(log_mu + log_phi)).rvs(size=(10, n, p))).logpmf(x).mean(axis=0).sum()
  log_mu_hat, log_phi_hat, log_alpha_hat, l1 = scmodes.ebnbm.ebnbm_gamma(x, s, tol=1e-3, alpha=1e3, max_iters=10000)
  assert log_mu_hat.shape == log_mu.shape
  assert log_phi_hat.shape == log_phi.shape
  assert log_alpha_hat.shape == ()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(log_phi_hat).all()
  assert np.isfinite(log_alpha_hat)
  assert l1 >= l0
