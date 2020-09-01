import numpy as np
import scipy.stats as st
import scmodes.ebnbm
from ebpm.fixtures import *

@pytest.fixture
def simulate_nb_gamma():
  np.random.seed(1)
  n = 500
  p = 10
  s = 1e5 * np.ones((n, 1))
  alpha = 0.2
  log_mu = np.random.uniform(-12, -6, size=(1, p))
  log_phi = np.random.uniform(-6, 0, size=(1, p))
  G = st.gamma(a=np.exp(-log_phi), scale=np.exp(log_mu + log_phi))
  lam = G.rvs(size=(n, p))
  x = st.nbinom(n=1 / alpha, p=1 / (1 + s * lam * alpha)).rvs()
  n_samples = 1000
  llik = st.nbinom(n=1 / alpha, p=1 / (1 + s * G.rvs(size=(n_samples, n, p)) * alpha)).logpmf(x).mean(axis=0).sum()
  return x, s, log_mu, log_phi, alpha, llik

def test_ebnbm_gamma_em(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  log_mu_hat, log_phi_hat, log_alpha_hat, elbo = scmodes.ebnbm.ebnbm_gamma(x, s, tol=5e-2, alpha=1e-3, max_iters=10000, extrapolate=False)
  assert log_mu_hat.shape == log_mu.shape
  assert log_phi_hat.shape == log_phi.shape
  assert log_alpha_hat.shape == ()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(log_phi_hat).all()
  assert np.isfinite(log_alpha_hat)
  assert np.isfinite(elbo)

def test_ebnbm_gamma_em_nb_measurement(simulate_nb_gamma):
  x, s, log_mu, log_phi, alpha, l0 = simulate_nb_gamma
  log_mu_hat, log_phi_hat, log_alpha_hat, elbo = scmodes.ebnbm.ebnbm_gamma(
    x, s, tol=5e-2, alpha=alpha, max_iters=10000, extrapolate=False, verbose=True)
  assert log_mu_hat.shape == log_mu.shape
  assert log_phi_hat.shape == log_phi.shape
  assert log_alpha_hat.shape == ()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(log_phi_hat).all()
  assert np.isfinite(log_alpha_hat)
  assert np.isfinite(elbo)

def test_ebnbm_gamma_squarem(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  log_mu_hat, log_phi_hat, log_alpha_hat, elbo = scmodes.ebnbm.ebnbm_gamma(x, s, tol=1e-3, alpha=1e3, max_iters=10000)
  assert log_mu_hat.shape == log_mu.shape
  assert log_phi_hat.shape == log_phi.shape
  assert log_alpha_hat.shape == ()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(log_phi_hat).all()
  assert np.isfinite(log_alpha_hat)
  assert np.isfinite(elbo)

def test_ebnbm_gamma_em_nb_measurement(simulate_nb_gamma):
  x, s, log_mu, log_phi, alpha, l0 = simulate_nb_gamma
  log_mu_hat, log_phi_hat, log_alpha_hat, elbo = scmodes.ebnbm.ebnbm_gamma(
    x, s, tol=5e-2, alpha=alpha, max_iters=10000, extrapolate=True)
  assert log_mu_hat.shape == log_mu.shape
  assert log_phi_hat.shape == log_phi.shape
  assert log_alpha_hat.shape == ()
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(log_phi_hat).all()
  assert np.isfinite(log_alpha_hat)
  assert np.isfinite(elbo)
