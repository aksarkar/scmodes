import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st

def _simulate_gamma():
  n = 500
  p = 10
  np.random.seed(0)
  # Typical values (Sarkar et al. PLoS Genet 2019)
  log_mu = np.random.uniform(-12, -6, size=(1, p))
  log_phi = np.random.uniform(-6, 0, size=(1, p))
  s = np.random.poisson(lam=1e5, size=(n, 1))
  # Important: NB success probability is (n, p)
  F = st.nbinom(n=np.exp(-log_phi), p=1 / (1 + s.dot(np.exp(log_mu + log_phi))))
  x = F.rvs()
  llik = F.logpmf(x).sum()
  return x, s, log_mu, log_phi, llik

@pytest.fixture
def simulate_gamma():
  return _simulate_gamma()

@pytest.fixture
def simulate_point_gamma():
  x, s, log_mu, log_phi, _ = _simulate_gamma()
  n, p = x.shape
  logodds = np.random.uniform(-3, -1, size=(1, p))
  pi0 = sp.expit(logodds)
  z = np.random.uniform(size=x.shape) < pi0
  y = np.where(z, 0, x)
  F = st.nbinom(n=np.exp(-log_phi), p=1 / (1 + s.dot(np.exp(log_mu + log_phi))))
  llik_nonzero = np.log(1 - pi0) + F.logpmf(y)
  llik = np.where(y < 1, np.log(pi0 + np.exp(llik_nonzero)), llik_nonzero).sum()
  return y, s, log_mu, log_phi, logodds, llik
