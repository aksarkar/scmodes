import numpy as np
import pytest
import scipy.stats as st

@pytest.fixture
def simulate():
  np.random.seed(0)
  l = np.random.normal(size=(100, 3))
  f = np.random.normal(size=(3, 200))
  eta = l.dot(f)
  eta *= 5 / eta.max()
  x = np.random.poisson(lam=np.exp(eta))
  return x, eta

def _simulate_lam_low_rank(n, p, k):
  np.random.seed(0)
  l = np.exp(np.random.normal(size=(n, k)))
  f = np.exp(np.random.normal(size=(k, p)))
  lam = l.dot(f)
  x = np.random.poisson(lam=lam)
  return x, lam

@pytest.fixture
def simulate_lam_rank1():
  return _simulate_lam_low_rank(100, 200, 1)

@pytest.fixture
def simulate_lam_rank2():
  return _simulate_lam_low_rank(100, 200, 2)

def _simulate_truncnorm(n, p, k):
  np.random.seed(0)
  l = np.random.lognormal(size=(n, k))
  f = np.random.lognormal(size=(p, k))
  F = st.norm(loc=l.dot(f.T))
  x = np.clip(F.rvs(size=(n, p)), 0, None)
  oracle_llik = F.logpdf(x).sum()
  return x, l, f, oracle_llik

@pytest.fixture
def simulate_truncnorm_rank1():
  return _simulate_truncnorm(n=100, p=200, k=1)

@pytest.fixture
def simulate_truncnorm_rank2():
  return _simulate_truncnorm(n=100, p=200, k=2)
