import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st
import scmodes.sgd
import torch

@pytest.fixture
def simulate_gamma():
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

def test__nb_llik(simulate_gamma):
  x, s, log_mu, log_phi, oracle_llik = simulate_gamma
  llik = scmodes.sgd._nb_llik(torch.tensor(x, dtype=torch.float),
                              torch.tensor(s, dtype=torch.float),
                              torch.tensor(log_mu, dtype=torch.float),
                              torch.tensor(-log_phi, dtype=torch.float)).sum()
  assert np.isclose(llik, oracle_llik)

def test__zinb_llik(simulate_gamma):
  x, s, log_mu, log_phi, oracle_llik = simulate_gamma
  llik = scmodes.sgd._zinb_llik(torch.tensor(x, dtype=torch.float),
                                torch.tensor(s, dtype=torch.float),
                                torch.tensor(log_mu, dtype=torch.float),
                                torch.tensor(-log_phi, dtype=torch.float),
                                torch.tensor(-100, dtype=torch.float)).sum()
  assert np.isclose(llik, oracle_llik)

def test__zinb_llik_zinb_data(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  n, p = x.shape
  logodds = np.random.uniform(-3, -1, size=(1, p))
  pi0 = sp.expit(logodds)
  z = np.random.uniform(size=x.shape) < pi0
  y = np.where(z, 0, x)
  oracle_llik_nonzero = np.log(1 - pi0) + st.nbinom(n=np.exp(-log_phi), p=1 / (1 + s.dot(np.exp(log_mu + log_phi)))).logpmf(y)
  oracle_llik = np.where(y < 1, np.log(pi0 + np.exp(oracle_llik_nonzero)), oracle_llik_nonzero).sum()
  llik = scmodes.sgd._zinb_llik(torch.tensor(y, dtype=torch.float),
                                torch.tensor(s, dtype=torch.float),
                                torch.tensor(log_mu, dtype=torch.float),
                                torch.tensor(-log_phi, dtype=torch.float),
                                torch.tensor(logodds, dtype=torch.float)).sum()
  assert np.isclose(llik, oracle_llik)

def test_ebpm_gamma_batch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, l1 = scmodes.sgd.ebpm_gamma(x, s, batch_size=n, max_epochs=2000)
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0

def test_ebpm_gamma_minibatch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, l1 = scmodes.sgd.ebpm_gamma(x, s, batch_size=100, max_epochs=100)
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0

def test_ebpm_gamma_sgd(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  # Important: learning rate has to lowered to compensate for increased
  # variance in gradient estimator
  log_mu_hat, neg_log_phi_hat, l1 = scmodes.sgd.ebpm_gamma(x, s, batch_size=1, max_epochs=10, lr=5e-3)
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0
