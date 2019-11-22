import numpy as np
import pytest
import scipy.stats as st
import scmodes.sgd
import torch
import torch.utils.data

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
  assert np.isclose(oracle_llik, llik)

def test_PoissonGamma():
  p = 10
  m = scmodes.sgd.PoissonGamma(p)
  log_mu, log_phi = m.opt()
  assert log_mu.shape == (1, p)
  assert log_phi.shape == (1, p)

def test_PoissonGamma_fit(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  D = torch.utils.data
  # Important: data must be float
  loader = D.DataLoader(
    D.TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(s, dtype=torch.float)),
    batch_size=1, pin_memory=True)
  m = scmodes.sgd.PoissonGamma(p)
  m.fit(loader, max_epochs=10)

  log_mu_hat, log_phi_hat = m.opt()
  assert log_mu_hat.shape == (1, p)
  assert log_phi_hat.shape == (1, p)
  l1 = st.nbinom(n=np.exp(-log_phi_hat), p=1 / (1 + s.dot(np.exp(log_mu_hat + log_phi_hat)))).logpmf(x).sum()
  assert l1 > l0
