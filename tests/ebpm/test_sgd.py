import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st
import scmodes.ebpm.sgd
import torch

from .fixtures import *

def test__nb_llik(simulate_gamma):
  x, s, log_mu, log_phi, oracle_llik = simulate_gamma
  llik = scmodes.ebpm.sgd._nb_llik(torch.tensor(x, dtype=torch.float),
                                   torch.tensor(s, dtype=torch.float),
                                   torch.tensor(log_mu, dtype=torch.float),
                                   torch.tensor(-log_phi, dtype=torch.float)).sum()
  assert np.isclose(llik, oracle_llik)

def test__zinb_llik(simulate_gamma):
  x, s, log_mu, log_phi, oracle_llik = simulate_gamma
  llik = scmodes.ebpm.sgd._zinb_llik(torch.tensor(x, dtype=torch.float),
                                     torch.tensor(s, dtype=torch.float),
                                     torch.tensor(log_mu, dtype=torch.float),
                                     torch.tensor(-log_phi, dtype=torch.float),
                                     torch.tensor(-100, dtype=torch.float)).sum()
  assert np.isclose(llik, oracle_llik)

def test__zinb_llik_zinb_data(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, oracle_llik = simulate_point_gamma
  n, p = x.shape
  llik = scmodes.ebpm.sgd._zinb_llik(torch.tensor(x, dtype=torch.float),
                                     torch.tensor(s, dtype=torch.float),
                                     torch.tensor(log_mu, dtype=torch.float),
                                     torch.tensor(-log_phi, dtype=torch.float),
                                     torch.tensor(logodds, dtype=torch.float)).sum()
  assert np.isclose(llik, oracle_llik)

def test_ebpm_gamma_batch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, l1 = scmodes.ebpm.sgd.ebpm_gamma(x, s, batch_size=n, max_epochs=2000)
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0

def test_ebpm_gamma_minibatch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, l1 = scmodes.ebpm.sgd.ebpm_gamma(x, s, batch_size=100, max_epochs=100)
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0

def test_ebpm_gamma_sgd(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  # Important: learning rate has to lowered to compensate for increased
  # variance in gradient estimator
  log_mu_hat, neg_log_phi_hat, l1 = scmodes.ebpm.sgd.ebpm_gamma(x, s, batch_size=1, max_epochs=10, lr=5e-3)
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0
