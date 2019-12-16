import numpy as np
import pytest
import scipy.sparse as ss
import scipy.special as sp
import scipy.stats as st
import scmodes.ebpm.sgd
import torch
import torch.utils.data as td

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
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0

def test_ebpm_gamma_minibatch(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, l1 = scmodes.ebpm.sgd.ebpm_gamma(x, s, batch_size=100, max_epochs=100)
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0

def test_ebpm_gamma_sgd(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  # Important: learning rate has to lowered to compensate for increased
  # variance in gradient estimator
  log_mu_hat, neg_log_phi_hat, l1 = scmodes.ebpm.sgd.ebpm_gamma(x, s, batch_size=1, max_epochs=10, lr=5e-3)
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert l1 > l0

def test_ebpm_gamma_trace(simulate_gamma):
  x, s, log_mu, log_phi, l0 = simulate_gamma
  n, p = x.shape
  max_epochs = 5
  log_mu_hat, neg_log_phi_hat, l1, trace = scmodes.ebpm.sgd.ebpm_gamma(x[:,0].reshape(-1, 1), s, batch_size=n, max_epochs=max_epochs, trace=True)
  assert len(trace) == max_epochs

def test_ebpm_point_gamma_oracle_init(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1 = scmodes.ebpm.sgd.ebpm_point_gamma(x, s, init=(log_mu, -log_phi), batch_size=n, max_epochs=2000)
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 > l0

def test_ebpm_point_gamma_batch(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1 = scmodes.ebpm.sgd.ebpm_point_gamma(x, s, batch_size=n, max_epochs=2000)
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 > l0

def test_ebpm_point_gamma_sparse(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  y = ss.csr_matrix(x)
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1 = scmodes.ebpm.sgd.ebpm_point_gamma(y, batch_size=100, max_epochs=100)
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 > l0

def test_ebpm_point_gamma_minibatch(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1 = scmodes.ebpm.sgd.ebpm_point_gamma(x, s, batch_size=100, max_epochs=100)
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 > l0

def test_ebpm_point_gamma_sgd(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  # Important: learning rate has to lowered to compensate for increased
  # variance in gradient estimator
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1 = scmodes.ebpm.sgd.ebpm_point_gamma(x, s, batch_size=1, max_epochs=10, lr=5e-3)
  assert np.isfinite(log_mu_hat).all()
  assert np.isfinite(neg_log_phi_hat).all()
  assert np.isfinite(logodds_hat).all()
  assert log_mu_hat.shape == (1, p)
  assert neg_log_phi_hat.shape == (1, p)
  assert logodds.shape == (1, p)
  assert l1 > l0

def test_ebpm_point_gamma_trace(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  n, p = x.shape
  max_epochs = 2000
  log_mu_hat, neg_log_phi_hat, logodds_hat, l1, trace = scmodes.ebpm.sgd.ebpm_point_gamma(x[:,0].reshape(-1, 1), s, batch_size=n, max_epochs=max_epochs, trace=True)
  assert len(trace) == max_epochs

def test_EBPMDataset_init(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  y = ss.csr_matrix(x)
  data = scmodes.ebpm.sgd.EBPMDataset(y, s)
  assert len(data) == y.shape[0]
  if torch.cuda.is_available():
    assert (data.data.cpu().numpy() == y.data).all()
    assert (data.indices.cpu().numpy() == y.indices).all()
    assert (data.indptr.cpu().numpy() == y.indptr).all()
  else:
    assert (data.data.numpy() == y.data).all()
    assert (data.indices.numpy() == y.indices).all()
    assert (data.indptr.numpy() == y.indptr).all()

def test_EBPMDataset_init_dense(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  data = scmodes.ebpm.sgd.EBPMDataset(x, s)
  if torch.cuda.is_available():
    y = ss.csr_matrix((data.data.cpu().numpy(), data.indices.cpu().numpy(), data.indptr.cpu().numpy()))
  else:
    y = ss.csr_matrix((data.data.numpy(), data.indices.numpy(), data.indptr.numpy()))
  assert (y.todense() == x).all()

def test_EBPMDataset_init_coo(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  y = ss.coo_matrix(x)
  data = scmodes.ebpm.sgd.EBPMDataset(y, s)
  if torch.cuda.is_available():
    z = ss.csr_matrix((data.data.cpu().numpy(), data.indices.cpu().numpy(), data.indptr.cpu().numpy())).tocoo()
  else:
    z = ss.csr_matrix((data.data.numpy(), data.indices.numpy(), data.indptr.numpy())).tocoo()
  # This is more efficient than ==
  assert not (y != z).todense().any()

def test_EBPMDataset__get_item__(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  data = scmodes.ebpm.sgd.EBPMDataset(x, s)
  y, t = data[0]
  if torch.cuda.is_available():
    assert (y.cpu().numpy() == x[0]).all()
    assert t.cpu().numpy() == s[0]
  else:
    assert (y.numpy() == x[0]).all()
    assert t.numpy() == s[0]

def test_EBPMDataset_collate_fn(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  data = scmodes.ebpm.sgd.EBPMDataset(x, s)
  batch_size = 10
  y, t = data.collate_fn(range(batch_size))
  if torch.cuda.is_available():
    assert (y.cpu().numpy() == x[:batch_size]).all()
    assert (t.cpu().numpy() == s[:batch_size]).all()
  else:
    assert (y.numpy() == x[:batch_size]).all()
    assert (t.numpy() == s[:batch_size]).all()

def test_EBPMDataset_DataLoader(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  batch_size = 10
  data = td.DataLoader(scmodes.ebpm.sgd.EBPMDataset(x, s), batch_size=batch_size, shuffle=False)
  y, t = next(iter(data))
  if torch.cuda.is_available():
    assert (y.cpu().numpy() == x[:batch_size]).all()
    assert (t.cpu().numpy() == s[:batch_size]).all()
  else:
    assert (y.numpy() == x[:batch_size]).all()
    assert (t.numpy() == s[:batch_size]).all()
