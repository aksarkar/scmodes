import numpy as np
import scipy.special as sp
import scipy.stats as st
import scmodes.ebpm
import scmodes.ebpm.wrappers

from .fixtures import *

def test_ebpm_point():
  np.random.seed(0)
  n = 500
  s = np.ones(n)
  F = st.poisson(mu=10)
  x = F.rvs(n)
  oracle_llik = F.logpmf(x).sum()
  log_mu, llik = scmodes.ebpm.ebpm_point(x, s)
  assert np.isfinite(log_mu)
  assert llik > oracle_llik

def test_ebpm_gamma(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  # Important: log_mu, log_phi are [1, p]. We want oracle log likelihood for
  # only gene 0
  oracle_llik = st.nbinom(n=np.exp(-log_phi[0,0]), p=1 / (1 + s.dot(np.exp(log_mu[0,0] + log_phi[0,0])))).logpmf(x[:,0]).sum()
  log_mu_hat, neg_log_phi_hat, llik = scmodes.ebpm.ebpm_gamma(x[:,0], s.ravel(), extrapolate=False)
  assert np.isfinite(log_mu_hat)
  assert np.isfinite(neg_log_phi_hat)
  assert llik > oracle_llik

def test_ebpm_gamma_extrapolate(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  # Important: log_mu, log_phi are [1, p]. We want oracle log likelihood for
  # only gene 0
  oracle_llik = st.nbinom(n=np.exp(-log_phi[0,0]), p=1 / (1 + s.dot(np.exp(log_mu[0,0] + log_phi[0,0])))).logpmf(x[:,0]).sum()
  log_mu_hat, neg_log_phi_hat, llik = scmodes.ebpm.ebpm_gamma(x[:,0], s.ravel(), extrapolate=True)
  assert np.isfinite(log_mu_hat)
  assert np.isfinite(neg_log_phi_hat)
  assert llik > oracle_llik

def test_ebpm_point_gamma(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, _ = simulate_point_gamma
  # Important: log_mu, log_phi, logodds are [1, p]. We want oracle log
  # likelihood for only gene 0
  pi0 = sp.expit(logodds[0,0])
  F = st.nbinom(n=np.exp(-log_phi[0,0]), p=1 / (1 + s.dot(np.exp(log_mu[0,0] + log_phi[0,0]))))
  oracle_llik_nonzero = np.log(1 - pi0) + F.logpmf(x[:,0])
  oracle_llik = np.where(x[:,0] < 1, np.log(pi0 + np.exp(oracle_llik_nonzero)), oracle_llik_nonzero).sum()
  log_mu_hat, neg_log_phi_hat, logodds_hat, llik = scmodes.ebpm.ebpm_point_gamma(x[:,0], s.ravel(), extrapolate=False)
  assert np.isfinite(log_mu_hat)
  assert np.isfinite(neg_log_phi_hat)
  assert np.isfinite(logodds_hat)
  assert llik > oracle_llik

def test_ebpm_point_gamma_extrapolate(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, _ = simulate_point_gamma
  # Important: log_mu, log_phi, logodds are [1, p]. We want oracle log
  # likelihood for only gene 0
  pi0 = sp.expit(logodds[0,0])
  F = st.nbinom(n=np.exp(-log_phi[0,0]), p=1 / (1 + s.dot(np.exp(log_mu[0,0] + log_phi[0,0]))))
  oracle_llik_nonzero = np.log(1 - pi0) + F.logpmf(x[:,0])
  oracle_llik = np.where(x[:,0] < 1, np.log(pi0 + np.exp(oracle_llik_nonzero)), oracle_llik_nonzero).sum()
  log_mu_hat, neg_log_phi_hat, logodds_hat, llik = scmodes.ebpm.ebpm_point_gamma(x[:,0], s.ravel(), extrapolate=True)
  assert np.isfinite(log_mu_hat)
  assert np.isfinite(neg_log_phi_hat)
  assert np.isfinite(logodds_hat)
  assert llik > oracle_llik

def test_ebpm_unimodal(simulate_gamma):
  x, s, log_mu, log_phi, oracle_llik = simulate_gamma
  res = scmodes.ebpm.ebpm_unimodal(x[:,0], s.ravel())
  llik = np.array(res.rx2('loglik'))
  assert llik > oracle_llik

def test_ebpm_point_expfam(simulate_gamma):
  x, s, log_mu, log_phi, oracle_llik = simulate_gamma
  res = scmodes.ebpm.ebpm_point_expfam(x[:,0], s.ravel())
  g = np.array(res.slots['distribution'])[:,:2]
  # Don't marginalize over lambda = 0 for x > 0, because
  # p(x > 0 | lambda = 0) = 0
  llik = np.where(x[:,0] > 0,
                  np.log(st.poisson(mu=s * g[1:,0]).pmf(x[:,:1]).dot(g[1:,1])),
                  np.log(st.poisson(mu=s * g[:,0]).pmf(x[:,:1]).dot(g[:,1]))).sum()
  assert llik > oracle_llik

def test_ebpm_npmle(simulate_gamma):
  x, s, log_mu, log_phi, oracle_llik = simulate_gamma
  res = scmodes.ebpm.ebpm_npmle(x[:,0], s.ravel())
  llik = np.array(res.rx2('loglik'))
  assert llik > oracle_llik
