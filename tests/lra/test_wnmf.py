import numpy as np
import pytest
import scmodes.lra
import scmodes.lra.wnmf
import scipy.stats as st
import sklearn.decomposition as skd

from .fixtures import *

def test__frob_loss():
  np.random.seed(0)
  n = 100
  p = 200
  x = np.random.normal(size=(n, p))
  assert np.isclose(scmodes.lra.wnmf._frob_loss(x, 0, w=1), np.square(np.linalg.norm(x)))

def test__frob_loss_weight():
  np.random.seed(0)
  n = 100
  p = 200
  x = np.random.normal(size=(n, p))
  # Important: w == 1 denotes presence
  w = np.random.uniform(size=(n, p)) < 0.75
  assert scmodes.lra.wnmf._frob_loss(x, lf=0, w=w) < np.square(np.linalg.norm(x))

def test__pois_loss(simulate_lam_rank1):
  x, lam = simulate_lam_rank1
  oracle_llik = st.poisson(mu=lam).logpmf(x).sum()
  loss = scmodes.lra.wnmf._pois_loss(x, lam, w=1)
  assert np.isclose(-loss, oracle_llik)

def test__pois_loss_weight(simulate_lam_rank1):
  x, lam = simulate_lam_rank1
  n, p = x.shape
  # Important: w == 1 denotes presence
  w = np.random.uniform(size=(n, p)) < 0.75
  oracle_llik = np.where(w, st.poisson(mu=lam).logpmf(x), 0).sum()
  loss = scmodes.lra.wnmf._pois_loss(x, lam=lam, w=w)
  assert np.isclose(-loss, oracle_llik)

def test_nmf_frob_loss(simulate_truncnorm_rank1):
  x, _, _, oracle_llik = simulate_truncnorm_rank1
  n, p = x.shape
  l, f, loss = scmodes.lra.nmf(x, rank=1, pois_loss=False)
  assert l.shape == (n, 1)
  assert f.shape == (p, 1)
  assert -loss > oracle_llik

def test_nmf_frob_loss_rank2(simulate_truncnorm_rank2):
  x, _, _, oracle_llik = simulate_truncnorm_rank2
  _, _, loss = scmodes.lra.nmf(x, rank=2, pois_loss=False)
  assert -loss > oracle_llik

def test_nmf_frob_loss_weight(simulate_truncnorm_rank2):
  x, l, f, _ = simulate_truncnorm_rank2
  _, _, full_model_loss = scmodes.lra.nmf(x, rank=2, pois_loss=False)
  n, p = x.shape
  # Important: w == 1 denotes presence
  w = np.random.uniform(size=(n, p)) < 0.75
  oracle_llik = np.where(w, st.norm(loc=l.dot(f.T)).logpdf(x), 0).sum()
  _, _, loss = scmodes.lra.nmf(x, w=w, rank=2, pois_loss=False)
  assert loss < full_model_loss
  assert -loss > oracle_llik

def test_nmf_pois_loss(simulate_lam_rank1):
  x, lam = simulate_lam_rank1
  n, p = x.shape
  oracle_llik = st.poisson(mu=lam).logpmf(x).sum()
  # For rank 1, MLE is analytic (up to a scaling factor)
  true_l = x.sum(axis=1).astype(float)
  true_f = x.sum(axis=0).astype(float)
  true_l /= true_f.sum()
  true_f /= true_l.sum()
  true_llik = st.poisson(mu=np.outer(true_l, true_f)).logpmf(x).sum()
  l, f, loss = scmodes.lra.nmf(x, rank=1, max_iters=2)
  assert l.shape == (n, 1)
  assert f.shape == (p, 1)
  assert np.isclose(-loss, true_llik)
  assert -loss > oracle_llik

def test_nmf_pois_loss_rank2(simulate_lam_rank2):
  x, lam = simulate_lam_rank2
  oracle_llik = st.poisson(mu=lam).logpmf(x).sum()
  l, f, loss = scmodes.lra.nmf(x, rank=2)
  assert -loss > oracle_llik

def test_nmf_pois_loss_weight():
  np.random.seed(0)
  n = 100
  p = 200
  mu = 1000
  x = mu * np.ones((n, p))
  # Important: w == 1 denotes presence
  w = np.random.uniform(size=(n, p)) < 0.75
  oracle_llik = np.where(w, st.poisson(mu=mu).logpmf(x), 0).sum()
  lhat, fhat, loss = scmodes.lra.nmf(x, w=w, rank=1, pois_loss=True, verbose=True)
  muhat = lhat.dot(fhat.T)
  assert -loss > oracle_llik
  assert np.isclose(muhat, mu).all()
