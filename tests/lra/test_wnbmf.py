import numpy as np
import scipy.stats as st
import scmodes
import scmodes.lra.wnbmf

from .fixtures import *

def test_wnbmf_rank1(simulate_lam_rank1):
  _, lam = simulate_lam_rank1
  n, p = lam.shape
  u = np.random.gamma(shape=1, scale=1, size=lam.shape)
  x = np.random.poisson(lam=lam * u)
  oracle_llik = st.nbinom(n=1, p=1 / (lam + 1)).logpmf(x).sum()
  l, f, loss = scmodes.lra.nbmf(x, rank=1, inv_disp=1)
  assert l.shape == (n, 1)
  assert f.shape == (p, 1)
  assert np.isfinite(l).all()
  assert np.isfinite(f).all()
  assert (l >= 0).all()
  assert (f >= 0).all()
  assert -loss > oracle_llik

def test_wnbmf_rank2(simulate_lam_rank2):
  _, lam = simulate_lam_rank2
  n, p = lam.shape
  u = np.random.gamma(shape=1, scale=1, size=lam.shape)
  x = np.random.poisson(lam=lam * u)
  oracle_llik = st.nbinom(n=1, p=1 / (1 + lam)).logpmf(x).sum()
  l, f, loss = scmodes.lra.nbmf(x, rank=2, inv_disp=1)
  assert l.shape == (n, 2)
  assert f.shape == (p, 2)
  assert np.isfinite(l).all()
  assert np.isfinite(f).all()
  assert (l >= 0).all()
  assert (f >= 0).all()
  assert -loss > oracle_llik

def test_wnbmf_rank1_est_theta_oracle_init(simulate_lam_rank1):
  _, lam = simulate_lam_rank1
  n, p = lam.shape
  u = np.random.gamma(shape=1, scale=1, size=lam.shape)
  x = np.random.poisson(lam=lam * u)
  oracle_llik = st.nbinom(n=1, p=1 / (1 + lam)).logpmf(x).sum()
  l, f, loss = scmodes.lra.nbmf(x, rank=1, inv_disp=1, fix_inv_disp=False)
  assert l.shape == (n, 1)
  assert f.shape == (p, 1)
  assert np.isfinite(l).all()
  assert np.isfinite(f).all()
  assert (l >= 0).all()
  assert (f >= 0).all()
  assert -loss > oracle_llik
