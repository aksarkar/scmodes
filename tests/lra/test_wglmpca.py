import pytest
import scipy.stats as st
import scmodes.lra

from .fixtures import *

def test_glmpca_rank1(simulate_rank1):
  x, _, _, oracle_llik = simulate_rank1
  n, p = x.shape
  l, f, loss = scmodes.lra.glmpca(x, rank=1, seed=0)
  assert l.shape == (n, 1)
  assert f.shape == (p, 1)
  assert np.isfinite(l).all()
  assert np.isfinite(f).all()
  assert -loss > oracle_llik

def test_glmpca_rank2(simulate_rank2):
  x, _, _, oracle_llik = simulate_rank2
  n, p = x.shape
  l, f, loss = scmodes.lra.glmpca(x, rank=2, seed=0, verbose=True)
  assert l.shape == (n, 2)
  assert f.shape == (p, 2)
  assert np.isfinite(l).all()
  assert np.isfinite(f).all()
  assert -loss > oracle_llik

def test_glmpca_rank1_weight(simulate_rank1):
  x, l, f, _ = simulate_rank1
  # Important: w == 1 denotes presence
  w = np.random.uniform(size=x.shape) < 0.75
  oracle_llik = np.where(w, st.poisson(mu=np.exp(l.dot(f))).logpmf(x), 0).sum()
  l, f, loss = scmodes.lra.glmpca(x, w=w, rank=1, seed=0)
  assert np.isfinite(l).all()
  assert np.isfinite(f).all()
  assert -loss > oracle_llik

def test_glmpca_rank2_weight(simulate_rank2):
  x, l, f, _ = simulate_rank2
  # Important: w == 1 denotes presence
  w = np.random.uniform(size=x.shape) < 0.75
  oracle_llik = np.where(w, st.poisson(mu=np.exp(l.dot(f))).logpmf(x), 0).sum()
  l, f, loss = scmodes.lra.glmpca(x, w=w, rank=1, seed=0)
  assert np.isfinite(l).all()
  assert np.isfinite(f).all()
  assert -loss > oracle_llik
