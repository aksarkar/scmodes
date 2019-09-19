import numpy as np
import os
import pandas as pd
import pytest
import scipy.stats as st
import scmodes

@pytest.fixture
def test_data():
  x = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'counts.txt.gz'), sep='\t', index_col=0)
  return x

def test_gof():
  np.random.seed(0)
  mu = 10
  px = st.poisson(mu=mu)
  x = px.rvs(size=100)
  d, p = scmodes.benchmark.gof(x, cdf=px.cdf, pmf=px.pmf)
  assert d >= 0
  assert 0 <= p <= 1

def test_rpp():
  np.random.seed(0)
  mu = 10
  px = st.poisson(mu=mu)
  x = px.rvs(size=100)
  F = px.cdf(x - 1)
  f = px.pmf(x)
  vals = scmodes.benchmark.rpp(F, f)
  assert vals.shape == x.shape

def test_gamma_cdf():
  np.random.seed(0)
  x = st.nbinom(n=10, p=.1).rvs(size=100)
  Fx = scmodes.benchmark.zig_cdf(x, size=1, log_mu=-5, log_phi=-1)
  assert Fx.shape == x.shape
  assert np.isfinite(Fx).all()
  assert (Fx >= 0).all()
  assert (Fx <= 1).all()

def test_zig_cdf():
  np.random.seed(0)
  x = st.nbinom(n=10, p=.1).rvs(size=100)
  Fx = scmodes.benchmark.zig_cdf(x, size=1, log_mu=-5, log_phi=-1, logodds=-3)
  assert Fx.shape == x.shape
  assert (Fx >= 0).all()
  assert (Fx <= 1).all()

def test_gof_gamma(test_data):
  x = test_data
  res = scmodes.benchmark.gof_gamma(x)
  assert res.shape[0] == x.shape[1]
  assert np.isfinite(res['stat']).all()
  assert np.isfinite(res['p']).all()

def test_gof_zig(test_data):
  x = test_data
  res = scmodes.benchmark.gof_zig(x)
  assert res.shape[0] == x.shape[1]
  assert np.isfinite(res['stat']).all()
  assert np.isfinite(res['p']).all()
