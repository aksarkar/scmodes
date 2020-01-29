import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pytest
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import scipy.stats as st
import scmodes
import scmodes.benchmark.gof

ashr = rpy2.robjects.packages.importr('ashr')
rpy2.robjects.pandas2ri.activate()

@pytest.fixture
def test_data():
  x = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'counts.txt.gz'), sep='\t', index_col=0)
  return x

def test__gof():
  np.random.seed(0)
  mu = 10
  px = st.poisson(mu=mu)
  x = px.rvs(size=100)
  d, p = scmodes.benchmark.gof._gof(x, cdf=px.cdf, pmf=px.pmf)
  assert d >= 0
  assert 0 <= p <= 1

def test__rpp():
  np.random.seed(0)
  mu = 10
  px = st.poisson(mu=mu)
  x = px.rvs(size=100)
  F = px.cdf(x - 1)
  f = px.pmf(x)
  vals = scmodes.benchmark.gof._rpp(F, f)
  assert vals.shape == x.shape

def test_gamma_cdf():
  np.random.seed(0)
  x = st.nbinom(n=10, p=.1).rvs(size=100)
  Fx = scmodes.benchmark.gof._zig_cdf(x, size=1, log_mu=-5, log_phi=-1)
  assert Fx.shape == x.shape
  assert np.isfinite(Fx).all()
  assert (Fx >= 0).all()
  assert (Fx <= 1).all()

def test_zig_cdf():
  np.random.seed(0)
  x = st.nbinom(n=10, p=.1).rvs(size=100)
  Fx = scmodes.benchmark.gof._zig_cdf(x, size=1, log_mu=-5, log_phi=-1, logodds=-3)
  assert Fx.shape == x.shape
  assert (Fx >= 0).all()
  assert (Fx <= 1).all()

def test_zig_pmf_cdf():
  x = np.arange(50)
  import scmodes.benchmark.gof
  size = 1000
  log_mu=-5
  log_phi=-1
  logodds=-1
  Fx = scmodes.benchmark.gof._zig_cdf(x, size=size, log_mu=log_mu, log_phi=log_phi, logodds=logodds)
  Fx_1 = scmodes.benchmark.gof._zig_cdf(x - 1, size=size, log_mu=log_mu, log_phi=log_phi, logodds=logodds)
  fx = scmodes.benchmark.gof._zig_pmf(x, size=size, log_mu=log_mu, log_phi=log_phi, logodds=logodds)
  assert np.isclose(Fx - Fx_1, fx).all()

def test_gof_gamma(test_data):
  x = test_data
  res = scmodes.benchmark.gof_gamma(x)
  assert res.shape[0] == x.shape[1]
  assert np.isfinite(res['stat']).all()
  assert np.isfinite(res['p']).all()

def test_gof_gamma_size(test_data):
  x = test_data
  s = 1 + np.median(x, axis=1).reshape(-1, 1)
  res = scmodes.benchmark.gof_gamma(x, s=s, lr=1e-3)
  assert res.shape[0] == x.shape[1]
  assert np.isfinite(res['stat']).all()
  assert np.isfinite(res['p']).all()

def test_gof_zig(test_data):
  x = test_data
  res = scmodes.benchmark.gof_zig(x)
  assert res.shape[0] == x.shape[1]
  assert np.isfinite(res['stat']).all()
  assert np.isfinite(res['p']).all()

def test_gof_zig_size(test_data):
  x = test_data
  s = 1 + np.median(x, axis=1).reshape(-1, 1)
  res = scmodes.benchmark.gof_zig(x, s=s, lr=1e-3)
  assert res.shape[0] == x.shape[1]
  assert np.isfinite(res['stat']).all()
  assert np.isfinite(res['p']).all()

def test__ash_pmf(test_data):
  x = test_data
  gene = 'ENSG00000116251'
  xj = x[gene]
  size = x.sum(axis=1)
  lam = xj / size
  fit = ashr.ash_workhorse(
    # these are ignored by ash
    pd.Series(np.zeros(xj.shape)),
    1,
    outputlevel=pd.Series(['fitted_g', 'data']),
    # numpy2ri doesn't DTRT, so we need to use pandas
    lik=ashr.lik_pois(y=xj, scale=size, link='identity'),
    mixsd=pd.Series(np.geomspace(lam.min() + 1e-8, lam.max(), 25)),
    mode=pd.Series([lam.min(), lam.max()]))
  res = scmodes.benchmark.gof._ash_pmf(xj, fit)
  assert res.shape == xj.shape
  assert np.isfinite(res).all()
  assert (res >= 0).all()
  assert (res <= 1).all()

def test__ash_cdf(test_data):
  x = test_data
  gene = 'ENSG00000116251'
  xj = x[gene]
  size = x.sum(axis=1)
  lam = xj / size
  fit = ashr.ash_workhorse(
    # these are ignored by ash
    pd.Series(np.zeros(xj.shape)),
    1,
    outputlevel=pd.Series(['fitted_g', 'data']),
    # numpy2ri doesn't DTRT, so we need to use pandas
    lik=ashr.lik_pois(y=xj, scale=size, link='identity'),
    mixsd=pd.Series(np.geomspace(lam.min() + 1e-8, lam.max(), 25)),
    mode=pd.Series([lam.min(), lam.max()]))
  res = scmodes.benchmark.gof._ash_cdf(xj, fit, s=size)
  assert np.isfinite(res).all()
  assert (res >= 0).all()
  assert (res <= 1).all()

def test__ash_cdf_pmf(test_data):
  x = test_data
  gene = 'ENSG00000116251'
  xj = x[gene]
  size = x.sum(axis=1)
  lam = xj / size
  fit = ashr.ash_workhorse(
    # these are ignored by ash
    pd.Series(np.zeros(xj.shape)),
    1,
    outputlevel=pd.Series(['fitted_g', 'data']),
    # numpy2ri doesn't DTRT, so we need to use pandas
    lik=ashr.lik_pois(y=xj, scale=size, link='identity'),
    mixsd=pd.Series(np.geomspace(lam.min() + 1e-8, lam.max(), 25)),
    mode=pd.Series([lam.min(), lam.max()]))
  Fx = scmodes.benchmark.gof._ash_cdf(xj, fit, s=size)
  Fx_1 = scmodes.benchmark.gof._ash_cdf(xj - 1, fit, s=size)
  fx = scmodes.benchmark.gof._ash_pmf(xj, fit)
  assert np.isclose(Fx - Fx_1, fx).all()

def test__gof_unimodal(test_data):
  x = test_data
  gene = 'ENSG00000116251'
  k, d, p = scmodes.benchmark.gof._gof_unimodal(gene, x[gene], x.sum(axis=1))
  assert k == gene
  assert np.isfinite(d)
  assert d >= 0
  assert np.isfinite(p)
  assert 0 <= p <= 1

def test_gof_unimodal(test_data):
  x = test_data
  res = scmodes.benchmark.gof_unimodal(x)
  assert res.shape[0] == x.shape[1]
  assert np.isfinite(res['stat']).all()
  assert np.isfinite(res['p']).all()

def test_gof_unimodal_size(test_data):
  x = test_data
  s = x.sum(axis=1)
  res = scmodes.benchmark.gof_unimodal(x, s=s)
  assert res.shape[0] == x.shape[1]
  assert np.isfinite(res['stat']).all()
  assert np.isfinite(res['p']).all()

def test__point_expfam_cdf(test_data):
  x = test_data
  s = x.sum(axis=1)
  xj = x['ENSG00000116251']
  res = scmodes.ebpm.ebpm_point_expfam(xj, s)
  F = scmodes.benchmark.gof._point_expfam_cdf(xj.values.ravel(), res=res, size=s)
  assert np.isfinite(F).all()
  assert (F >= 0).all()
  assert (F <= 1).all()

def test__point_expfam_pmf(test_data):
  x = test_data
  s = x.sum(axis=1)
  xj = x['ENSG00000116251']
  res = scmodes.ebpm.ebpm_point_expfam(xj, s)
  f = scmodes.benchmark.gof._point_expfam_pmf(xj.values.ravel(), res=res, size=s)
  assert np.isfinite(f).all()
  assert (f >= 0).all()
  assert (f <= 1).all()

def test__point_expfam_cdf_pmf(test_data):
  x = test_data
  s = x.sum(axis=1)
  xj = x['ENSG00000116251']
  res = scmodes.ebpm.ebpm_point_expfam(xj, s)
  F = scmodes.benchmark.gof._point_expfam_cdf(xj.values.ravel(), res=res, size=s)
  F_1 = scmodes.benchmark.gof._point_expfam_cdf(xj.values.ravel() - 1, res=res, size=s)
  f = scmodes.benchmark.gof._point_expfam_pmf(xj.values.ravel(), res=res, size=s)
  assert np.isclose(F - F_1, f).all()

def test_evaluate_gof(test_data):
  x = test_data
  res = scmodes.benchmark.evaluate_gof(x, methods=['gamma', 'zig'])
  assert res.shape == (2 * x.shape[1], 4)
