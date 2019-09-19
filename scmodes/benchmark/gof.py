import functools as ft
import numpy as np
import pandas as pd
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri
import scipy.stats as st
import scipy.special as sp

rpy2.robjects.pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()

ashr = rpy2.robjects.packages.importr('ashr')
descend = rpy2.robjects.packages.importr('descend')

def gof(x, cdf, pmf, **kwargs):
  """Test for goodness of fit x_i ~ \hat{F}(.)

  If x_i ~ F(.), then F(x_i) ~ Uniform(0, 1). Use randomized predictive
  p-values to handle discontinuous F.

  Parameters:

  x - array-like (n,)
  cdf - function returning F(x)
  pmf - function returning f(x)
  kwargs - arguments to cdf, pmf

  """
  F = cdf(x, **kwargs)
  f = pmf(x, **kwargs)
  q = rpp(F, f)
  return st.kstest(q, 'uniform')

def rpp(cdf, pmf):
  """Return randomized predictive p-values for discrete data x.

  Randomized predictive p-values q_i have the property that if x_i ~ F(.), then
  F(q_i) ~ Uniform(0, 1), even if F is discontinuous (as is the case for
  discrete random variables).

  Parameters:

  cdf - F(x - 1) array-like (n,)
  pmf - f(x) array-like (n,)

  References:

  - Dunn & Smyth. "Randomized Quantile Residuals." J Comp Graph
    Statist. 1996;5(3):236--244.

  - Feng et al. "Randomized Predictive P-values: A Versatile Model Diagnostic
    Tool with Unified Reference Distribution." https://arxiv.org/abs/1708.08527

  """
  u = np.random.uniform(size=cdf.shape[0])
  rpp = cdf + u * pmf
  return rpp

def zig_cdf(x, size, log_mu, log_phi, logodds=None):
  """Return marginal CDF of Poisson-(point) Gamma model

  x_i ~ Poisson(s_i \lambda_i)
  lambda_i ~ pi0 \delta_0(\cdot) + Gamma(1 / \phi, 1 / (\mu\phi))

  size - scalar or array-like (n,)
  log_mu - scalar
  log_phi - scalar
  logodds - scalar

  """
  n = np.exp(-log_phi)
  p = 1 / (1 + (size * np.exp(log_mu + log_phi)))
  if logodds is not None:
    pi0 = sp.expit(-logodds)
  else:
    pi0 = 0
  cdf = st.nbinom(n=n, p=p).cdf(x - 1)
  # Important: this excludes the right endpoint, so we need to special case x =
  # 0
  cdf = np.where(x > 0, pi0 + (1 - pi0) * cdf, cdf)
  return cdf

def zig_pmf(x, size, log_mu, log_phi, logodds=None):
  """Return marginal PMF of Poisson-(point) Gamma model

  x_i ~ Poisson(s_i \lambda_i)
  lambda_i ~ pi0 \delta_0(\cdot) + Gamma(1 / \phi, 1 / (\mu\phi))

  size - scalar or array-like (n,)
  log_mu - scalar
  log_phi - scalar
  logodds - scalar

  """
  n = np.exp(-log_phi)
  p = 1 / (1 + (size * np.exp(log_mu + log_phi)))
  pmf = st.nbinom(n=n, p=p).pmf(x)
  if logodds is not None:
    pi0 = sp.expit(-logodds)
    pmf *= (1 - pi0)
    pmf[x == 0] += pi0
  return pmf

def gof_gamma(x, **kwargs):
  import scqtl
  onehot = np.ones((x.shape[0], 1))
  size_factor = x.sum(axis=1).values.reshape(-1, 1)
  design = np.zeros((x.shape[0], 1))
  log_mu, log_phi, *_ = scqtl.tf.fit(
    umi=x.values.astype(np.float32),
    onehot=onehot.astype(np.float32),
    design=design.astype(np.float32),
    size_factor=size_factor.astype(np.float32),
    learning_rate=1e-3,
    max_epochs=30000)
  log_mu = pd.DataFrame(log_mu, columns=x.columns)
  log_phi = pd.DataFrame(log_phi, columns=x.columns)
  result = []
  for k in x:
    d, p = gof(x[k].values.ravel(), cdf=zig_cdf, pmf=zig_pmf,
               size=size_factor.ravel(), log_mu=log_mu.loc[0,k],
               log_phi=log_phi.loc[0,k])
    result.append((k, d, p))
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def gof_zig(x, **kwargs):
  import scqtl
  onehot = np.ones((x.shape[0], 1))
  size_factor = x.sum(axis=1).values.reshape(-1, 1)
  design = np.zeros((x.shape[0], 1))
  init = scqtl.tf.fit(
    umi=x.values.astype(np.float32),
    onehot=onehot.astype(np.float32),
    design=design.astype(np.float32),
    size_factor=size_factor.astype(np.float32),
    learning_rate=1e-3,
    max_epochs=30000)
  log_mu, log_phi, logodds, *_ = scqtl.tf.fit(
    umi=x.astype(np.float32),
    onehot=onehot.astype(np.float32),
    size_factor=size_factor.astype(np.float32),
    learning_rate=1e-3,
    max_epochs=30000,
    warm_start=init[:3])
  log_mu = pd.DataFrame(log_mu, columns=x.columns)
  log_phi = pd.DataFrame(log_phi, columns=x.columns)
  logodds = pd.DataFrame(logodds, columns=x.columns)
  result = []
  for k in x:
    d, p = gof(x[k].values.ravel(), cdf=zig_cdf, pmf=zig_pmf,
               size=size_factor.ravel(), log_mu=log_mu.loc[0,k],
               log_phi=log_phi.loc[0,k], logodds=logodds.loc[0,k])
    result.append((k, d, p))
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def _ash_cdf(x, a):
  """Wrap around ashr::cdf.ash"""
  return np.array(ashr.cdf_ash(a, x).rx2('y')).ravel()

def _ash_pmf(x, a):
  """Compute marginal PMF using ashr::cdf.ash"""
  Fx = _ash_cdf(x, a)
  Fx_1 = _ash_cdf(x - 1, a)
  return Fx - Fx_1

def gof_unimodal(x, **kwargs):
  result = []
  size = x.sum(axis=1)
  for k in x:
    lam = x[k] / size
    if np.isclose(lam.min(), lam.max()):
      # No variation
      raise RuntimeError
    res = ashr.ash_workhorse(
      # these are ignored by ash
      pd.Series(np.zeros(x[k].shape)),
      1,
      outputlevel='fitted_g',
      # numpy2ri doesn't DTRT, so we need to use pandas
      lik=ashr.lik_pois(y=x[k], scale=size, link='identity'),
      mixsd=pd.Series(np.geomspace(lam.min() + 1e-8, lam.max(), 25)),
      mode=pd.Series([lam.min(), lam.max()]))
    d, p = gof(x[k].values.ravel(), cdf=_ash_cdf, pmf=_ash_pmf, a=res)
    result.append((k, d, p))
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def gof_zief(x, **kwargs):
  raise NotImplementedError

def gof_npmle(x, K=100, **kwargs):
  result = []
  size = x.sum(axis=1)
  for k in x:
    lam = x[k] / size
    grid = np.linspace(0, lam.max(), K + 1)
    res = ashr.ash_workhorse(
      pd.Series(np.zeros(x.shape[0])),
      1,
      outputlevel='fitted_g',
      lik=ashr.lik_pois(y=x[k], scale=size, link='identity'),
      g=ashr.unimix(pd.Series(np.ones(K) / K), pd.Series(grid[:-1]), pd.Series(grid[1:])))
    d, p = gof(x[k].values.ravel(), cdf=_ash_cdf, pmf=_ash_pmf, a=res)
    result.append((k, d, p))
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))
    
