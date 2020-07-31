import anndata
import functools as ft
import numpy as np
import pandas as pd
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import scipy.sparse as ss
import scipy.special as sp
import scipy.stats as st
import scmodes
import sys

rpy2.robjects.pandas2ri.activate()

def _gof(x, cdf, pmf, **kwargs):
  """Test for goodness of fit x_i ~ \\hat{F}(.)

  If x_i ~ F(.), then F(x_i) ~ Uniform(0, 1). Use randomized predictive
  p-values to handle discontinuous F.

  Parameters:

  x - array-like (n,)
  cdf - function returning F(x)
  pmf - function returning f(x)
  kwargs - arguments to cdf, pmf

  """
  F = cdf(x - 1, **kwargs)
  f = pmf(x, **kwargs)
  q = _rpp(F, f)
  return st.kstest(q, 'uniform')

def _rpp(cdf, pmf):
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

def _zig_cdf(x, size, log_mu, log_phi, logodds=None):
  """Return marginal CDF of Poisson-(point) Gamma model

  x_i ~ Poisson(s_i \\lambda_i)
  lambda_i ~ \\pi_0 \\delta_0(\\cdot) + (1 - \\pi_0) Gamma(1 / \\phi, 1 / (\\mu\\phi))

  size - scalar or array-like (n,)
  log_mu - scalar
  log_phi - scalar
  logodds - scalar

  """
  n = np.exp(-log_phi)
  p = 1 / (1 + (size * np.exp(log_mu + log_phi)))
  if logodds is not None:
    pi0 = sp.expit(logodds)
  else:
    pi0 = 0
  cdf = st.nbinom(n=n, p=p).cdf(x)
  # We need to handle x = -1
  cdf = np.where(x >= 0, pi0 + (1 - pi0) * cdf, cdf)
  return cdf

def gof_point(x, s=None, **kwargs):
  """Fit and test for departure from Poisson distribution for each column of x

  x - pd.DataFrame (n, p)
  s - size factor (n,) (default: total molecules per sample)

  """
  if s is None:
    s = x.values.sum(axis=1).ravel()
  result = []
  for gene in x:
    log_mu, _ = scmodes.ebpm.ebpm_point(x[gene], s)
    fit = st.poisson(mu=s * np.exp(log_mu))
    d, p = _gof(x[gene], cdf=fit.cdf, pmf=fit.pmf)
    result.append((gene, d, p))
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def _zig_pmf(x, size, log_mu, log_phi, logodds=None):
  """Return marginal PMF of Poisson-(point) Gamma model

  x_i ~ Poisson(s_i \\lambda_i)
  lambda_i ~ \\pi_0 \\delta_0(\\cdot) + (1 - \\pi_0) Gamma(1 / \\phi, 1 / (\\mu\\phi))

  size - scalar or array-like (n,)
  log_mu - scalar
  log_phi - scalar
  logodds - scalar

  """
  n = np.exp(-log_phi)
  p = 1 / (1 + (size * np.exp(log_mu + log_phi)))
  pmf = st.nbinom(n=n, p=p).pmf(x)
  if logodds is not None:
    pi0 = sp.expit(logodds)
    pmf *= (1 - pi0)
    pmf[x == 0] += pi0
  return pmf

def _sgd_prepare(x, s, key, batch_size):
  if isinstance(x, pd.DataFrame):
    x_csr = ss.csr_matrix(x.values)
    genes = x.columns
  elif isinstance(x, anndata.AnnData):
    if not ss.isspmatrix_csr(x.X):
      x_csr = ss.csr_matrix(x.X)
    else:
      x_csr = x.X
    if key is None:
      genes = x.var.iloc[:,0]
    else:
      genes = x.var[key]
  elif not ss.isspmatrix_csr(x):
    x_csr = ss.csr_matrix(x)
  else:
    x_csr = x
  x_csc = x_csr.tocsc()
  if s is None:
    s = x_csc.sum(axis=1).A
  # Heuristic: fix the total number of updates
  max_epochs = 6000 * batch_size // x.shape[0]
  return x_csr, x_csc, s, genes, max_epochs

def gof_gamma(x, s=None, key=None, batch_size=64, lr=1e-2, **kwargs):
  """Fit and test for departure from Poisson-Gamma distribution for each column of x

  x - array-like (n, p) (can be e.g. pd.DataFrame or anndata.AnnData)
  s - size factor (n, 1) (default: total molecules per sample)
  key - if x is anndata.AnnData, column of x.var to use as key (default: first column)

  """
  x_csr, x_csc, s, genes, max_epochs = _sgd_prepare(x, s, key, batch_size)
  fit = scmodes.ebpm.sgd.ebpm_gamma(x_csr, s=s, batch_size=batch_size, lr=lr, max_epochs=max_epochs)
  result = []
  for j, (gene, (log_mu, neg_log_phi)) in enumerate(zip(genes, np.vstack(fit[:-1]).T)):
    d, p = _gof(x_csc[:,j].A.ravel(), cdf=_zig_cdf, pmf=_zig_pmf,
                size=s.ravel(), log_mu=log_mu, log_phi=-neg_log_phi)
    result.append((gene, d, p))
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def gof_zig(x, s=None, key=None, batch_size=64, lr=1e-2, **kwargs):
  """Fit and test for departure from Poisson-point-Gamma distribution for each column of x

  x - array-like (n, p) (can be e.g. pd.DataFrame or anndata.AnnData)
  s - size factor (n, 1) (default: total molecules per sample)
  key - if x is anndata.AnnData, column of x.var to use as key (default: first column)

  """
  x_csr, x_csc, s, genes, max_epochs = _sgd_prepare(x, s, key, batch_size)
  fit = scmodes.ebpm.sgd.ebpm_point_gamma(x_csr, s=s, batch_size=batch_size, lr=lr, max_epochs=max_epochs)
  result = []
  for j, (gene, (log_mu, neg_log_phi, logodds)) in enumerate(zip(genes, np.vstack(fit[:-1]).T)):
    d, p = _gof(x_csc[:,j].A.ravel(), cdf=_zig_cdf, pmf=_zig_pmf,
                size=s.ravel(), log_mu=log_mu, log_phi=-neg_log_phi, logodds=logodds)
    result.append((gene, d, p))
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def _ash_cdf(x, fit, s, thresh=1e-8):
  """Compute marginal CDF of the data"""
  # Ref: https://lsun.github.io/truncash/diagnostic_plot.html#ash:_normal_likelihood,_uniform_mixture_prior
  a = np.array(fit.rx2('fitted_g').rx2('a'))
  b = np.array(fit.rx2('fitted_g').rx2('b'))
  pi = np.array(fit.rx2('fitted_g').rx2('pi'))
  N = x.shape[0]
  K = a.shape[0]
  F = np.zeros((N, K))
  for i in range(N):
    for k in range(K):
      if pi[k] < thresh:
        continue
      elif x[i] < 0:
        # Important: we need to handle x = -1
        F[i,k] = 0
      elif np.isclose(a[k], b[k]):
        F[i,k] = st.poisson(mu=s[i] * a[k]).cdf(x[i])
      else:
        ak = min(a[k], b[k])
        bk = max(a[k], b[k])
        # Marginal PMF involves Gamma CDF. Important: arange excludes endpoint
        F_gamma = st.gamma(a=np.arange(1, x[i] + 2), scale=1 / s[i]).cdf
        F[i,k] = (F_gamma(bk) - F_gamma(ak)).sum() / (s[i] * (bk - ak))
      # Important: floating point addition could lead to F[i,k] > 1, but not
      # too much larger
      assert F[i,k] <= 1 or np.isclose(F[i,k], 1)
  return F.dot(pi)

def _ash_pmf(x, fit, **kwargs):
  """Compute marginal PMF of the data"""
  ashr = rpy2.robjects.packages.importr('ashr')
  # Important: use fit$data, not x
  return np.array(fit.rx2('fitted_g').rx2('pi')).dot(np.array(ashr.comp_dens_conv(fit.rx2('fitted_g'), fit.rx2('data'))))

def _gof_unimodal(k, x, size):
  """Helper function to fit one gene"""
  ashr = rpy2.robjects.packages.importr('ashr')
  lam = x / size
  if np.isclose(lam.min(), lam.max()):
    # No variation
    muhat = size * x.sum() / size.sum()
    d, p = _gof(x.values.ravel(), cdf=st.poisson.cdf, pmf=st.poisson.pmf, mu=muhat)
  else:
    res = scmodes.ebpm.ebpm_unimodal(x, size)
    d, p = _gof(x.values.ravel(), cdf=_ash_cdf, pmf=_ash_pmf, fit=res, s=size)
  return k, d, p

def gof_unimodal(x, s=None, pool=None, **kwargs):
  """Fit and test for departure from Poisson-unimodal distribution for each column of x

  x - pd.DataFrame (n, p)
  s - size factor (n,) (default: total molecules per sample)
  pool - multiprocessing.Pool object

  """
  result = []
  if s is None:
    s = x.sum(axis=1)
  else:
    # Important: data must be coerced to Series to pass through rpy2
    s = pd.Series(s)
  f = ft.partial(_gof_unimodal, size=s)
  if pool is not None:
    result = pool.starmap(f, x.iteritems())
  else:
    result = [f(*args) for args in x.iteritems()]
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def _point_expfam_cdf(x, res, size):
  if tuple(res.rclass) != ('DESCEND',):
    raise ValueError('res is not a DESCEND object')
  n = x.shape[0]
  assert x.shape == (n,)
  assert size.shape == (n,)
  g = np.array(res.slots['distribution'])[:,:2]
  F = np.zeros(x.shape[0])
  for i in range(x.shape[0]):
    if x[i] < 0:
      F[i] = 0
    elif x[i] == 0:
      F[i] = st.poisson(mu=size[i] * g[:,0]).pmf(x[i]).dot(g[:,1])
    else:
      F[i] = st.poisson(mu=size[i] * g[:,0]).pmf(np.arange(x[i] + 1).reshape(-1, 1)).dot(g[:,1]).sum()
  return F

def _point_expfam_pmf(x, size, res):
  if tuple(res.rclass) != ('DESCEND',):
    raise ValueError('res is not a DESCEND object')
  n = x.shape[0]
  assert x.shape == (n,)
  assert size.shape == (n,)
  g = np.array(res.slots['distribution'])[:,:2]
  # Don't marginalize over lambda = 0 for x > 0, because
  # p(x > 0 | lambda = 0) = 0
  return np.where(x > 0,
                  st.poisson(mu=np.outer(size, g[1:,0])).pmf(x.reshape(-1, 1)).dot(g[1:,1]),
                  st.poisson(mu=np.outer(size, g[:,0])).pmf(x.reshape(-1, 1)).dot(g[:,1]))

def _gof_npmle(k, x, size):
  """Helper function to fit one gene"""
  ashr = rpy2.robjects.packages.importr('ashr')
  lam = x / size
  if np.isclose(lam.min(), lam.max()):
    # No variation
    muhat = size * x.sum() / size.sum()
    d, p = _gof(x.values.ravel(), cdf=st.poisson.cdf, pmf=st.poisson.pmf, mu=muhat)
  else:
    res = scmodes.ebpm.ebpm_npmle(x, size)
    d, p = _gof(x.values.ravel(), cdf=_ash_cdf, pmf=_ash_pmf, fit=res, s=size)
  return k, d, p

def gof_npmle(x, s=None, pool=None, **kwargs):
  """Fit and test for departure from Poisson-nonparametric distribution for each column of x

  x - pd.DataFrame (n, p)
  s - size factor (n,) (default: total molecules per sample)
  pool - multiprocessing.Pool object

  """
  result = []
  if s is None:
    s = x.sum(axis=1)
  else:
    # Important: data must be coerced to Series to pass through rpy2
    s = pd.Series(s)
  f = ft.partial(_gof_npmle, size=s)
  if pool is not None:
    result = pool.starmap(f, x.iteritems())
  else:
    result = [f(*args) for args in x.iteritems()]
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def _point_expfam_cdf(x, res, size):
  if tuple(res.rclass) != ('DESCEND',):
    raise ValueError('res is not a DESCEND object')
  n = x.shape[0]
  assert x.shape == (n,)
  assert size.shape == (n,)
  g = np.array(res.slots['distribution'])[:,:2]
  F = np.zeros(x.shape[0])
  for i in range(x.shape[0]):
    if x[i] < 0:
      F[i] = 0
    elif x[i] == 0:
      F[i] = st.poisson(mu=size[i] * g[:,0]).pmf(x[i]).dot(g[:,1])
    else:
      F[i] = st.poisson(mu=size[i] * g[:,0]).pmf(np.arange(x[i] + 1).reshape(-1, 1)).dot(g[:,1]).sum()
  return F

def evaluate_gof(x, methods, **kwargs):
  result = {}
  for m in methods:
    # Hack: get functions by name
    result[m] = getattr(sys.modules[__name__], f'gof_{m}')(x, **kwargs)
  return pd.concat(result).reset_index().rename({'level_0': 'method'}, axis='columns')


def _lr(k, x, size):
  """Helper function to fit one gene"""
  ashr = rpy2.robjects.packages.importr('ashr')
  lam = x / size
  if np.isclose(lam.min(), lam.max()):
    return k, 0
  else:
    res0 = scmodes.ebpm.ebpm_unimodal(x, size)
    res1 = scmodes.ebpm.ebpm_npmle(x, size)
    return k, np.array(res1.rx2('loglik'))[0] - np.array(res0.rx2('loglik'))[0]

def evaluate_lr(x, s, pool=None):
  result = []
  f = ft.partial(_lr, size=s)
  if pool is not None:
    result = pool.starmap(f, x.iteritems())
  else:
    result = [f(*args) for args in x.iteritems()]
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'llr'])), axis='columns')
          .set_index('gene'))
