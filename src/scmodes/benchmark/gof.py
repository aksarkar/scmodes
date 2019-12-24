import functools as ft
import numpy as np
import pandas as pd
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import scipy.sparse as ss
import scipy.special as sp
import scipy.stats as st
import sys

rpy2.robjects.pandas2ri.activate()

ashr = rpy2.robjects.packages.importr('ashr')
descend = rpy2.robjects.packages.importr('descend')

def _gof(x, cdf, pmf, **kwargs):
  """Test for goodness of fit x_i ~ \hat{F}(.)

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

  x_i ~ Poisson(s_i \lambda_i)
  lambda_i ~ \pi_0 \delta_0(\cdot) + (1 - \pi_0) Gamma(1 / \phi, 1 / (\mu\phi))

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

def _zig_pmf(x, size, log_mu, log_phi, logodds=None):
  """Return marginal PMF of Poisson-(point) Gamma model

  x_i ~ Poisson(s_i \lambda_i)
  lambda_i ~ \pi_0 \delta_0(\cdot) + (1 - \pi_0) Gamma(1 / \phi, 1 / (\mu\phi))

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

def gof_gamma(x, s=None, batch_size=128, lr=1e-2, max_epochs=10, **kwargs):
  import scmodes.ebpm.sgd
  # Important: x is assumed to be pd.DataFrame
  x_csr = ss.csr_matrix(x.values)
  x_csc = x_csr.tocsc()
  if s is None:
    s = x_csc.sum(axis=1)
  fit = scmodes.ebpm.sgd.ebpm_gamma(x_csr, s=s, batch_size=batch_size, lr=lr, max_epochs=max_epochs)
  result = []
  for j, (gene, (log_mu, neg_log_phi)) in enumerate(zip(x.columns, np.vstack(fit[:-1]).T)):
    d, p = _gof(x_csc[:,j].A.ravel(), cdf=_zig_cdf, pmf=_zig_pmf,
                size=s, log_mu=log_mu, log_phi=-neg_log_phi)
    result.append((gene, d, p))
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'stat', 'p'])), axis='columns')
          .set_index('gene'))

def gof_zig(x, s=None, batch_size=128, lr=1e-2, max_epochs=10, **kwargs):
  import scmodes.ebpm.sgd
  # Important: x is assumed to be pd.DataFrame
  x_csr = ss.csr_matrix(x.values)
  x_csc = x_csr.tocsc()
  if s is None:
    s = x_csc.sum(axis=1)
  fit = scmodes.ebpm.sgd.ebpm_point_gamma(x_csr, s=s, batch_size=batch_size, lr=lr, max_epochs=max_epochs)
  result = []
  for j, (gene, (log_mu, neg_log_phi, logodds)) in enumerate(zip(x.columns, np.vstack(fit[:-1]).T)):
    d, p = _gof(x_csc[:,j].A.ravel(), cdf=_zig_cdf, pmf=_zig_pmf,
                size=s, log_mu=log_mu, log_phi=-neg_log_phi, logodds=logodds)
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
  # Important: use fit$data, not x
  return np.array(fit.rx2('fitted_g').rx2('pi')).dot(np.array(ashr.comp_dens_conv(fit.rx2('fitted_g'), fit.rx2('data'))))

def _gof_unimodal(k, x, size):
  """Helper function to fit one gene"""
  lam = x / size
  if np.isclose(lam.min(), lam.max()):
    # No variation
    muhat = size * x.sum() / size.sum()
    d, p = _gof(x.values.ravel(), cdf=st.poisson.cdf, pmf=st.poisson.pmf, mu=muhat)
  else:
    res = ashr.ash_pois(x, size,
      # Important: we need to deal with asymmetric distributions
      mixcompdist='halfuniform')
    d, p = _gof(x.values.ravel(), cdf=_ash_cdf, pmf=_ash_pmf, fit=res, s=size)
  return k, d, p

def gof_unimodal(x, s=None, pool=None, **kwargs):
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

def evaluate_gof(x, methods, **kwargs):
  result = {}
  for m in methods:
    # Hack: get functions by name
    result[m] = getattr(sys.modules[__name__], f'gof_{m}')(x, **kwargs)
  return pd.concat(result).reset_index().rename({'level_0': 'method'}, axis='columns')
