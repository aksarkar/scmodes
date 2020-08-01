import anndata
import functools as ft
import numpy as np
import pandas as pd
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import scipy.stats as st
import scmodes
import sys

def _llik_point(k, x, s):
  """Return marginal likelihood assuming point mass expression model for one
gene"""
  _, llik = scmodes.ebpm.ebpm_point(x, s)
  return k, llik

def _llik_unimodal(k, x, s):
  """Return marginal likelihood assuming unimodal (non-parametric) expression
model for one gene

  """
  ashr = rpy2.robjects.packages.importr('ashr')
  lam = x / s
  if np.isclose(lam.min(), lam.max()):
    return _llik_point(k, x, s)
  else:
    res = scmodes.ebpm.ebpm_unimodal(x, s)
    return k, np.array(res.rx2('loglik'))[0]

def _llik_npmle(k, x, s):
  """Return marginal likelihood assuming non-parametric expression model for one
gene

  """
  ashr = rpy2.robjects.packages.importr('ashr')
  lam = x / s
  if np.isclose(lam.min(), lam.max()):
    return _llik_point(k, x, s)
  else:
    res = scmodes.ebpm.ebpm_npmle(x, s)
    return k, np.array(res.rx2('loglik'))[0]

def _map_llik(f, x, s=None, key=None, pool=None):
  """Return marginal likelihood, returned by f, for each column of x

  f - function returning (key, log likelihood) pair
  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)
  key - column of x.var to use as key (default: first column)
  pool - multiprocessing.Pool

  """
  if s is None:
    s = x.X.sum(axis=1).ravel()
  result = []
  f = ft.partial(f, s=s)
  if key is not None:
    args = zip(x.var[key], x.X.T)
  else:
    args = zip(x.var[0], x.X.T)
  if pool is not None:
    result = pool.starmap(f, args)
  else:
    result = [f(*a) for a in args]
  return (pd.DataFrame(result)
          .rename(dict(enumerate(['gene', 'llik'])), axis='columns')
          .set_index('gene'))
  
def llik_point(x, s=None, pool=None, **kwargs):
  """Return marginal log likelihood of point mass expression model for each
column of x

  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)

  """
  return _map_llik(_llik_point, x, s, pool)
  
def llik_gamma(x, s=None, key=None, batch_size=64, lr=1e-2, **kwargs):
  """Return marginal log likelihood of Gamma expression model for each
column of x

  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)
  key - column of x.var to use as key (default: first column)

  """
  x_csr, x_csc, s, genes, max_epochs = scmodes.benchmark.gof._sgd_prepare(x, s, key, batch_size)
  log_mean, log_inv_disp, _ = scmodes.ebpm.sgd.ebpm_gamma(x_csr, s=s, batch_size=batch_size, lr=lr, max_epochs=max_epochs)
  llik = []
  for j in range(x.shape[1]):
    llik.append((genes[j], st.nbinom(n=np.exp(log_inv_disp[0,j]), p=1 / (1 + s.ravel() * np.exp(log_mean[0,j] - log_inv_disp[0,j]))).logpmf(x_csc[:,j].A.ravel()).sum()))
  return pd.DataFrame(llik, columns=['gene', 'llik']).set_index('gene')
  
def llik_point_gamma(x, s=None, key=None, batch_size=64, lr=1e-2, **kwargs):
  """Return marginal log likelihood of Gamma expression model for each
column of x

  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)
  key - column of x.var to use as key (default: first column)

  """
  x_csr, x_csc, s, genes, max_epochs = scmodes.benchmark.gof._sgd_prepare(x, s, key, batch_size)
  log_mean, log_inv_disp, logodds, _ = scmodes.ebpm.sgd.ebpm_point_gamma(
    x_csr, s=s, batch_size=batch_size, lr=lr, max_epochs=max_epochs)
  llik = []
  for j in range(x.shape[1]):
    xj = x_csc[:,j].A.ravel()
    nb_llik = st.nbinom(n=np.exp(log_inv_disp[0,j]), p=1 / (1 + s.ravel() * np.exp(log_mean[0,j] - log_inv_disp[0,j]))).logpmf(xj)
    case_zero = -np.log1p(np.exp(-logodds[0,j])) + np.log1p(np.exp(nb_llik - logodds[0,j]))
    case_non_zero = -np.log1p(np.exp(logodds[0,j])) + nb_llik
    llik.append((genes[j], np.where(xj < 1, case_zero, case_non_zero).sum()))
  return pd.DataFrame(llik, columns=['gene', 'llik']).set_index('gene')

def llik_unimodal(x, s=None, pool=None, **kwargs):
  """Return marginal log likelihood of unimodal non-parametric expression model
for each column of x

  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)

  """
  return _map_llik(_llik_unimodal, x, s, pool)
  
def llik_npmle(x, s=None, pool=None, **kwargs):
  """Return marginal log likelihood of non-parametric expression model for each
column of x

  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)

  """
  return _map_llik(_llik_npmle, x, s, pool)

def evaluate_llik(x, methods, **kwargs):
  result = {}
  for m in methods:
    # Hack: get functions by name
    result[m] = getattr(sys.modules[__name__], f'llik_{m}')(x, **kwargs)
  return pd.concat(result).reset_index().rename({'level_0': 'method'}, axis='columns')
