import functools as ft
import numpy as np
import pandas as pd
import rpy2.robjects.packages
import scipy.stats as st
import scmodes
import sys

def _llik_point(k, x, s, **kwargs):
  """Return marginal likelihood assuming point mass expression model for one
gene"""
  _, llik = scmodes.ebpm.ebpm_point(x.A.ravel(), s)
  return k, llik

def _llik_gamma(k, x, s, max_iters, tol, extrapolate, **kwargs):
  """Return marginal likelihood assuming Gamma expression model for one
gene

  """
  *_, llik = scmodes.ebpm.ebpm_gamma(x.A.ravel(), s, max_iters=max_iters,
                                     tol=tol, extrapolate=extrapolate)
  return k, llik

def _llik_point_gamma(k, x, s, max_iters, tol, extrapolate, **kwargs):
  """Return marginal likelihood assuming point-Gamma expression model for one
gene

  """
  *_, llik = scmodes.ebpm.ebpm_point_gamma(x.A.ravel(), s, max_iters=max_iters,
                                           tol=tol, extrapolate=extrapolate)
  return k, llik

def _llik_unimodal(k, x, s, **kwargs):
  """Return marginal likelihood assuming unimodal (non-parametric) expression
model for one gene

  """
  ashr = rpy2.robjects.packages.importr('ashr')
  lam = x.A.ravel() / s
  if np.isclose(lam.min(), lam.max()):
    return _llik_point(k, x.A.ravel(), s)
  else:
    res = scmodes.ebpm.ebpm_unimodal(x.A.ravel(), s)
    return k, np.array(res.rx2('loglik'))[0]

def _llik_npmle(k, x, s, **kwargs):
  """Return marginal likelihood assuming non-parametric expression model for one
gene

  """
  ashr = rpy2.robjects.packages.importr('ashr')
  lam = x.A.ravel() / s
  if np.isclose(lam.min(), lam.max()):
    return _llik_point(k, x.A.ravel(), s)
  else:
    res = scmodes.ebpm.ebpm_npmle(x.A.ravel(), s)
    return k, np.array(res.rx2('loglik'))[0]

def _map_llik(f, x, s=None, pool=None, **kwargs):
  """Return marginal likelihood, returned by f, for each column of x

  f - function returning (key, log likelihood) pair
  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)
  pool - multiprocessing.Pool

  """
  if s is None:
    s = x.X.sum(axis=1).A.ravel()
  result = []
  f = ft.partial(f, s=s, **kwargs)
  args = zip(x.var.iloc[:,0], x.X.T)
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
  
def llik_gamma(x, s=None, pool=None, max_iters=10000, tol=1e-7,
               extrapolate=True, **kwargs):
  """Return marginal log likelihood of Gamma expression model for each
column of x

  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)

  """
  return _map_llik(_llik_gamma, x, s, pool, max_iters=max_iters, tol=tol,
                   extrapolate=extrapolate)
  
def llik_point_gamma(x, s=None, pool=None, max_iters=10000, tol=1e-7,
                     extrapolate=True, lr=1e-2, **kwargs):
  """Return marginal log likelihood of point-Gamma expression model for each
column of x

  x - Anndata (n, p)
  s - size factor (n,) (default: total molecules per sample)
  key - column of x.var to use as key (default: first column)

  """
  return _map_llik(_llik_point_gamma, x, s, pool, max_iters=max_iters, tol=tol,
                   extrapolate=extrapolate)

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
