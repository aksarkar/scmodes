"""Imputation methods benchmark"""
import numpy as np
import pandas as pd
import scipy.stats as st
import scmodes
import sys

def _mask_entries(x, frac, seed):
  np.random.seed(seed)
  return np.random.uniform(size=x.shape) > frac

def _pois_loss(x, w, mu):
  """Return the per-observation negative log likelihood of the masked entries"""
  # Important: oracle can produce mu == 0
  return -np.where(w, 0, st.poisson(mu=mu).logpmf(x)).mean()

def imputation_score_oracle(x, frac=0.1, seed=0, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  return _pois_loss(x, w, x)

def imputation_score_ebpm_point(x, frac=0.1, seed=0, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  s = (w * x).sum(axis=1)
  mu = (w * x).sum(axis=0) / s.sum()
  return _pois_loss(x, w, np.outer(s, mu))

def imputation_score_wnmf(x, rank=10, frac=0.1, seed=0, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  l, f, _ = scmodes.lra.nmf(x, w=w, rank=rank)
  return _pois_loss(x, w, l.dot(f.T))

def imputation_score_wglmpca(x, rank=10, frac=0.1, seed=0, max_retries=10, max_iters=5000, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  opt = None
  obj = np.inf
  for i in range(max_retries):
    try:
      l, f, loss = scmodes.lra.glmpca(x, w=w, rank=rank, max_iters=max_iters)
      if loss < obj:
        opt = l, f
        obj = loss
    except RuntimeError:
      pass
  if opt is None:
    raise RuntimeError('failed to converge after max_retries restarts')
  l, f = opt
  return _pois_loss(x, w, np.exp(l.dot(f.T)))

def imputation_score_wnbmf(x, rank=10, frac=0.1, seed=0, inv_disp=1, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  l, f, loss = scmodes.lra.nbmf(x, w=w, rank=rank, inv_disp=inv_disp)
  return _pois_loss(x, w, l.dot(f.T))

def evaluate_imputation(x, methods, n_trials=1, **kwargs):
  result = []
  for m in methods:
    for i in range(n_trials):
      score = getattr(sys.modules[__name__], f'imputation_score_{m}')(x, seed=i, **kwargs)
      result.append((m, i, score))
  return pd.DataFrame(result, columns=['method', 'trial', 'loss'])
