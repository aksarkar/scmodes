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
  """Return the negative log likelihood of the masked entries, up to a constant"""
  return np.where(w, 0, mu - x * np.log(mu)).sum()

def imputation_score_wnmf(x, rank=10, frac=0.1, seed=0):
  w = _mask_entries(x, frac=frac, seed=seed)
  l, f, _ = scmodes.lra.nmf(x, w=w, rank=rank)
  return _pois_loss(x, w, l.dot(f.T))

def imputation_score_wglmpca(x, rank=10, frac=0.1, seed=0, max_retries=10, max_iters=5000):
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

def evaluate_imputation(x, methods, n_trials=1, **kwargs):
  result = []
  for m in methods:
    for i in range(n_trials):
      score = getattr(sys.modules[__name__], f'imputation_score_{m}')(x, seed=i, **kwargs)
      result.append((m, i, score))
  return pd.DataFrame(result, columns=['method', 'trial', 'loss'])
