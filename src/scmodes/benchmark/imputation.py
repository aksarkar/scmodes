"""Imputation methods benchmark"""
import numpy as np
import pandas as pd
import scipy.stats as st
import scmodes
import sys
import torch

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

def imputation_score_nmf(x, rank=10, frac=0.1, tol=1e-4, max_iters=100000, seed=0, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  l, f, _ = scmodes.lra.nmf(x, w=w, rank=rank, tol=tol, max_iters=max_iters)
  return _pois_loss(x, w, l.dot(f.T))

def imputation_score_glmpca(x, rank=10, frac=0.1, tol=1e-4, max_iters=100000, seed=0, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  l, f, _ = scmodes.lra.glmpca(x, w=w, rank=rank, tol=tol, max_iters=max_iters)
  return _pois_loss(x, w, np.exp(l.dot(f.T)))

def imputation_score_nbmf(x, rank=10, frac=0.1, tol=1e-4, max_iters=100000, inv_disp=1, fix_inv_disp=False, seed=0, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  l, f, inv_disp, loss = scmodes.lra.nbmf(x, w=w, rank=rank, tol=tol, max_iters=max_iters, inv_disp=inv_disp, fix_inv_disp=fix_inv_disp)
  return _pois_loss(x, w, l.dot(f.T))

def imputation_score_pvae(x, rank=10, frac=0.1, lr=1e-3, max_epochs=200, seed=0, **kwargs):
  w = _mask_entries(x, frac=frac, seed=seed)
  n, p = x.shape
  x = torch.tensor(x, dtype=torch.float)
  m = scmodes.lra.PVAE(p, rank).fit(x, lr=lr, max_epochs=max_epochs)
  return _pois_loss(x, w, m.denoise(x))

def evaluate_imputation(x, methods, n_trials=1, rank=10, **kwargs):
  result = []
  for m in methods:
    for i in range(n_trials):
      score = getattr(sys.modules[__name__], f'imputation_score_{m}')(x, seed=i, rank=rank, **kwargs)
      result.append((m, i, score))
  return pd.DataFrame(result, columns=['method', 'trial', 'loss'])
