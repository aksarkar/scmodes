"""Non-negative matrix factorization (NMF), supporting missing values

NMF with Frobenius norm loss is maximum likelihood estimation of L, F in

x_ij ~ N(mu_ij, sigma^2)

where mu_ij = (LF)_ij. NMF with generalized KL divergence loss is MLE in

x_ij ~ Poisson(mu_ij)

where mu_ij = (LF)_ij. In both cases, MLE can be performed using multiplicative
updates (Lee and Seung 2001), which correspond exactly to the EM algorithm
(e.g., Cemgil 2009).

To handle missing data, we introduce weights (indicators of non-missingness)
w_ij into the log likelihood, which leads to simple modifications of the
estimation algorithm (Zhang et al. 2006).

"""

import numpy as np
import scipy.special as sp

def _safe_log(x):
  """Numerically safe log"""
  return np.log(x + 1e-8)

def _frob_loss(x, lf, w):
  """Return the (weighted) squared Frobenius norm \\sum_{ij} w_{ij} (x_{ij} -
  (LF)_{ij})^2

  """
  return (w * np.square(x - lf)).sum()  

def _pois_loss(x, lam, w):
  """Return the (weighted) Poisson negative log likelihood

  x_ij ~ Pois([LF]_ij).

  Weights are assume to be binary, denoting presence/absence.

  """
  return -(w * (x * _safe_log(lam) - lam - sp.gammaln(x + 1))).sum()

def nmf(x, rank, w=None, pois_loss=True, tol=1e-4, max_iters=10000, eps=1e-10, verbose=False):
  """Return non-negative loadings and factors (Lee and Seung 2001).

  Returns loadings [n, rank] and factors [p, rank]

  x - array-like [n, p]
  w - array-like [n, p]
  pois_loss - if True, fit Poisson NMF; otherwise, fit Gaussian NMF
  tol - threshold for change in log likelihood (convergence criterion)
  max_iters - maximum number of EM updates
  eps - pseudocount (prevent division by zero)
  verbose - report likelihood after each update

  """
  if w is None:
    # Important: this simplifies the implementation, but is costly
    w = np.ones(x.shape)
  elif w.shape != x.shape:
    raise ValueError('w must be same shape as x')
  n, p = x.shape
  # Random initialization (c.f. https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/decomposition/nmf.py#L315)
  scale = np.sqrt(x.mean() / rank)
  l = np.random.uniform(1e-8, scale, size=(n, rank))
  f = np.random.uniform(1e-8, scale, size=(p, rank))
  lam = l @ f.T

  if pois_loss:
    loss = _pois_loss
  else:
    loss = _frob_loss
  obj = loss(x, lam, w=w)
  if verbose:
    print(f'nmf [0]: {obj}')

  for i in range(max_iters):
    if pois_loss:
      l *= (w * x) / (l @ f.T + eps) @ f / (w @ f)
      f *= ((w * x) / (l @ f.T + eps)).T @ l / (w.T @ l)
    else:
      l *= (w * x) @ f / (w * (l @ f.T) @ f)
      f *= (w.T * x.T) @ l / (w.T * (f @ l.T) @ l)
    lam = l @ f.T
    update = loss(x, lam, w=w)
    # Important: the updates are monotonic
    assert update <= obj
    if verbose:
      print(f'nmf [{i + 1}]: {update}')
    if obj - update <= tol:
      return l, f, update
    else:
      obj = update
  raise RuntimeError('max_iters exceeded')
