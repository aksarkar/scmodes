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

def _frob_loss(x, lf, w=None):
  """Return the (weighted) squared Frobenius norm \\sum_{ij} w_{ij} (x_{ij} -
  (LF)_{ij})^2

  """
  if w is None:
    w = 1
  return (w * np.square(x - lf)).sum()  

def _pois_loss(x, lam, w=None):
  """Return the (weighted) Poisson negative log likelihood

  x_ij ~ Pois([LF]_ij).

  Weights are assume to be binary, denoting presence/absence.

  """
  if w is None:
    w = 1
  return -(w * (x * _safe_log(lam) - lam - sp.gammaln(x + 1))).sum()

def nmf(x, rank, w=None, pois_loss=True, max_iters=1000, atol=1e-8, eps=1e-10, verbose=False):
  """Return non-negative loadings and factors (Lee and Seung 2001).

  Returns loadings [n, rank] and factors [p, rank]

  x - array-like [n, p]
  frob - fit Gaussian model

  """
  if w is None:
    # Important: this simplifies the implementation, but is costly
    w = np.ones(x.shape)
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
    if np.isclose(update, obj, atol=atol):
      return l, f, update
    else:
      obj = update
  raise RuntimeError('failed to converge')
