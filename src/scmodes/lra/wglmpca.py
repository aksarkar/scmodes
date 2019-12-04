"""GLM-PCA, supporting missing values

We seek to fit the model

x_{ij} ~ Poisson(\\mu_{ij})

where \\ln \\mu_{ij} = (LF)_{ij}. GLM-PCA fits this model using Fisher scoring
updates (Newton-Raphson updates, using the Fisher information instead of the
Hessian) to maximize the log likelihood (Townes 2019). To handle missing data,
we introduce weights (indicators of non-missingness) w_{ij} into the log
likelihood, which leads to simple modifications of the estimation algorithm.

"""
import numpy as np
from .wnmf import _pois_loss

def glmpca(x, rank, init=None, w=None, max_iters=100, atol=1e-8, verbose=False, seed=None):
  """Return loadings and factors of a log-linear factor model

  x - array-like [n, p]
  init - (l [n, rank], f [p, rank])
  w - array-like [n, p]

  """
  n, p = x.shape
  if w is None:
    w = np.array(1)
  if seed is not None:
    np.random.seed(seed)
  if init is None:
    l = np.random.normal(size=(n, rank))
    f = np.random.normal(size=(p, rank))
  else:
    l, f = init
    assert l.shape == (n, rank)
    assert f.shape == (p, rank)
  # TODO: this can have severe numerical problems
  lam = np.exp(l @ f.T)
  obj = _pois_loss(x, lam, w=w)
  if verbose:
    print(f'wglmpca [0]: {obj}')
  for i in range(max_iters):
    l += (w * (x - lam)) @ f / ((w * lam) @ np.diag(f @ f.T).reshape(-1, 1))
    lam = np.exp(l @ f.T)
    f += (w * (x - lam)).T @ l / ((w.T * lam.T) @ np.diag(l @ l.T).reshape(-1, 1))
    lam = np.exp(l @ f.T)
    update = _pois_loss(x, lam, w=w)
    if verbose:
      print(f'wglmpca [{i}]: {update}')
    if update > obj:
      # Important: this can mean the initialization was bad and the update blew
      # up
      raise RuntimeError('objective increased')
    elif np.isclose(obj, update, atol=atol):
      return l, f, update
    else:
      obj = update
  raise RuntimeError('failed to converge')
