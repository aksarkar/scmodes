"""GLM-PCA, supporting missing values

We seek to fit the model

x_{ij} ~ Poisson(s_i μ_{ij})

where ln μ_{ij} = (LF)_{ij}. GLM-PCA fits this model using Fisher scoring
updates (Newton-Raphson updates, using the Fisher information instead of the
Hessian) to maximize the log likelihood (Townes 2019). To handle missing data,
we introduce weights (indicators of non-missingness) w_{ij} into the log
likelihood, which leads to simple modifications of the estimation algorithm.

"""
import numpy as np
from .wnmf import _pois_loss

def _update_l_k(l, f, k, x, w, step=1, c=0.5, tau=0.5, max_iters=30):
  """Return updated loadings and loss function

  Use backtracking line search to find the step size for the update

  l - array-like (n, m)
  f - array-like (p, m)
  k - scalar (0 <= k < m)
  x - array-like (n, p)
  w - array-like (,) or (n, p). This needs to be compatible with matrix multiplication
  step - initial step size
  c - control parameter (Armijo-Goldstein condition)
  tau - control parameter (step size update)
  max_iters - maximum number of backtracking steps

  """
  lam = np.exp(l @ f.T)
  loss = _pois_loss(x, lam, w)
  d = (w * (x - lam)) @ f[:,k] / ((w * lam) @ np.square(f[:,k]))
  update = _pois_loss(x, lam * np.exp(step * np.outer(d, f[:,k])), w)
  while (not np.isfinite(update) or (update > loss + c * step * d).any()) and max_iters > 0:
    step *= tau
    update = _pois_loss(x, lam * np.exp(step * np.outer(d, f[:,k])), w)
    max_iters -= 1
  if max_iters == 0:
    # Step size is small enough that update can be skipped
    return l[:,k], loss
  else:
    return l[:,k] + step * d, update

def _update_f_k(l, f, k, x, w, step=1, c=0.5, tau=0.5, max_iters=30):
  """Return updated factors and loss function

  Use backtracking line search to find the step size for the update

  l - array-like (n, m)
  f - array-like (p, m)
  k - scalar (0 <= k < m)
  x - array-like (n, p)
  w - array-like (,) or (n, p). This needs to be compatible with matrix multiplication
  step - initial step size
  c - control parameter (Armijo-Goldstein condition)
  tau - control parameter (step size update)
  max_iters - maximum number of backtracking steps

  """
  lam = np.exp(l @ f.T)
  loss = _pois_loss(x, lam, w)
  d = (w * (x - lam)).T @ l[:,k] / ((w.T * lam.T) @ np.square(l[:,k]))
  update = _pois_loss(x, lam * np.exp(step * np.outer(l[:,k], d)), w)
  while (not np.isfinite(update) or (update > loss + c * step * d).any()) and max_iters > 0:
    step *= tau
    update = _pois_loss(x, lam * np.exp(step * np.outer(l[:,k], d)), w)
    max_iters -= 1
  if max_iters == 0:
    # Step size is small enough that update can be skipped
    return f[:,k], loss
  else:
    return f[:,k] + step * d, update

def glmpca(x, rank, s=None, init=None, w=None, max_iters=100, atol=1e-8, verbose=False, seed=None):
  """Return loadings and factors of a log-linear factor model

  x - array-like [n, p]
  rank - scalar
  s - size factor [n, 1]
  init - (l [n, rank], f [p, rank])
  w - array-like [n, p]
  max_iters - maximum number of updates to loadings/factors
  atol - threshold for change in loss (convergence criterion)
  verbose - print loss function updates
  seed - random seed (initialization)

  """
  n, p = x.shape
  if s is None:
    s = x.sum(axis=1).reshape(-1, 1)
  else:
    assert s.shape == (n, 1)
  if w is None:
    # Important: this needs to be compatible with matrix multiplication
    w = np.array(1)
  if seed is not None:
    np.random.seed(seed)
  if init is None:
    # TODO: if this is too close to zero, the update can explode
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
    for k in range(rank):
      l[:,k], update = _update_l_k(l, f, k, x, w) 
    for k in range(rank):
      f[:,k], update = _update_f_k(l, f, k, x, w) 
    if verbose:
      print(f'wglmpca [{i + 1}]: {update}')
    if update > obj:
      # Important: this can mean the initialization was bad and the update blew
      # up
      raise RuntimeError('objective increased')
    elif obj - update < atol:
      return l, f, update
    else:
      obj = update
  raise RuntimeError('max_iters exceeded')
