"""Negative Binomial Matrix Factorization (NBMF), supporting missing values

NBMF (Gouvert et al. 2018) is the model

x_{ij} ~ Poisson(u_ij (LF)_{ij})
u_{ij} ~ Gamma(1 / phi, 1 / phi)

Assuming phi fixed, this model admits an EM algorithm for L, F. The expected
log joint is analytic, so a numerical update to phi is possible. To handle
missing data, we introduce weights (indicators of non-missingness) w_ij into
the log likelihood, which leads to simple modifications of the estimation
algorithm.

"""
import numpy as np
import scipy.stats as st

def _nbmf_loss(x, lam, inv_disp, w=None):
  """Return the (weighted) negative log likelihood

  x - array-like [n, p]
  lam - array-like [n, p]
  inv_disp - scalar
  w - array-like [n, p]

  """
  if w is None:
    w = 1
  return -np.where(w, st.nbinom(n=inv_disp, p=1 / (1 + lam * inv_disp)).logpmf(x), 0).sum()

def nbmf(x, rank, inv_disp, init=None, w=None, max_iters=1000, atol=1e-8, fix_inv_disp=True, verbose=False):
  """Return non-negative loadings and factors (Gouvert et al. 2018).

  Returns loadings [n, rank] and factors [p, rank]

  x - array-like [n, p]
  inv_disp - log inverse dispersion (scalar)
  init - tuple (l, f), where l [n, rank] and f [p, rank]
  w - array-like [n, p]

  """
  if w is None:
    # Important: this simplifies the implementation, but is costly
    w = np.ones(x.shape)
  n, p = x.shape
  if init is None:
    # Random initialization (c.f. https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/decomposition/nmf.py#L315)
    scale = np.sqrt(x.mean() / rank)
    l = np.random.uniform(1e-8, scale, size=(n, rank))
    f = np.random.uniform(1e-8, scale, size=(p, rank))
  else:
    l, f = init
    assert l.shape == (n, rank)
    assert f.shape == (p, rank)
  lam = l @ f.T

  obj = _nbmf_loss(x, lam, inv_disp, w=w)
  if verbose:
    print(f'nbmf [0]: {obj}')

  for i in range(max_iters):
    l *= ((x / lam) @ f) / (((x + inv_disp) / (lam + inv_disp)) @ f)
    lam = l @ f.T
    f *= ((x / lam).T @ l) / (((x + inv_disp) / (lam + inv_disp)).T @ l)
    lam = l @ f.T
    if not fix_inv_disp:
      raise NotImplementedError('estimation of inv_disp not implemented')
    update = _nbmf_loss(x, lam, inv_disp, w=w)
    # Important: the updates are monotonic
    assert update <= obj
    if verbose:
      print(f'nbmf [{i + 1}]: {update}')
    if np.isclose(update, obj, atol=atol):
      return l, f, update
    else:
      obj = update
  raise RuntimeError('failed to converge')
