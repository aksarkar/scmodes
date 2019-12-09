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
import scipy.optimize as so
import scipy.special as sp
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
  # Important: scipy.stats parameterizes p(k | n, p) ∝ p^n (1 - p)^k
  return -np.where(w, st.nbinom(n=inv_disp, p=1 / (1 + lam / inv_disp)).logpmf(x), 0).sum()

def _D_loss_theta(theta, u, log_u, w):
  """Return the partial derivative of the expected log joint with respect to
theta = 1 / phi

  theta - scalar
  u - array-like (n, p)
  log_u - scalar
  w - array-like (n, p)

  """
  return (w * (1 + np.log(theta) + log_u - u - sp.digamma(theta))).sum()

def _update_inv_disp(x, w, lam, inv_disp, step=1, c=0.5, tau=0.5):
  """Backtracking line search to update inverse dispersion

  x - array-like (n, p)
  w - array-like (n, p)
  lam - array-like (n, p)
  inv_disp - scalar (>= 0)
  step - initial step size
  c - control parameter (Armijo-Goldstein condition)
  tau - control parameter (step size update)

  """
  # Important: these are expectations under the posterior
  u = (x + inv_disp) / (lam + inv_disp)
  log_u = sp.digamma(x + inv_disp) - np.log(lam + inv_disp)

  # Important: take steps wrt log_inv_disp to avoid non-negativity constraint
  log_inv_disp = np.log(inv_disp)
  d = _D_loss_theta(inv_disp, u, log_u, w) * inv_disp
  loss = _nbmf_loss(x, lam, inv_disp=inv_disp, w=w)
  update = _nbmf_loss(x, lam, inv_disp=np.exp(log_inv_disp + step * d), w=w)
  while not np.isfinite(update) or update > loss + c * step * d:
    step *= tau
    update = _nbmf_loss(x, lam, inv_disp=np.exp(log_inv_disp + step * d), w=w)
  return np.exp(log_inv_disp + step * d) + 1e-15

def nbmf(x, rank, inv_disp, init=None, w=None, max_iters=1000, tol=1, fix_inv_disp=True, verbose=False):
  """Return non-negative loadings and factors (Gouvert et al. 2018).

  Returns loadings [n, rank] and factors [p, rank]

  x - array-like [n, p]
  inv_disp - inverse dispersion (scalar)
  init - tuple (l, f), where l [n, rank] and f [p, rank]
  w - array-like [n, p]
  tol - threshold for change in log likelihood (convergence criterion)

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
    print(f'nbmf [0]: {obj} {inv_disp}')

  for i in range(max_iters):
    l *= ((w * x / lam) @ f) / ((w * (x + inv_disp) / (lam + inv_disp)) @ f)
    lam = l @ f.T
    f *= ((w * x / lam).T @ l) / ((w * (x + inv_disp) / (lam + inv_disp)).T @ l)
    lam = l @ f.T
    if not fix_inv_disp:
      inv_disp = _update_inv_disp(x, w, lam, inv_disp)
    update = _nbmf_loss(x, lam, inv_disp, w=w)
    # Important: the updates are monotonic
    assert update <= obj
    if verbose:
      print(f'nbmf [{i + 1}]: {update} {inv_disp}')
    if obj - update <= tol:
      return l, f, update
    else:
      obj = update
  raise RuntimeError('failed to converge')
