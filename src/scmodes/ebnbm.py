"""Empirical Bayes Negative Binomial Means

"""
import numpy as np
import scipy.special as sp
import scipy.stats as st

from scmodes.ebpm.wrappers import _em, _squarem

def _check_args(x, s):
  n, p = x.shape
  s = np.array(s)
  if s.shape != () and s.shape != (n, 1):
    raise ValueError
  if s.shape == ():
    s = np.ones(n) * s
  return x, s

def _ebnbm_gamma_unpack(par, p):
  a = par[:p].reshape(1, -1)
  b = par[p:-1].reshape(1, -1)
  theta = par[-1]
  return a, b, theta

def _ebnbm_gamma_obj(par, x, s, alpha, beta, gamma, delta, **kwargs):
  """Return ELBO (up to a constant)"""
  a, b, theta = _ebnbm_gamma_unpack(par, x.shape[1])
  eta = 1 / theta
  return ((x + a - alpha) * (sp.digamma(alpha) - np.log(beta)) - (b - beta) * (alpha / beta)
          + (x + eta - gamma) * (sp.digamma(gamma) - np.log(delta)) - (eta - delta) * (gamma / delta)
          - s * (alpha / beta) * (gamma / delta)
          + a * np.log(b) + eta * np.log(eta) - alpha * np.log(beta) - gamma * np.log(delta)
          - sp.gammaln(a) - sp.gammaln(eta) + sp.gammaln(alpha) + sp.gammaln(gamma)).sum()

def _ebnbm_gamma_update_a_j(init, b_j, plm, step=1, c=0.5, tau=0.5, max_iters=30):
  """Backtracking line search to select step size for Newton-Raphson update of
  a_j

  """
  def loss(a):
    return -(a * np.log(b_j) + a * plm - sp.gammaln(a)).sum()
  obj = loss(init)
  d = (np.log(b_j) - sp.digamma(init) + plm).mean() / sp.polygamma(1, init)
  update = loss(init + step * d)
  while (not np.isfinite(update) or update > obj + c * step * d) and max_iters > 0:
    step *= tau
    update = loss(init + step * d)
    max_iters -= 1
  if max_iters == 0:
    # Step size is small enough that update can be skipped
    return init
  else:
    return init + step * d

def _ebnbm_gamma_update_eta(init, pm, plm, step=1, c=0.5, tau=0.5, max_iters=30):
  """Backtracking line search to select step size for Newton-Raphson update of
  theta

  """
  def loss(a):
    return -(a * np.log(a) + a * plm - a * pm - sp.gammaln(a)).sum()
  obj = loss(init)
  d = (1 + plm - pm - sp.digamma(init)).mean() / sp.polygamma(1, init)
  update = loss(init + step * d)
  while (not np.isfinite(update) or update > obj + c * step * d) and max_iters > 0:
    step *= tau
    update = loss(init + step * d)
    max_iters -= 1
  if max_iters == 0:
    # Step size is small enough that update can be skipped
    return init
  else:
    return init + step * d

def _ebnbm_gamma_update(par, x, s, alpha, beta, gamma, delta, fix_g, fix_theta):
  n, p = x.shape
  a, b, theta = _ebnbm_gamma_unpack(par, p)
  eta = 1 / theta
  l0 = _ebnbm_gamma_obj(par, x, s, alpha, beta, gamma, delta)
  # Important: we need to thread this through VBEM updates
  alpha[...] = x + a
  beta[...] = s * (gamma / delta) + b
  gamma[...] = x + eta
  delta[...] = s * (alpha / beta) + eta
  l1 = _ebnbm_gamma_obj(par, x, s, alpha, beta, gamma, delta)
  assert l1 >= l0
  if not fix_g:
    b = a / (alpha / beta).mean(axis=0)
    for j in range(x.shape[1]):
      a[0,j] = _ebnbm_gamma_update_a_j(a[0,j], b[0,j], sp.digamma(alpha[:,j]) - np.log(beta[:,j]))
  if not fix_theta:
    # Important: update is simpler wrt 1 / theta
    eta = _ebnbm_gamma_update_eta(eta, gamma / delta, sp.digamma(gamma) - np.log(delta))
  l2 = _ebnbm_gamma_obj(np.hstack([a.ravel(), b.ravel(), 1 / eta]), x, s, alpha, beta, gamma, delta)
  assert l2 >= l1
  return np.hstack([a.ravel(), b.ravel(), 1 / eta])

def ebnbm_gamma(x, s, init=None, max_iters=10000, tol=1e-3, extrapolate=True,
                fix_g=False, fix_theta=True):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
  distribution

  Returns log mu, -log phi, log theta

  x - array-like [n,]
  s - array-like [n,]

  """
  x, s = _check_args(x, s)
  # Important: theta is NB dispersion => 1 / theta is Gamma shape
  if init is None:
    init = np.hstack([1e-3 * np.ones(x.shape[1]), (s.sum() / x.sum(axis=0)), 1e-3])
  else:
    assert init.shape == (2 * x.shape[1] + 1,)
  # Important: these get passed by ref to thread through (SQUAR)EM
  alpha = np.ones(x.shape)
  beta = np.ones(x.shape)
  gamma = np.ones(x.shape)
  delta = np.ones(x.shape)
  if extrapolate:
    par, elbo = _squarem(init, _ebnbm_gamma_obj, _ebnbm_gamma_update, x=x, s=s,
                         alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                         fix_g=fix_g, fix_theta=fix_theta, max_iters=max_iters,
                         tol=tol)
  else:
    par, elbo = _em(init, _ebnbm_gamma_obj, _ebnbm_gamma_update, x=x, s=s,
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                    fix_g=fix_g, fix_theta=fix_theta, max_iters=max_iters,
                    tol=tol)
  # Add back the constant that was left out for computational efficiency
  elbo += (x * np.log(s) - sp.gammaln(x + 1)).sum()
  a, b, theta = _ebnbm_gamma_unpack(par, x.shape[1])
  return np.log(a) - np.log(b), np.log(a), np.log(theta), alpha, beta, gamma, delta, elbo
