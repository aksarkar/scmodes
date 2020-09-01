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

def _ebnbm_gamma_unpack(theta, p):
  a = theta[:p].reshape(1, -1)
  b = theta[p:-1].reshape(1, -1)
  alpha = theta[-1]
  return a, b, alpha

def _ebnbm_gamma_obj(par, x, s, alpha, beta, gamma, delta):
  """Return ELBO (up to a constant)"""
  a, b, theta = _ebnbm_gamma_unpack(par, x.shape[1])
  return ((x + a - alpha) * (sp.digamma(alpha) - np.log(beta)) - (b - beta) * (alpha / beta)
          + (x + theta - gamma) * (sp.digamma(gamma) - np.log(delta)) - (theta - delta) * (gamma / delta)
          - s * (alpha / beta) * (gamma / delta)
          + a * np.log(b) + theta * np.log(theta) - alpha * np.log(beta) - gamma * np.log(delta)
          - sp.gammaln(a) - sp.gammaln(theta) + sp.gammaln(alpha) + sp.gammaln(gamma)).sum()

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

def _ebnbm_gamma_update_theta(init, pm, plm, step=1, c=0.5, tau=0.5, max_iters=30):
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

def _ebnbm_gamma_update(par, x, s, alpha, beta, gamma, delta):
  n, p = x.shape
  a, b, theta = _ebnbm_gamma_unpack(par, p)
  l0 = _ebnbm_gamma_obj(par, x, s, alpha, beta, gamma, delta)
  # Important: we need to thread this through VBEM updates
  alpha[...] = x + a
  beta[...] = s * (gamma / delta) + b
  gamma[...] = x + theta
  delta[...] = s * (alpha / beta) + theta
  l1 = _ebnbm_gamma_obj(par, x, s, alpha, beta, gamma, delta)
  assert np.isfinite(l1)
  assert l1 >= l0
  b = a / (alpha / beta).mean(axis=0)
  theta = _ebnbm_gamma_update_theta(theta, gamma / delta, sp.digamma(gamma) - np.log(delta))
  for j in range(x.shape[1]):
    a[0,j] = _ebnbm_gamma_update_a_j(a[0,j], b[0,j], sp.digamma(alpha[:,j]) - np.log(beta[:,j]))
  return np.hstack([a.ravel(), b.ravel(), theta])

def ebnbm_gamma(x, s, alpha=1e-3, num_samples=1000, max_iters=10000, tol=1e-3, extrapolate=True, verbose=False):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
  distribution

  Returns log mu, -log phi, log theta

  x - array-like [n,]
  s - array-like [n,]
  alpha - initial guess for alpha

  """
  x, s = _check_args(x, s)
  init = np.hstack([np.ones(x.shape[1]), (s.sum() / x.sum(axis=0)), alpha])
  # Important: these get passed by ref to thread through (SQUAR)EM
  alpha = np.ones(x.shape)
  beta = np.ones(x.shape)
  gamma = np.ones(x.shape)
  delta = np.ones(x.shape)
  if extrapolate:
    theta, elbo = _squarem(init, _ebnbm_gamma_obj, _ebnbm_gamma_update, x=x,
                           s=s, alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                           max_iters=max_iters, tol=tol)
  else:
    theta, elbo = _em(init, _ebnbm_gamma_obj, _ebnbm_gamma_update, x=x, s=s,
                      alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                      max_iters=max_iters, tol=tol, verbose=verbose)
  a, b, alpha = _ebnbm_gamma_unpack(theta, x.shape[1])
  return np.log(a) - np.log(b), np.log(a), np.log(alpha), elbo
