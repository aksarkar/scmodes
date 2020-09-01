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

def _ebnbm_gamma_obj(theta, x, s, mu, mu_plm, u, u_plm):
  """Return log joint probability"""
  a, b, alpha = _ebnbm_gamma_unpack(theta, x.shape[1])
  return (x * (np.log(s) + mu_plm + u_plm) - s * mu * u
          + (a - 1) * mu_plm - b * mu + a * np.log(b) - sp.gammaln(a)
          + (alpha - 1) * u_plm - alpha * u + alpha * np.log(alpha) - sp.gammaln(alpha)).sum()

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

def _ebnbm_gamma_update(theta, x, s, mu, mu_plm, u, u_plm):
  n, p = x.shape
  a, b, alpha = _ebnbm_gamma_unpack(theta, p)
  b = a / mu.mean(axis=0)
  alpha = n * p / u.sum()
  for j in range(x.shape[1]):
    a[0,j] = _ebnbm_gamma_update_a_j(a[0,j], b[0,j], mu_plm[:,j])
  # Important: we need to thread this through (SQUAR)EM updates
  mu[...] = (x + a) / (s * u + b)
  mu_plm[...] = sp.digamma(x + a) - np.log(s * u + b)
  u[...] = (x + alpha) / (s * mu + alpha)
  u_plm[...] = sp.digamma(x + alpha) - np.log(s * mu + alpha)
  return np.hstack([a.ravel(), b.ravel(), alpha])

def ebnbm_gamma(x, s, alpha=1e-3, num_samples=1000, max_iters=10000, tol=1e-3, extrapolate=True):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
  distribution

  Returns log mu and -log phi

  x - array-like [n,]
  s - array-like [n,]
  alpha - initial guess for alpha

  """
  x, s = _check_args(x, s)
  init = np.hstack([np.ones(x.shape[1]), (s.sum() / x.sum(axis=0)), alpha])
  # Important: these get passed by ref to thread through (SQUAR)EM
  mu = (x / s).astype(float)
  mu_plm = np.log(x + 1) - np.log(s)
  u = np.ones(x.shape)
  u_plm = np.zeros(x.shape)
  if extrapolate:
    theta, _ = _squarem(init, _ebnbm_gamma_obj, _ebnbm_gamma_update, x=x,
                           s=s, mu=mu, mu_plm=mu_plm, u=u, u_plm=u_plm,
                           max_iters=max_iters, tol=tol)
  else:
    theta, _ = _em(init, _ebnbm_gamma_obj, _ebnbm_gamma_update, x=x, s=s,
                      mu=mu, mu_plm=mu_plm, u=u, u_plm=u_plm,
                      max_iters=max_iters, tol=tol)
  a, b, alpha = _ebnbm_gamma_unpack(theta, x.shape[1])
  # Monte Carlo integral for llik
  mu = st.gamma(a=a, scale=1 / b).rvs(size=(num_samples, 1, x.shape[1]))
  llik = st.nbinom(n=1 / alpha, p=1 / (1 + s * mu * alpha)).logpmf(x).mean(axis=0).sum()
  return np.log(a) - np.log(b), np.log(a), np.log(alpha), llik
