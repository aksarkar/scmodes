"""EBPM via full data numerical optimization

These methods are suitable to solve EBPM for one gene at a time, but will not
(immediately) scale to large numbers of genes.

We include wrappers around ashr and DESCEND for convenience.

"""
import numpy as np
import pandas as pd
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st

rpy2.robjects.pandas2ri.activate()

def _check_args(x, s):
  n = x.shape[0]
  if x.shape != (n,):
    raise ValueError
  s = np.array(s)
  if s.shape != () and s.shape != x.shape:
    raise ValueError
  if s.shape == ():
    s = np.ones(x.shape) * s
  return x, s

def ebpm_point(x, s):
  """Return fitted parameters and marginal log likelihood assuming g is a point
mass

  Returns log mu

  x - array-like [n,]
  s - array-like [n,]

  """
  x, s = _check_args(x, s)
  mean = x.sum() / s.sum()
  return np.log(mean), st.poisson(mu=s * mean).logpmf(x).sum()

def _em(init, objective_fn, update_fn, max_iters, tol, *args, **kwargs):
  theta = init.copy()
  obj = objective_fn(theta, *args, **kwargs)
  for i in range(max_iters):
    theta = update_fn(theta, *args, **kwargs)
    update = objective_fn(theta, *args, **kwargs)
    diff = update - obj
    if i == 0 and diff < 0:
      # Hack: this is needed for numerical reasons, because in e.g.,
      # ebpm_gamma, a point mass is the limit as b → ∞
      return init, obj
    elif diff < 0:
      raise RuntimeError('llik decreased')
    elif diff < tol:
      return theta, update
    else:
      obj = update
  else:
    raise RuntimeError(f'failed to converge in max_iters ({diff:.4g} > {tol:.4g})')

def _squarem(init, objective_fn, update_fn, max_iters, tol, max_step_updates=10, *args, **kwargs):
  """Squared extrapolation scheme for accelerated EM

  Reference: 

    Varadhan, R. and Roland, C. (2008), Simple and Globally Convergent Methods
    for Accelerating the Convergence of Any EM Algorithm. Scandinavian Journal
    of Statistics, 35: 335-353. doi:10.1111/j.1467-9469.2007.00585.x

  """
  theta = init
  obj = objective_fn(theta, *args, **kwargs)
  for i in range(max_iters):
    x1 = update_fn(theta, *args, **kwargs)
    r = x1 - theta
    if i == 0 and objective_fn(x1, *args, **kwargs) < obj:
      # Hack: this is needed for numerical reasons, because in e.g.,
      # ebpm_gamma, a point mass is the limit as a = 1/φ → ∞
      return init, obj
    x2 = update_fn(x1, *args, **kwargs)
    v = (x2 - x1) - r
    step = -np.sqrt(r @ r) / np.sqrt(v @ v)
    if step > -1:
      step = -1
      theta += - 2 * step * r + step * step * v
      update = objective_fn(theta, *args, **kwargs)
      diff = update - obj
    else:
      # Step length = -1 is EM; use as large a step length as is feasible to
      # maintain monotonicity
      for j in range(max_step_updates):
        candidate = theta - 2 * step * r + step * step * v
        update = objective_fn(candidate, *args, **kwargs)
        diff = update - obj
        if np.isfinite(update) and diff > 0:
          theta = candidate
          break
        else:
          step = (step - 1) / 2
      else:
        # No acceptable step length found, reject the update
        return theta, obj
    if diff < tol:
      return theta, update
    else:
      obj = update
  else:
    raise RuntimeError(f'failed to converge in max_iters ({diff:.3g} > {tol:.3g})')

def _ebpm_gamma_obj(theta, x, s):
  a, b = theta
  return st.nbinom(n=a, p=1 / (1 + s / b)).logpmf(x).sum()

def _ebpm_gamma_update(theta, x, s):
  a, b = theta
  pm = (x + a) / (s + b)
  plm = sp.digamma(x + a) - np.log(s + b)
  b = a / pm.mean()
  # Important: this appears to be given incorrectly in Karlis 2005
  a += (np.log(b) - sp.digamma(a) + plm).mean() / sp.polygamma(1, a)
  return np.array([a, b])

def ebpm_gamma(x, s, max_iters=10000, tol=1e-3, extrapolate=True):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
distribution

  Returns log mu and -log phi

  x - array-like [n,]
  s - array-like [n,]

  """
  x, s = _check_args(x, s)
  # a = 1 / phi; b = 1 / (mu phi)
  # Initialize at the Poisson MLE
  init = np.array([1, s.sum() / x.sum()])
  if extrapolate:
    theta, llik = _squarem(init, _ebpm_gamma_obj, _ebpm_gamma_update, x=x, s=s,
                           max_iters=max_iters, tol=tol)
  else:
    theta, llik = _em(init, _ebpm_gamma_obj, _ebpm_gamma_update, x=x, s=s,
                      max_iters=max_iters, tol=tol)
  if np.isclose(theta, init).all():
    # First EM update rejected, so just use the point mass model
    log_mean, llik = ebpm_point(x, s)
    return log_mean, np.inf, llik
  else:
    a, b = theta
    return np.log(a) - np.log(b), np.log(a), llik

def _ebpm_point_gamma_obj(theta, x, s):
  """Return negative log likelihood

  x_i ~ Poisson(s_i lambda_i)
  lambda_i ~ g = sigmoid(logodds) \delta_0(.) + sigmoid(-logodds) Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

  theta - array-like [3,]
  x - array-like [n,]
  s - array-like [n,]

  """
  logodds, a, b = theta
  nb = st.nbinom(n=a, p=1 / (1 + s / b)).logpmf(x)
  case_zero = -np.log1p(np.exp(logodds)) + np.log1p(np.exp(nb + logodds))
  case_nonzero = -np.log1p(np.exp(-logodds)) + nb
  return np.where(x < 1, case_zero, case_nonzero).sum()

def _ebpm_point_gamma_update(theta, x, s):
  logodds, a, b = theta
  p = sp.expit(logodds)
  nb_lik = st.nbinom(n=a, p=1 / (1 + s / b)).pmf(x)
  z = np.where(x < 1, p * nb_lik / (1 - p + p * nb_lik), 1)
  pm = (x + a) / (s + b)
  plm = sp.digamma(x + a) - np.log(s + b)
  logodds = np.log(z.sum()) - np.log((1 - z).sum())
  b = a * z.sum() / (z * pm).sum()
  a += (z * (np.log(b) - sp.digamma(a) + plm)).sum() / (z * sp.polygamma(1, a)).sum()
  return np.array([logodds, a, b])

def ebpm_point_gamma(x, s, max_iters=10000, tol=1e-3, extrapolate=True):
  """Return fitted parameters and marginal log likelihood assuming g is a
point-Gamma distribution

  Returns log mu, -log phi, logit pi

  x - array-like [n, p]
  s - array-like [n, 1]

  """
  x, s = _check_args(x, s)
  init = np.array([8, 1, s.sum() / x.sum()])
  if extrapolate:
    theta, llik = _squarem(init, _ebpm_point_gamma_obj,
                           _ebpm_point_gamma_update, x=x, s=s,
                           max_iters=max_iters, tol=tol)
  else:
    theta, llik = _em(init, _ebpm_point_gamma_obj, _ebpm_point_gamma_update,
                      x=x, s=s, max_iters=max_iters, tol=tol)
  logodds, a, b = theta
  return np.log(a) - np.log(b), np.log(a), logodds, llik

def ebpm_unimodal(x, s, mixcompdist='halfuniform', **kwargs):
  """Return fitted parameters and marginal log likelihood assuming g is a
unimodal distribution

  Wrap around ashr::ash_pois, and return the R object directly.

  kwargs - arguments to ashr::ash_pois

  """
  ashr = rpy2.robjects.packages.importr('ashr')
  x, s = _check_args(x, s)
  return ashr.ash_pois(pd.Series(x), pd.Series(s), mixcompdist=mixcompdist, **kwargs)

def ebpm_point_expfam(x, s):
  """Return fitted parameters and marginal log likelihood assuming g is a mixture
of a point mass on zero and an exponential family parameterized by a natural
spline

  Wrap around descend::deconvSingle, and return the R object directly.

  """
  descend = rpy2.robjects.packages.importr('descend')
  return descend.deconvSingle(pd.Series(x), scaling_consts=pd.Series(s),
                              do_LRT_test=False, plot_density=False, verbose=False)

def ebpm_npmle(x, s, K=100):
  """Return fitted parameters and marginal log likelihood assuming g is an
arbitrary distribution on non-negative reals

  Wrap around ashr::ash_pois, and return the R object directly.

  K - number of grid points

  """
  ashr = rpy2.robjects.packages.importr('ashr')
  x, s = _check_args(x, s)
  lam = x / s
  grid = np.linspace(0, lam.max(), K + 1)
  return ashr.ash_pois(
    pd.Series(x), pd.Series(s),
    g=ashr.unimix(pd.Series(np.ones(K) / K), pd.Series(grid[:-1]), pd.Series(grid[1:])))
