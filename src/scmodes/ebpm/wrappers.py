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

def _em(x0, objective_fn, update_fn, max_iters=100, tol=1e-3, *args, **kwargs):
  x = x0
  obj = objective_fn(x0, *args, **kwargs)
  for i in range(max_iters):
    x = update_fn(x, *args, **kwargs)
    update = objective_fn(x, *args, **kwargs)
    if update < obj:
      raise RuntimeError('llik decreased')
    elif update - obj < tol:
      return x, update
    else:
      obj = update
  else:
    raise RuntimeError(f'failed to converge in max_iters ({update - obj:.3g} > {tol:.3g})')

def _squarem(x0, objective_fn, update_fn, max_iters=100, tol=1e-3, *args, **kwargs):
  """Squared extrapolation method"""
  x = x0
  obj = objective_fn(x, *args, **kwargs)
  for i in range(max_iters):
    x1 = update_fn(x)
    diff1 = x1 - x
    if abs(diff1).sum() < tol:
      return x1
    x2 = update_fn(x1)
    diff2 = x2 - x1
    if abs(diff2).sum() < tol:
      return x2
    x = p + 2 * diff1 + (diff2 - diff1)
  else:
    raise RuntimeError(f'failed to converge in max_iters ({update - obj:.3g} > {tol:.3g})')

def _ebpm_gamma_obj(theta, x, s):
  a, b = theta
  return st.nbinom(n=a, p=1 / (1 + s / b)).logpmf(x).sum()

def _ebpm_gamma_update(theta, x, s):
  a, b = theta
  pm = (x + a) / (s + b)
  plm = sp.digamma(x + a) - np.log(s + b)
  b = a / pm.mean()
  # Important: this appears to be given incorrectly in Karlis 2005
  a += (np.log(b) - sp.digamma(a) + plm.mean()) / sp.polygamma(1, a)
  return np.array([a, b])

def ebpm_gamma(x, s, max_iters=100, tol=1e-3, extrapolate=False):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
distribution

  Returns log mu and -log phi

  x - array-like [n,]
  s - array-like [n,]

  """
  x, s = _check_args(x, s)
  # a = 1 / phi; b = 1 / (mu phi)
  # Initialize at the Poisson MLE
  theta = np.array([1, s.sum() / x.sum()])
  if extrapolate:
    raise NotImplementedError
  else:
    theta, llik = _em(theta, _ebpm_gamma_obj, _ebpm_gamma_update, x=x, s=s,
                      max_iters=max_iters, tol=tol)
  a, b = theta
  return -np.log(b) + np.log(a), np.log(a), llik

def _zinb_obj(theta, x, s):
  """Return negative log likelihood

  x_i ~ Poisson(s_i lambda_i)
  lambda_i ~ g = sigmoid(logodds) \delta_0(.) + sigmoid(-logodds) Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

  theta - array-like [3,]
  x - array-like [n,]
  s - array-like [n,]

  """
  mean = np.exp(theta[0])
  inv_disp = np.exp(theta[1])
  logodds = theta[2]
  nb = st.nbinom(n=inv_disp, p=1 / (1 + s * mean / inv_disp)).logpmf(x)
  case_zero = -np.log1p(np.exp(-logodds)) + np.log1p(np.exp(nb - logodds))
  case_nonzero = -np.log1p(np.exp(logodds)) + nb
  return -np.where(x < 1, case_zero, case_nonzero).sum()

def ebpm_point_gamma(x, s):
  """Return fitted parameters and marginal log likelihood assuming g is a
point-Gamma distribution

  Returns log mu, -log phi, logit pi

  x - array-like [n, p]
  s - array-like [n, 1]

  """
  x, s = _check_args(x, s)
  init = ebpm_gamma(x, s)
  opt = so.minimize(_zinb_obj, x0=[init[0], init[1], -8], args=(x, s), method='Nelder-Mead')
  if not opt.success:
    raise RuntimeError(opt.message)
  nll = opt.fun
  return opt.x[0], opt.x[1], opt.x[2], -nll

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
