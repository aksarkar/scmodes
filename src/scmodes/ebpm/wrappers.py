"""EBPM via full data numerical solvers"""
import numpy as np
import pandas as pd
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st

rpy2.robjects.pandas2ri.activate()

def check_args(x, s):
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

  x - array-like [n, 1]
  s - array-like [n, 1]

  """
  x, s = check_args(x, s)
  mean = x.sum() / s.sum()
  return np.log(mean), st.poisson(mu=s * mean).logpmf(x).sum()

def _nb_obj(theta, x, s):
  """Return negative log likelihood

  x_i ~ Poisson(s_i lambda_i)
  lambda_i ~ g = Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

  theta - array-like [2,]
  x - array-like [n,]
  s - array-like [n]

  """
  mean = np.exp(theta[0])
  inv_disp = np.exp(theta[1])
  return -st.nbinom(n=inv_disp, p=1 / (1 + s * mean / inv_disp)).logpmf(x).sum()

def ebpm_gamma(x, s):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
distribution

  Returns log mu and -log phi

  x - array-like [n,]
  s - array-like [n,]

  """
  x, s = check_args(x, s)
  # Important: initialize at ebpm_point solution
  opt = so.minimize(_nb_obj, x0=[np.log(x.sum() / s.sum()), 10], args=(x, s), method='Nelder-Mead')
  if not opt.success:
    raise RuntimeError(opt.message)
  nll = opt.fun
  return opt.x[0], opt.x[1], -nll

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
  x, s = check_args(x, s)
  init = so.minimize(_nb_obj, x0=[np.log(x.sum() / s.sum()), 10], args=(x, s), method='Nelder-Mead')
  if not init.success:
    raise RuntimeError(init.message)
  opt = so.minimize(_zinb_obj, x0=[init.x[0], init.x[1], -8], args=(x, s), method='Nelder-Mead')
  if not opt.success:
    raise RuntimeError(opt.message)
  mean = np.exp(opt.x[0])
  inv_disp = np.exp(opt.x[1])
  logodds = opt.x[2]
  nll = opt.fun
  return mean, inv_disp, logodds, -nll

def ebpm_unimodal(x, s, mixcompdist='halfuniform', **kwargs):
  """Return fitted parameters and marginal log likelihood assuming g is a
unimodal distribution

  Wrap around ashr::ash_pois, and return the R object directly.

  """
  ashr = rpy2.robjects.packages.importr('ashr')
  x, s = check_args(x, s)
  return ashr.ash_pois(pd.Series(x), pd.Series(s), mixcompdist=mixcompdist, **kwargs)

def ebpm_point_expfam(x, s):
  """Return fitted parameters and marginal log likelihood assuming g is a mixture
of a point mass on zero and an exponential family parameterized by a natural
spline

  Wrap around descend::deconvSingle, and return the R object directly.

  """
  descend = rpy2.robjects.packages.importr('descend')
  return descend.deconvSingle(pd.Series(x), scaling_consts=pd.Series(s), verbose=False)
