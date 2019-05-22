"""Python wrapper for distribution deconvolution methods"""
import numpy as np
import pandas as pd
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri
import scipy.special as sp
import scipy.stats as st
import scqtl.simple

rpy2.robjects.pandas2ri.activate()

ashr = rpy2.robjects.packages.importr('ashr')
descend = rpy2.robjects.packages.importr('descend')

def fit_gamma(x, s, num_points=1000):
  lam = x / s
  grid = np.linspace(lam.min(), lam.max(), num_points)
  res = scqtl.simple.fit_nb(x, s)
  return [grid, st.gamma(a=res[1], scale=res[0] / res[1]).cdf(grid)]

def fit_zig(x, s, num_points=1000):
  lam = x / s
  grid = np.linspace(lam.min(), lam.max(), num_points)
  res = scqtl.simple.fit_zinb(x, s)
  F = sp.expit(res[2]) + sp.expit(-res[2]) * st.gamma(a=res[1], scale=res[0] / res[1]).cdf(grid)
  return [grid, F]

def fit_unimodal(x, s, num_points=1000):
  lam = x / s
  grid = np.linspace(lam.min(), lam.max(), num_points)
  res = ashr.ash_workhorse(
    pd.Series(np.zeros(x.shape[0])),
    1,
    lik=ashr.lik_pois(y=pd.Series(x), scale=pd.Series(s), link='identity'),
    outputlevel='fitted_g',
    mixsd=pd.Series(np.geomspace(lam.min() + 1e-4, lam.max(), 25)),
    mode=pd.Series([lam.min(), lam.max()]))
  F = ashr.cdf_ash(res, grid)
  return [np.array(F.rx2('x')), np.array(F.rx2('y')).ravel()]

def fit_zief(x, s, num_points=1000):
  lam = x / s
  grid = np.linspace(lam.min(), lam.max(), num_points)
  res = descend.deconvSingle(pd.Series(x), scaling_consts=pd.Series(s), verbose=False)
  if tuple(res.rclass) != ('DESCEND',):
    return [np.array(0), np.array(0)]
  z = np.array(res.slots['distribution'])[:,0]
  Fz = np.cumsum(np.array(res.slots['distribution'])[:,1])
  return [grid, np.interp(grid, z, Fz)]

def fit_npmle(x, s, num_points=1000, K=200):
  lam = x / s
  grid = np.linspace(0, lam.max(), K + 1)
  res = ashr.ash_workhorse(
    pd.Series(np.zeros(x.shape[0])),
    1,
    outputlevel='fitted_g',
    lik=ashr.lik_pois(y=pd.Series(x), scale=pd.Series(s), link='identity'),
    g=ashr.unimix(pd.Series(np.ones(K) / K), pd.Series(grid[:-1]), pd.Series(grid[1:])))
  F = ashr.cdf_ash(res, grid)
  grid = np.linspace(lam.min(), lam.max(), num_points)
  return [np.array(F.rx2('x')), np.array(F.rx2('y')).ravel()]
