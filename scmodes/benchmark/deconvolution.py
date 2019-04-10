import functools as ft
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.special as sp
import sklearn.model_selection as skms

import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri

rpy2.robjects.pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()

ashr = rpy2.robjects.packages.importr('ashr')
descend = rpy2.robjects.packages.importr('descend')

def nb_llik(x, mean, inv_disp):
  return (x * np.log(mean / inv_disp) -
          x * np.log(1 + mean / inv_disp) -
          inv_disp * np.log(1 + mean / inv_disp) +
          sp.gammaln(x + inv_disp) -
          sp.gammaln(inv_disp) -
          sp.gammaln(x + 1))

def score_nb(x_train, x_test, **kwargs):
  import scqtl
  onehot = np.ones((x_train.shape[0], 1))
  size_factor = x_train.sum(axis=1).reshape(-1, 1)
  design = np.zeros((x_train.shape[0], 1))
  log_mu, log_phi, *_ = scqtl.tf.fit(
    umi=x_train.astype(np.float32),
    onehot=onehot.astype(np.float32),
    design=design.astype(np.float32),
    size_factor=size_factor.astype(np.float32),
    learning_rate=1e-3,
    max_epochs=30000)
  return nb_llik(x_test, x_test.sum(axis=1, keepdims=True) * np.exp(log_mu), np.exp(-log_phi)).sum(axis=0)

def softplus(x):
  return np.where(x > 30, x, np.log(1 + np.exp(x)))

def zinb_llik(x, mean, inv_disp, logodds):
  case_zero = -softplus(-logodds) + softplus(nb_llik(x, mean, inv_disp) - logodds)
  case_non_zero = -softplus(logodds) + nb_llik(x, mean, inv_disp)
  return np.where(x < 1, case_zero, case_non_zero)

def score_zinb(x_train, x_test, **kwargs):
  import scqtl
  onehot = np.ones((x_train.shape[0], 1))
  size_factor = x_train.sum(axis=1).reshape(-1, 1)
  init = scqtl.tf.fit(
    umi=x_train.astype(np.float32),
    onehot=onehot.astype(np.float32),
    size_factor=size_factor.astype(np.float32),
    learning_rate=1e-3,
    max_epochs=30000)
  log_mu, log_phi, logodds, *_ = scqtl.tf.fit(
    umi=x_train.astype(np.float32),
    onehot=onehot.astype(np.float32),
    size_factor=size_factor.astype(np.float32),
    learning_rate=1e-3,
    max_epochs=30000,
    warm_start=init[:3])
  return zinb_llik(x_test, x_test.sum(axis=1, keepdims=True) * np.exp(log_mu), np.exp(-log_phi), logodds).sum(axis=0)

def _score_unimix(train, test, train_size_factor, test_size_factor):
  lam = train / train_size_factor
  try:
    res0 = ashr.ash_workhorse(
      # these are ignored by ash
      pd.Series(np.zeros(train.shape)),
      1,
      outputlevel='fitted_g',
      # numpy2ri doesn't DTRT, so we need to use pandas
      lik=ashr.lik_pois(y=pd.Series(train), scale=train_size_factor, link='identity'),
      mode=pd.Series([lam.min(), lam.max()]))
    res = ashr.ash_workhorse(
      pd.Series(np.zeros(test.shape)),
      1,
      lik=ashr.lik_pois(y=pd.Series(test), scale=test_size_factor, link='identity'),
      fixg=True,
      g=res0.rx2('fitted_g'))
    ret = np.array(res.rx2('loglik'))
  except:
    ret = -np.inf
  return ret

def score_unimix(x_train, x_test, pool, **kwargs):
  result = []
  train_size_factor = pd.Series(x_train.sum(axis=1))
  test_size_factor = pd.Series(x_test.sum(axis=1))
  f = ft.partial(_score_unimix, train_size_factor=train_size_factor,
                 test_size_factor=test_size_factor)
  # np iterates over rows
  result = pool.starmap(f, zip(x_train.T, x_test.T))
  return np.array(result).ravel()

def _score_descend(train, test, train_size_factor, test_size_factor):
  res = descend.deconvSingle(pd.Series(train), scaling_consts=train_size_factor, verbose=False)
  # DESCEND returns NA on errors
  if tuple(res.rclass) != ('DESCEND',):
    return -np.inf
  g = np.array(res.slots['distribution'])[:,:2]
  # Don't marginalize over lambda = 0 for x > 0, because p(x > 0 | lambda =
  # 0) = 0
  case_nonzero = (st.poisson(mu=test_size_factor * g[1:,0])
                  .logpmf(test.reshape(-1, 1))
                  .dot(g[1:,1]))
  case_zero = (st.poisson(mu=test_size_factor * g[:,0])
               .logpmf(test.reshape(-1, 1))
               .dot(g[:,1]))
  llik = np.where(test > 0, case_nonzero, case_zero).sum()
  return llik

def score_descend(x_train, x_test, pool, **kwargs):
  result = []
  # numpy2ri doesn't DTRT, so we need to use pandas
  train_size_factor = pd.Series(x_train.sum(axis=1))
  test_size_factor = x_test.sum(axis=1).reshape(-1, 1)
  f = ft.partial(_score_descend, train_size_factor=train_size_factor,
                 test_size_factor=test_size_factor)
  result = pool.starmap(f, zip(x_train.T, x_test.T))
  return np.array(result).ravel()

def _score_npmle(train, test, train_size_factor, test_size_factor, K):
  lam = train / train_size_factor
  grid = np.linspace(0, lam.max(), K + 1)
  try:
    res0 = ashr.ash_workhorse(
      # these are ignored by ash
      pd.Series(np.zeros(train.shape)),
      1,
      outputlevel='fitted_g',
      # numpy2ri doesn't DTRT, so we need to use pandas
      lik=ashr.lik_pois(y=pd.Series(train), scale=train_size_factor, link='identity'),
      g=ashr.unimix(pd.Series(np.ones(K) / K), pd.Series(grid[:-1]), pd.Series(grid[1:])))
    res = ashr.ash_workhorse(
      pd.Series(np.zeros(test.shape)),
      1,
      lik=ashr.lik_pois(y=pd.Series(test), scale=test_size_factor, link='identity'),
      fixg=True,
      g=res0.rx2('fitted_g'))
    ret = res.rx2('loglik')
  except:
    ret = -np.inf
  return ret

def score_npmle(x_train, x_test, pool, K=100, **kwargs):
  result = []
  train_size_factor = pd.Series(x_train.sum(axis=1))
  test_size_factor = pd.Series(x_test.sum(axis=1))
  f = ft.partial(_score_npmle, train_size_factor=train_size_factor,
                 test_size_factor=test_size_factor, K=K)
  result = pool.starmap(f, zip(x_train.T, x_test.T))
  return np.array(result).ravel()

def score_saturated(x_train, x_test, **kwargs):
  return st.poisson(mu=x_test).logpmf(x_test).sum(axis=0)

def evaluate_generalization(x, pool, methods=None, **kwargs):
  result = {}
  train, val = skms.train_test_split(x, **kwargs)
  if methods is None:
    methods = ['nb', 'zinb', 'unimix', 'descend', 'npmle', 'saturated']
  for m in methods:
    # Hack: get functions by name
    result[m] = globals()[f'score_{m}'](train, val, pool=pool)
  return pd.DataFrame.from_dict(result, orient='columns')
