import collections
import numpy as np
import pandas as pd
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import scipy.sparse as ss
import scipy.stats as st
import scmodes
import sklearn.decomposition as skd
import torch
import wlra
import wlra.vae

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
glmpca = rpy2.robjects.packages.importr('glmpca')

def training_score_nmf(x, n_components=10, **kwargs):
  m = skd.NMF(n_components=n_components, solver='mu', beta_loss=1).fit(x)
  return st.poisson(mu=m.transform(x).dot(m.components_)).logpmf(x).mean()

def _glmpca(x, n_components, max_restarts, penalty=0):
  # GLMPCA can fail for some (random) initializations, so restart to find one
  # which works
  obj = None
  for i in range(max_restarts):
    try:
      # We use samples x genes, but GLM-PCA expects genes x samples
      res = glmpca.glmpca(x.values.T,
                          L=n_components,
                          fam='poi',
                          penalty=penalty,
                          ctl=rpy2.robjects.vectors.ListVector({'maxIter': 1000, 'eps': 1e-15}))
      L = pd.DataFrame(res.rx2('loadings')).values
      F = pd.DataFrame(res.rx2('factors')).values
      lam = np.exp(F @ L.T)
      llik = st.poisson(mu=lam).logpmf(x.values).mean()
      print(f'glmpca {i} {llik:.3g}')
      if obj is None or llik > obj:
        obj = llik
    except Exception as e:
      print(f'glmpca {i} failed: {e.__cause__}')
      continue
  if obj is None:
    L = None
    F = None
    obj = np.nan
  # Important: return loadings (n, k) and factors (p, k)
  return F, L, obj

def training_score_glmpca(x, n_components=10, max_restarts=1, penalty=0, **kwargs):
  res = _glmpca(x, n_components=n_components, max_restarts=max_restarts, penalty=penalty)
  return res[-1]

def training_score_pvae(x, n_components=10, lr=1e-3, max_epochs=2000, **kwargs):
  n, p = x.shape
  s = x.values.sum(axis=1, keepdims=True)
  x = torch.tensor(x.values, dtype=torch.float)
  s = torch.tensor(s, dtype=torch.float)
  m = (wlra.vae.PVAE(p, n_components)
       .fit(x, s, lr=lr, max_epochs=max_epochs))
  lam = m.denoise(x)
  return st.poisson(mu=lam).logpmf(x).mean()

def training_score_wglmpca(x, n_components=10, max_restarts=1, max_iters=5000, **kwargs):
  n, p = x.shape
  opt = np.inf
  for i in range(max_restarts):
    try:
      # Important: x is assumed to be pd.DataFrame
      l, f, loss = scmodes.lra.glmpca(x.values, rank=n_components, max_iters=max_iters)
    except RuntimeError:
      continue
    if loss < opt:
      opt = loss
  # Important: scmodes.lra.glmpca return total, we want mean
  return -opt / (n * p)

def pois_llik(lam, train, test):
  if ss.issparse(train):
    raise NotImplementedError
  elif isinstance(train, pd.DataFrame):
    assert isinstance(lam, np.ndarray)
    assert isinstance(test, pd.DataFrame)
    lam *= test.values.sum(axis=1, keepdims=True) / train.values.sum(axis=1, keepdims=True)
  else:
    lam *= test.sum(axis=1, keepdims=True) / train.sum(axis=1, keepdims=True)
  return st.poisson(mu=lam).logpmf(test).mean()

def train_test_split(x, p=0.5):
  if ss.issparse(x):
    data = np.random.binomial(n=x.data.astype(np.int), p=p, size=x.data.shape)
    if ss.isspmatrix_csr(x):
      train = ss.csr_matrix((data, x.indices, x.indptr), shape=x.shape)
    elif ss.isspmatrix_csc(x):
      train = ss.csc_matrix((data, x.indices, x.indptr), shape=x.shape)
    else:
      raise NotImplementedError('sparse matrix type not supported')
  else:
    train = np.random.binomial(n=x.astype(np.int), p=p, size=x.shape)
  if isinstance(x, pd.DataFrame):
    train = pd.DataFrame(train, index=x.index, columns=x.columns)
  test = x - train
  return train, test

def generalization_score_nmf(train, test, n_components=10, **kwargs):
  m = skd.NMF(n_components=n_components, solver='mu', beta_loss=1).fit(train)
  return pois_llik(m.transform(train).dot(m.components_), train, test)

def generalization_score_glmpca(train, test, n_components=10, max_restarts=1, **kwargs):
  L, F, llik = _glmpca(train, n_components, max_restarts)
  if np.isnan(llik):
    return np.nan
  else:
    return pois_llik(np.exp(L @ F.T), train, test)

def generalization_score_pvae(train, test, n_components=10, lr=1e-3, max_epochs=2000, **kwargs):
  n, p = train.shape
  s = train.values.sum(axis=1, keepdims=True)
  x = torch.tensor(train.values, dtype=torch.float)
  s = torch.tensor(s, dtype=torch.float)
  m = (wlra.vae.PVAE(p, n_components)
       .fit(x, s, lr=lr, max_epochs=max_epochs))
  return pois_llik(m.denoise(x), train, test)

def generalization_score_wglmpca(train, test, n_components=10, max_restarts=1, max_iters=5000, **kwargs):
  opt = None
  obj = np.inf
  for i in range(max_restarts):
    try:
      l, f, loss = scmodes.lra.glmpca(train.values, rank=n_components, max_iters=max_iters)
    except RuntimeError:
      continue
    if loss < obj:
      opt = l, f
      obj = loss
  if opt is None:
    return np.nan
  else:
    return pois_llik(np.exp(l.dot(f.T)), train, test)

def evaluate_lra_generalization(x, methods, n_trials=1, **kwargs):
  result = collections.defaultdict(list)
  for method in methods:
    result[('train', method)] = []
    result[('validation', method)] = []
    for trial in range(n_trials):
      train, val = scmodes.benchmark.train_test_split(x)
      training_score = getattr(scmodes.benchmark, f'training_score_{method}')(train, **kwargs)
      result[('train', method)].append(training_score)
      validation_score = getattr(scmodes.benchmark, f'generalization_score_{method}')(train, val, **kwargs)
      result[('validation', method)].append(validation_score)
  result = pd.DataFrame.from_dict(result)
  result.index.name = 'trial'
  return result
