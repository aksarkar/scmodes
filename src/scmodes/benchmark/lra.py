import collections
import numpy as np
import pandas as pd
import scipy.sparse as ss
import scipy.stats as st
import scmodes
import torch

def training_score_nmf(x, n_components=10, tol=1e-4, max_iters=100000, **kwargs):
  l, f, _ = scmodes.lra.nmf(x.values, rank=n_components, tol=tol, max_iters=max_iters)
  lam = l @ f.T
  return st.poisson(mu=lam).logpmf(x).mean()

def training_score_glmpca(x, n_components=10, tol=1e-4, max_iters=100000, **kwargs):
  l, f, _ = scmodes.lra.glmpca(x.values, rank=n_components, tol=tol, max_iters=max_iters)
  lam = np.exp(l @ f.T)
  return st.poisson(mu=lam).logpmf(x).mean()

def training_score_pvae(x, n_components=10, lr=1e-3, max_epochs=200, **kwargs):
  n, p = x.shape
  s = x.values.sum(axis=1, keepdims=True)
  x = torch.tensor(x.values, dtype=torch.float)
  m = (scmodes.lra.PVAE(p, n_components)
       .fit(x, lr=lr, max_epochs=max_epochs))
  lam = m.denoise(x, n_samples=100)
  return st.poisson(mu=lam).logpmf(x).mean()

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

def generalization_score_nmf(train, test, n_components=10, tol=1e-4, max_iters=100000, **kwargs):
  l, f, _ = scmodes.lra.nmf(train.values, rank=n_components, tol=tol, max_iters=max_iters)
  lam = l @ f.T
  return pois_llik(lam, train, test)

def generalization_score_glmpca(train, test, n_components=10, tol=1e-4, max_iters=100000, **kwargs):
  l, f, _ = scmodes.lra.glmpca(train.values, rank=n_components, tol=tol, max_iters=max_iters)
  lam = np.exp(l @ f.T)
  return pois_llik(lam, train, test)

def generalization_score_pvae(train, test, n_components=10, lr=1e-3, max_epochs=200, **kwargs):
  n, p = train.shape
  x = torch.tensor(train.values, dtype=torch.float)
  m = (scmodes.lra.PVAE(p, n_components)
       .fit(x, lr=lr, max_epochs=max_epochs))
  return pois_llik(m.denoise(x), train, test)

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
