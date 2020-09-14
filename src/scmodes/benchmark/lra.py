import collections
import numpy as np
import pandas as pd
import scipy.sparse as ss
import scipy.stats as st
import scmodes
import torch

def pois_llik(lam, train, test):
  if ss.issparse(train):
    raise NotImplementedError
  elif isinstance(train, pd.DataFrame):
    assert isinstance(lam, np.ndarray)
    assert isinstance(test, pd.DataFrame)
    s = test.values.sum(axis=1, keepdims=True) / train.values.sum(axis=1, keepdims=True)
  else:
    s = test.sum(axis=1, keepdims=True) / train.sum(axis=1, keepdims=True)
  return st.poisson(mu=lam).logpmf(train).mean(), np.ma.masked_invalid(st.poisson(mu=s * lam).logpmf(test)).mean()

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
  test = x - train
  if isinstance(x, pd.DataFrame):
    test = test.values
  if ss.issparse(x):
    train = train.A
    test = test.A
  return train, test

def generalization_score_nmf(train, test, n_components=10, tol=1e-4, max_iters=100000, **kwargs):
  l, f, _ = scmodes.lra.nmf(train, rank=n_components, tol=tol, max_iters=max_iters)
  lam = l @ f.T
  return pois_llik(lam, train, test)

def generalization_score_glmpca(train, test, n_components=10, tol=1e-4, max_iters=100000, **kwargs):
  l, f, _ = scmodes.lra.glmpca(train, rank=n_components, tol=tol, max_iters=max_iters)
  lam = np.exp(l @ f.T)
  return pois_llik(lam, train, test)

def generalization_score_pvae(train, test, n_components=10, lr=1e-3, max_epochs=1600, **kwargs):
  n, p = train.shape
  x = torch.tensor(train, dtype=torch.float)
  m = (scmodes.lra.PVAE(p, n_components)
       .fit(x, lr=lr, max_epochs=max_epochs))
  return pois_llik(m.denoise(x, n_samples=100), train, test)

def generalization_score_nbvae(train, test, n_components=10, lr=1e-3, max_epochs=1600, **kwargs):
  n, p = train.shape
  x = torch.tensor(train, dtype=torch.float)
  m = (scmodes.lra.NBVAE(p, n_components, disp_by_gene=True)
       .fit(x, lr=lr, max_epochs=max_epochs))
  return pois_llik(m.denoise(x, n_samples=100), train, test)

def evaluate_lra_generalization(x, methods, n_trials=1, **kwargs):
  result = collections.defaultdict(list)
  for method in methods:
    result[('train', method)] = []
    result[('validation', method)] = []
    for trial in range(n_trials):
      train, val = scmodes.benchmark.train_test_split(x)
      train_score, val_score = getattr(scmodes.benchmark, f'generalization_score_{method}')(train, val, **kwargs)
      result[('train', method)].append(train_score)
      result[('validation', method)].append(val_score)
  result = pd.DataFrame.from_dict(result)
  result.index.name = 'trial'
  return result
