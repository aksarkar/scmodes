import collections
import numpy as np
import pandas as pd
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages
import rpy2.robjects.pandas2ri
import scipy.sparse as ss
import scipy.stats as st
import scmodes
import sklearn.decomposition
import wlra

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
glmpca = rpy2.robjects.packages.importr('glmpca')

def training_score_nmf(x, n_components=10, **kwargs):
  m = sklearn.decomposition.NMF(n_components=n_components, solver='mu', beta_loss=1).fit(x)
  return st.poisson(mu=m.transform(x).dot(m.components_)).logpmf(x).sum()

def _glmpca(x, n_components, max_restarts):
  # GLMPCA can fail for some (random) initializations, so restart to find one
  # which works
  obj = None
  for i in range(max_restarts):
    try:
      # We use samples x genes, but GLM-PCA expects genes x samples
      res = glmpca.glmpca(x.values.T, L=n_components, fam='poi')
      # Follow GLM-PCA code here, not the paper
      s = np.log(x.values.mean(axis=1, keepdims=True))
      L = np.array(res.rx2('loadings'))
      F = np.array(res.rx2('factors'))
      lam = np.exp(s + F.T.dot(L))
      llik = st.poisson(mu=lam).logpmf(x.values).sum()
      print(f'glmpca {i} {llik:.3g}')
      if obj is None or llik > obj:
        obj = llik
    except:
      print(f'glmpca {i} failed')
      continue
  if obj is None:
    obj = np.nan
  return s, L, F, obj

def training_score_glmpca(x, n_components=10, max_restarts=1, **kwargs):
  res = _glmpca(x, n_components, max_restarts)
  return res[-1]

def training_score_plra1(x, n_components=10, **kwargs):
  res = wlra.plra(x.values, rank=n_components, max_iters=50000)
  return st.poisson(mu=np.exp(res)).logpmf(x.values).sum()

def pois_llik(lam, train, test):
  if ss.issparse(train):
    raise NotImplementedError
  elif isinstance(train, pd.DataFrame):
    assert isinstance(lam, np.ndarray)
    assert isinstance(test, pd.DataFrame)
    lam *= test.values.sum(axis=1, keepdims=True) / train.values.sum(axis=1, keepdims=True)
  else:
    lam *= test.sum(axis=1, keepdims=True) / train.sum(axis=1, keepdims=True)
  return st.poisson(mu=lam).logpmf(test).sum()

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
    train = np.random.binomial(n=x, p=p, size=x.shape)
  if isinstance(x, pd.DataFrame):
    train = pd.DataFrame(train, index=x.index, columns=x.columns)
  test = x - train
  return train, test

def generalization_score_nmf(train, test, n_components=10, **kwargs):
  m = sklearn.decomposition.NMF(n_components=n_components, solver='mu', beta_loss=1).fit(train)
  return pois_llik(m.transform(train).dot(m.components_), train, test)

def generalization_score_glmpca(train, test, n_components=10, max_restarts=1, **kwargs):
  s, L, F, llik = _glmpca(train, n_components, max_restarts)
  if np.isnan(llik):
    print(f'glmpca failed on training data')
  # Follow GLM-PCA paper here
  s1 = np.log(test.values.mean(axis=1, keepdims=True))
  lam = np.exp(s1 - s + F.T.dot(L))
  return st.poisson(mu=lam).logpmf(test.values).sum()

def generalization_score_plra1(train, test, n_components=10, **kwargs):
  res = wlra.plra(train.values, rank=n_components, max_iters=50000)
  return st.poisson(mu=np.exp(res)).logpmf(test.values).sum()

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
