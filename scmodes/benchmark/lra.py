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

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
glmpca = rpy2.robjects.packages.importr('glmpca')

def training_score_nmf(x, n_components=10):
  m = sklearn.decomposition.NMF(n_components=n_components, solver='mu', beta_loss=1).fit(x)
  return st.poisson(mu=m.transform(x).dot(m.components_)).logpmf(x).sum()

def training_score_lda(x, n_components=10, learning_method='online', batch_size=100, **kwargs):
  model = sklearn.decomposition.LatentDirichletAllocation(n_components=n_components, learning_method=learning_method, batch_size=batch_size, **kwargs)
  L = model.fit_transform(x)
  F = model.components_
  lam = (L / L.sum(axis=0)).dot(F)
  return st.poisson(mu=lam).logpmf(x).sum()

def training_score_glmpca(x, n_components=10):
  # We use samples x genes, but GLM-PCA expects genes x samples
  res = glmpca.glmpca(x.values.T, L=n_components, fam='poi')
  # Follow GLM-PCA code here, not the paper
  s = np.log(x.values.sum(axis=0) / x.mean(axis=1).sum()).reshape(-1, 1)
  L = np.array(res.rx2('loadings'))
  F = np.array(res.rx2('factors'))
  lam = np.exp(s.reshape(1, -1) + F.T.dot(L))
  return st.poisson(mu=lam).logpmf(x.values).sum()

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

def generalization_score_lda(train, test, n_components=10, learning_method='online', batch_size=100, **kwargs):
  import sklearn.decomposition
  model = sklearn.decomposition.LatentDirichletAllocation(n_components=n_components, learning_method=learning_method, batch_size=batch_size, **kwargs)
  L = model.fit_transform(train)
  F = model.components_
  lam = (L / L.sum(axis=0)).dot(F)
  return pois_llik(lam, train, test)

def generalization_score_glmpca(train, test, n_components=10):
  # We use samples x genes, but GLM-PCA expects genes x samples
  res = glmpca.glmpca(train.values.T, L=n_components, fam='poi')
  # Follow GLM-PCA code here, not the paper
  s = np.log(test.values.sum(axis=0) / test.mean(axis=1).sum()).reshape(-1, 1)
  L = np.array(res.rx2('loadings'))
  F = np.array(res.rx2('factors'))
  lam = np.exp(s.reshape(1, -1) + F.T.dot(L))
  return st.poisson(mu=lam).logpmf(test.values).sum()

def evaluate_lra_generalization(x, methods, n_trials=1):
  result = collections.defaultdict(list)
  for method in methods:
    result[('train', method)] = []
    result[('validation', method)] = []
    for trial in range(n_trials):
      train, val = scmodes.benchmark.train_test_split(x)
      try:
        training_score = getattr(scmodes.benchmark, f'training_score_{method}')(train)
        result[('train', method)].append(training_score)
      except:
        result[('train', method)].append(np.nan)
      try:
        validation_score = getattr(scmodes.benchmark, f'generalization_score_{method}')(train, val)
        result[('validation', method)].append(validation_score)
      except:
        result[('validation', method)].append(np.nan)
  result = pd.DataFrame.from_dict(result)
  result.index.name = 'trial'
  return result
