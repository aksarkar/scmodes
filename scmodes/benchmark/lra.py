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
import wlra

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
glmpca = rpy2.robjects.packages.importr('glmpca')

def training_score_nmf(x, n_components=10, **kwargs):
  m = skd.NMF(n_components=n_components, solver='mu', beta_loss=1).fit(x)
  return st.poisson(mu=m.transform(x).dot(m.components_)).logpmf(x).mean()

def _glmpca(x, n_components, max_restarts):
  # GLMPCA can fail for some (random) initializations, so restart to find one
  # which works
  obj = None
  s = np.log(x.values.mean(axis=1, keepdims=True))
  for i in range(max_restarts):
    try:
      # We use samples x genes, but GLM-PCA expects genes x samples
      res = glmpca.glmpca(x.values.T, L=n_components, fam='poi')
      # Follow GLM-PCA code here, not the paper
      L = np.array(res.rx2('loadings'))
      F = np.array(res.rx2('factors'))
      lam = np.exp(s + F.T.dot(L))
      llik = st.poisson(mu=lam).logpmf(x.values).mean()
      print(f'glmpca {i} {llik:.3g}')
      if obj is None or llik > obj:
        obj = llik
    except:
      print(f'glmpca {i} failed')
      continue
  if obj is None:
    L = None
    F = None
    obj = np.nan
  return s, L, F, obj

def training_score_glmpca(x, n_components=10, max_restarts=1, **kwargs):
  res = _glmpca(x, n_components, max_restarts)
  return res[-1]

def training_score_scvi(x, n_components=10, **kwargs):
  from scvi.dataset import GeneExpressionDataset
  from scvi.inference import UnsupervisedTrainer
  from scvi.models import VAE
  data = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(x.values))
  vae = VAE(n_input=x.shape[1])
  m = UnsupervisedTrainer(vae, data, verbose=False)
  m.train(n_epochs=100)
  # Training permuted the data for minibatching. Unpermute before "imputing"
  # (estimating lambda)
  lam = np.vstack([m.train_set.sequential().imputation(),
                   m.test_set.sequential().imputation()])
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
    train = np.random.binomial(n=x, p=p, size=x.shape)
  if isinstance(x, pd.DataFrame):
    train = pd.DataFrame(train, index=x.index, columns=x.columns)
  test = x - train
  return train, test

def generalization_score_nmf(train, test, n_components=10, **kwargs):
  m = skd.NMF(n_components=n_components, solver='mu', beta_loss=1).fit(train)
  return pois_llik(m.transform(train).dot(m.components_), train, test)

def generalization_score_glmpca(train, test, n_components=10, max_restarts=1, **kwargs):
  _, L, F, llik = _glmpca(train, n_components, max_restarts)
  if np.isnan(llik):
    return np.nan
  else:
    # Follow GLM-PCA code here
    s = np.log(test.values.mean(axis=1, keepdims=True))
    lam = np.exp(s + F.T.dot(L))
    return st.poisson(mu=lam).logpmf(test.values).mean()

def generalization_score_scvi(train, test, n_components=10, **kwargs):
  from scvi.dataset import GeneExpressionDataset
  from scvi.inference import UnsupervisedTrainer
  from scvi.models import VAE
  data = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(train.values))
  vae = VAE(n_input=train.shape[1])
  m = UnsupervisedTrainer(vae, data, verbose=False)
  m.train(n_epochs=100)
  # Training permuted the data for minibatching. Unpermute before "imputing"
  # (estimating lambda)
  lam = np.vstack([m.train_set.sequential().imputation(),
                   m.test_set.sequential().imputation()])
  return pois_llik(lam, train, test)

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
