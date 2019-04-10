import itertools
import numpy as np
import pandas as pd
import scipy.sparse as ss
import scipy.stats as st

def training_score_oracle(x, eta):
  return st.poisson(mu=np.exp(eta)).logpmf(x).sum()

def training_score_nmf(x, rank=10):
  from wlra.nmf import nmf
  return st.poisson(mu=nmf(x, rank)).logpmf(x).sum()

def training_score_nmf_kl(x, rank=10):
  import sklearn.decomposition
  m = sklearn.decomposition.NMF(n_components=rank, solver='mu', beta_loss=1).fit(x)
  return st.poisson(mu=m.transform(x).dot(m.components_)).logpmf(x).sum()

def training_score_grad(x, rank):
  import torch
  import wlra.grad
  with torch.autograd.set_grad_enabled(True):
    m = (wlra.grad.PoissonFA(n_samples=x.shape[0], n_features=x.shape[1], n_components=rank)
         .fit(x, atol=1e-3, max_epochs=10000))
    return st.poisson(mu=np.exp(m.L.dot(m.F))).logpmf(x).sum()

def training_score_plra(x, rank):
  import wlra
  return st.poisson(mu=np.exp(wlra.plra(x, rank=rank, max_outer_iters=100, check_converged=True))).logpmf(x).sum()

def training_score_plra1(x, rank=10):
  import wlra
  lam = np.exp(wlra.plra(x, rank=rank))
  return st.poisson(mu=lam).logpmf(x).sum()

def training_score_lda(x, rank=10, learning_method='online', batch_size=100, **kwargs):
  import sklearn.decomposition
  model = sklearn.decomposition.LatentDirichletAllocation(n_components=rank, learning_method=learning_method, batch_size=batch_size, **kwargs)
  L = model.fit_transform(x)
  F = model.components_
  lam = (L / L.sum(axis=0)).dot(F)
  return st.poisson(mu=lam).logpmf(x).sum()

def training_score_maptpx(x, rank=10, **kwargs):
  import rpy2.robjects.packages
  import rpy2.robjects.numpy2ri
  rpy2.robjects.numpy2ri.activate()
  maptpx = rpy2.robjects.packages.importr('maptpx')
  res = maptpx.topics(x, K=rank, **kwargs)
  L = np.array(res.rx2('omega'))
  F = np.array(res.rx2('theta'))
  return st.poisson(mu=x.sum(axis=1, keepdims=True) * L.dot(F.T)).logpmf(x).sum()

def training_score_hpf(x, rank=50, **kwargs):
  try:
    import tensorflow as tf
  except ImportError:
    return np.nan
  import scHPF.preprocessing
  import scHPF.train
  import tempfile
  with tempfile.TemporaryDirectory(prefix='/scratch/midway2/aksarkar/ideas/') as d:
    tf.reset_default_graph()
    # scHPF assumes genes x cells
    scHPF.preprocessing.split_dataset_hpf(x.T, outdir=d)
    # Set bp, dp as in scHPF.train
    bp = x.sum(axis=1).mean() / x.sum(axis=1).var()
    dp = x.sum(axis=0).mean() / x.sum(axis=0).var()
    opt = scHPF.train.run_trials(
      indir=d, outdir=d, prefix='',
      nfactors=rank, a=0.3, ap=1, bp=bp, c=0.3, cp=1, dp=dp,
      # This is broken when we call the API directly
      logging_options={'log_phi': False})
    L = np.load(f'{opt}/beta_shape.npy') / np.load(f'{opt}/beta_invrate.npy')
    F = np.load(f'{opt}/theta_shape.npy') / np.load(f'{opt}/theta_invrate.npy')
    # We assume cells x genes
    return st.poisson(mu=F.dot(L.T)).logpmf(x).sum()

def training_score_scvi(train, **kwargs):
  from scvi.dataset import GeneExpressionDataset
  from scvi.inference import UnsupervisedTrainer
  from scvi.models import VAE
  data = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(train))
  vae = VAE(n_input=train.shape[1])
  m = UnsupervisedTrainer(vae, data, verbose=False)
  m.train(n_epochs=100)
  # Training permuted the data for minibatching. Unpermute before "imputing"
  # (estimating lambda)
  lam = np.vstack([m.train_set.sequential().imputation(),
                   m.test_set.sequential().imputation()])
  return st.poisson(mu=lam).logpmf(train).sum()

def evaluate_training(rank=3, eta_max=2, num_trials=10):
  result = []
  for trial in range(num_trials):
    x, eta = scmodes.dataset.simulate_pois(n=200, p=300, rank=rank, eta_max=eta_max, seed=trial)
    result.append([
      trial,
      training_score_oracle(x, eta),
      training_score_nmf(x, rank),
      training_score_grad(x, rank),
      training_score_plra(x, rank),
      training_score_plra1(x, rank)
    ])
  result = pd.DataFrame(result)
  result.columns = ['trial', 'Oracle', 'NMF', 'Grad', 'PLRA', 'PLRA1']
  return result

def rmse(pred, true):
  return np.sqrt(np.square(pred - true).mean())

def pois_loss(pred, true):
  return (pred - true * np.log(pred + 1e-8)).mean()

losses = [rmse, pois_loss]

def loss(pred, true):
  return [f(pred, true) for f in losses]

def imputation_score_mean(x):
  """Mean-impute the data"""
  return loss(x.mean(), x.data[x.mask])

def imputation_score_nmf(x, rank):
  try:
    from wlra.nmf import nmf
    res = nmf(x, rank, atol=1e-3)
    return loss(res[x.mask], x.data[x.mask])
  except RuntimeError:
    return [np.nan for f in losses]

def imputation_score_plra1(x, rank):
  try:
    import wlra
    res = np.exp(wlra.plra(x, rank=rank, max_outer_iters=1))
    return loss(res[x.mask], x.data[x.mask])
  except RuntimeError:
    return [np.nan for f in losses]

def imputation_score_plra(x, rank):
  try:
    import wlra
    res = np.exp(wlra.plra(x, rank=rank, max_outer_iters=100, check_converged=True))
    return loss(res[x.mask], x.data[x.mask])
  except RuntimeError:
    return [np.nan for f in losses]

def evaluate_pois_imputation(rank=3, holdout=0.25, eta_max=None, num_trials=10):
  result = []
  for trial in range(num_trials):
    x, eta = simulate_pois(n=200, p=300, rank=rank, eta_max=eta_max,
                           holdout=holdout, seed=trial)
    result.append(list(itertools.chain.from_iterable(
      [[trial],
       imputation_score_mean(x),
       imputation_score_nmf(x, rank),
       imputation_score_plra(x, rank),
       imputation_score_plra1(x, rank),
      ])))
  result = pd.DataFrame(result)
  result.columns = ['trial', 'rmse_mean', 'pois_loss_mean', 'rmse_nmf',
                    'pois_loss_nmf', 'rmse_plra', 'pois_loss_plra',
                    'rmse_plra1', 'pois_loss_plra1']
  return result

def pois_llik(lam, train, test):
  if ss.issparse(train):
    raise NotImplementedError
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
  test = x - train
  return train, test

def generalization_score_oracle(train, test, eta):
  return pois_llik(np.exp(eta), train, test)

def generalization_score_plra1(train, test, rank=10, **kwargs):
  import wlra
  lam = np.exp(wlra.plra(train, rank=rank))
  return pois_llik(lam, train, test)

def generalization_score_nmf(train, test, rank=10, **kwargs):
  from wlra.nmf import nmf
  lam = nmf(train, rank=rank)
  return pois_llik(lam, train, test)

def generalization_score_nmf_kl(train, test, n_components=10, **kwargs):
  import sklearn.decomposition
  m = sklearn.decomposition.NMF(n_components=n_components, solver='mu', beta_loss=1).fit(train)
  return pois_llik(m.transform(train).dot(m.components_), train, test)

def generalization_score_grad(train, test, rank=10, **kwargs):
  import torch
  from wlra.grad import PoissonFA
  with torch.autograd.set_grad_enabled(True):
    model = PoissonFA(n_samples=train.shape[0], n_features=train.shape[1], n_components=rank).fit(train, atol=1e-3, max_epochs=10000)
    lam = np.exp(model.L.dot(model.F))
    return pois_llik(lam, train, test)

def generalization_score_hpf(train, test, rank=50, **kwargs):
  try:
    import tensorflow as tf
  except:
    return np.nan
  import scHPF.preprocessing
  import scHPF.train
  import tempfile
  with tempfile.TemporaryDirectory(prefix='/scratch/midway2/aksarkar/ideas/') as d:
    tf.reset_default_graph()
    # scHPF assumes genes x cells
    scHPF.preprocessing.split_dataset_hpf(train.T, outdir=d)
    # Set bp, dp as in scHPF.train
    bp = train.sum(axis=1).mean() / train.sum(axis=1).var()
    dp = train.sum(axis=0).mean() / train.sum(axis=0).var()
    opt = scHPF.train.run_trials(
      indir=d, outdir=d, prefix='',
      nfactors=rank, a=0.3, ap=1, bp=bp, c=0.3, cp=1, dp=dp,
      # This is broken when we call the API directly
      logging_options={'log_phi': False})
    L = np.load(f'{opt}/beta_shape.npy') / np.load(f'{opt}/beta_invrate.npy')
    F = np.load(f'{opt}/theta_shape.npy') / np.load(f'{opt}/theta_invrate.npy')
    # We assume cells x genes
    return pois_llik(F.dot(L.T), train, test)

def generalization_score_scvi(train, test, **kwargs):
  from scvi.dataset import GeneExpressionDataset
  from scvi.inference import UnsupervisedTrainer
  from scvi.models import VAE
  data = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(train))
  vae = VAE(n_input=train.shape[1])
  m = UnsupervisedTrainer(vae, data, verbose=False)
  m.train(n_epochs=100)
  # Training permuted the data for minibatching. Unpermute before "imputing"
  # (estimating lambda)
  with torch.autograd.set_grad_enabled(False):
    lam = np.vstack([m.train_set.sequential().imputation(),
                     m.test_set.sequential().imputation()])
    return pois_llik(lam, train, test)

def generalization_score_dca(train, test, **kwargs):
  import anndata
  import scanpy.api
  data = anndata.AnnData(X=train)
  # "Denoising" is estimating lambda
  scanpy.api.pp.dca(data, mode='denoise')
  lam = data.X
  return pois_llik(lam, train, test)

def generalization_score_lda(train, test, n_components=10, learning_method='online', batch_size=100, **kwargs):
  import sklearn.decomposition
  model = sklearn.decomposition.LatentDirichletAllocation(n_components=n_components, learning_method=learning_method, batch_size=batch_size, **kwargs)
  L = model.fit_transform(train)
  F = model.components_
  lam = (L / L.sum(axis=0)).dot(F)
  return pois_llik(lam, train, test)

def generalization_score_maptpx(train, test, rank=10, **kwargs):
  import rpy2.robjects.packages
  import rpy2.robjects.numpy2ri
  rpy2.robjects.numpy2ri.activate()
  maptpx = rpy2.robjects.packages.importr('maptpx')
  res = maptpx.topics(train, K=rank, **kwargs)
  L = np.array(res.rx2('omega'))
  F = np.array(res.rx2('theta'))
  lam = train.sum(axis=1, keepdims=True) * L.dot(F.T)
  return pois_llik(lam, train, test)
