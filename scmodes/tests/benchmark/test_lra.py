import numpy as np
import pytest
import scaa
import scipy.sparse as ss
import torch

from fixtures import *

def test_simulate_pois_rank1():
  x, eta = scaa.benchmark.simulate_pois(n=30, p=60, rank=1)
  assert x.shape == (30, 60)
  assert eta.shape == (30, 60)
  assert (x >= 0).all()
  assert (~np.isclose(np.linalg.svd(eta, compute_uv=False, full_matrices=False), 0)).sum() == 1

def test_simulate_pois_rank2():
  x, eta = scaa.benchmark.simulate_pois(n=30, p=60, rank=2)
  assert x.shape == (30, 60)
  assert eta.shape == (30, 60)
  assert (x >= 0).all()
  assert (~np.isclose(np.linalg.svd(eta, compute_uv=False, full_matrices=False), 0)).sum() == 2

def test_simulate_pois_masked():
  x, eta = scaa.benchmark.simulate_pois(n=30, p=60, rank=2, holdout=.25)
  assert np.ma.is_masked(x)

def test_training_score_oracle(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_oracle(x, eta)
  assert res <= 0

def test_training_score_nmf(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_nmf(x, rank=10)
  assert res <= 0

def test_training_score_nmf_kl(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_nmf_kl(x, rank=10)
  assert res <= 0

def test_training_score_grad(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_grad(x, rank=1)
  assert res <= 0

def test_training_score_plra(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_plra(x, rank=1)
  assert res <= 0

def test_training_score_plra1(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_plra1(x)
  assert res <= 0

def test_training_score_lda(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_lda(x)
  assert res <= 0

def test_training_score_maptpx(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_maptpx(x)
  assert res <= 0

def test_training_score_hpf(simulate):
  x, eta = simulate
  try:
    res = scaa.benchmark.training_score_hpf(x)
  except ImportError:
    pytest.skip('tensorflow import failed')
  assert res <= 0

def test_training_score_scvi(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_scvi(x)
  assert res <= 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason='torch reports cuda not available')
def test_evaluate_training():
  res = scaa.benchmark.evaluate_training(num_trials=1)
  assert res.shape == (1, 6)

def test_loss():
  pred = np.random.normal(size=100)
  true = np.random.normal(size=100)
  res = scaa.benchmark.loss(pred, true)
  assert len(res) == 2

def test_imputation_score_mean(simulate_holdout):
  x, eta = simulate_holdout
  res = scaa.benchmark.imputation_score_mean(x)

def test_imputation_score_nmf(simulate_holdout):
  x, eta = simulate_holdout
  res = scaa.benchmark.imputation_score_nmf(x, rank=10)

def test_imputation_score_plra1(simulate_holdout):
  x, eta = simulate_holdout
  res = scaa.benchmark.imputation_score_plra1(x, rank=1)

def test_imputation_score_plra(simulate_holdout):
  x, eta = simulate_holdout
  res = scaa.benchmark.imputation_score_plra(x, rank=1)

def test_evaluate_pois_imputation():
  res = scaa.benchmark.evaluate_pois_imputation(eta_max=3, num_trials=1)

def test_pois_llik(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.pois_llik(np.exp(eta), train, test)
  assert np.isscalar(res)
  assert res < 0

@pytest.mark.xfail(reason='Not implemented')
def test_pois_llik_sparse(simulate_train_test):
  train, test, eta = simulate_train_test
  train = ss.csr_matrix(train)
  test = ss.csr_matrix(test)
  res = scaa.benchmark.pois_llik(np.exp(eta), train, test)
  assert np.isscalar(res)
  assert res < 0

def test_train_test_split(simulate):
  x, eta = simulate
  train, test = scaa.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape

def test_train_test_split_sparse_csr(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csr_matrix(x.filled(0))
  train, test = scaa.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert ss.isspmatrix_csr(train)
  assert ss.isspmatrix_csr(test)

def test_train_test_split_sparse_csc(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csc_matrix(x.filled(0))
  train, test = scaa.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert ss.isspmatrix_csc(train)
  assert ss.isspmatrix_csc(test)

def test_get_data_loader(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csr_matrix(x.filled(0))
  loader = scaa.benchmark.get_data_loader(x)
  assert next(iter(loader)).shape == (25, x.shape[1])

def test_get_data_loader_dtype():
  y = (np.random.uniform(size=(100, 1)) < 0.5).astype(int)
  loader = scaa.benchmark.get_data_loader(y, dtype=torch.long)
  batch = next(iter(loader))
  assert batch.shape == (25, 1)
  assert batch.dtype == torch.long  

def test_get_data_loader_batch_size():
  y = (np.random.uniform(size=(100, 1)) < 0.5).astype(int)
  loader = scaa.benchmark.get_data_loader(y, dtype=torch.long, batch_size=10)
  batch = next(iter(loader))
  assert batch.shape == (10, 1)
  assert batch.dtype == torch.long  

def test_generalization_score_oracle(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_oracle(train, test, eta=eta)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_plra1(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_plra1(train, test, eta=eta)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_nmf(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_nmf(train, test, eta=eta)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_nmf_kl(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_nmf_kl(train, test, eta=eta)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_grad(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_grad(train, test, eta=eta)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_hpf(simulate_train_test):
  train, test, eta = simulate_train_test
  try:
    res = scaa.benchmark.generalization_score_hpf(train, test)
  except ImportError:
    pytest.skip('tensorflow import failed')
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_scvi(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_scvi(train, test, eta=eta)
  assert np.isfinite(res)
  assert res < 0

@pytest.mark.skip(reason='Broken package')
def test_generalization_score_dca(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_dca(train, test, eta=eta)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_lda(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_lda(train, test)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_maptpx(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scaa.benchmark.generalization_score_maptpx(train, test)
  assert np.isfinite(res)
  assert res < 0
