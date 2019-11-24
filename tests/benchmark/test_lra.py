import numpy as np
import pandas as pd
import pytest
import scmodes
import scipy.sparse as ss
import torch

from fixtures import *

def test_training_score_nmf(simulate):
  x, eta = simulate
  res = scmodes.benchmark.training_score_nmf(x, n_components=10)
  assert res <= 0

def test_training_score_glmpca(simulate):
  x, eta = simulate
  res = scmodes.benchmark.training_score_glmpca(pd.DataFrame(x), n_components=10)
  assert res <= 0

def test_training_score_pvae(simulate):
  x, eta = simulate
  res = scmodes.benchmark.training_score_pvae(pd.DataFrame(x), n_components=10)
  assert res <= 0

def test_pois_llik(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scmodes.benchmark.pois_llik(np.exp(eta), train, test)
  assert np.isscalar(res)
  assert res < 0

@pytest.mark.xfail(reason='Not implemented')
def test_pois_llik_sparse(simulate_train_test):
  train, test, eta = simulate_train_test
  train = ss.csr_matrix(train)
  test = ss.csr_matrix(test)
  res = scmodes.benchmark.pois_llik(np.exp(eta), train, test)
  assert np.isscalar(res)
  assert res < 0

def test_train_test_split(simulate):
  x, eta = simulate
  train, test = scmodes.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape

def test_train_test_split_df(simulate):
  x, eta = simulate
  x = pd.DataFrame(x)
  train, test = scmodes.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert isinstance(train, pd.DataFrame)
  assert isinstance(test, pd.DataFrame)

def test_train_test_split_sparse_csr(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csr_matrix(x.filled(0))
  train, test = scmodes.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert ss.isspmatrix_csr(train)
  assert ss.isspmatrix_csr(test)

def test_train_test_split_sparse_csc(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csc_matrix(x.filled(0))
  train, test = scmodes.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert ss.isspmatrix_csc(train)
  assert ss.isspmatrix_csc(test)

def test_generalization_score_nmf(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scmodes.benchmark.generalization_score_nmf(train, test, n_components=10)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_glmpca(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scmodes.benchmark.generalization_score_glmpca(train, test, n_components=10)
  assert np.isfinite(res)
  assert res < 0

def test_generalization_score_pvae(simulate_train_test):
  train, test, eta = simulate_train_test
  res = scmodes.benchmark.generalization_score_pvae(train, test, n_components=10)
  assert np.isfinite(res)
  assert res < 0

def test_evaluate_lra_generalization(simulate):
  x, eta = simulate
  x = pd.DataFrame(x)
  res = scmodes.benchmark.evaluate_lra_generalization(x, methods=['nmf'])
  assert not res.empty
  assert np.isfinite(res.values).all()
