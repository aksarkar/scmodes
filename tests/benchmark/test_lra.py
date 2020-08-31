import numpy as np
import pandas as pd
import pytest
import scmodes
import scipy.sparse as ss
import torch

from .fixtures import *

def test_pois_llik(simulate_train_test):
  train, test, eta = simulate_train_test
  train_llik, test_llik = scmodes.benchmark.pois_llik(np.exp(eta), train, test)
  assert np.isscalar(train_llik)
  assert np.isscalar(test_llik)
  assert np.isfinite(train_llik)
  assert np.isfinite(test_llik)
  assert train_llik < 0
  assert test_llik < 0

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
  assert isinstance(train, np.ndarray)
  assert isinstance(test, np.ndarray)

def test_train_test_split_sparse_csr(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csr_matrix(x.filled(0))
  train, test = scmodes.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert not ss.isspmatrix(train)
  assert not ss.isspmatrix(test)

def test_train_test_split_sparse_csc(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csc_matrix(x.filled(0))
  train, test = scmodes.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert not ss.isspmatrix(train)
  assert not ss.isspmatrix(test)

def test_generalization_score_nmf(simulate_train_test):
  train, test, eta = simulate_train_test
  train_llik, test_llik = scmodes.benchmark.generalization_score_nmf(train, test, n_components=1)
  assert np.isscalar(train_llik)
  assert np.isscalar(test_llik)
  assert np.isfinite(train_llik)
  assert np.isfinite(test_llik)
  assert train_llik < 0
  assert test_llik < 0

def test_generalization_score_glmpca(simulate_train_test):
  train, test, eta = simulate_train_test
  np.random.seed(1)
  train_llik, test_llik = scmodes.benchmark.generalization_score_glmpca(train, test, n_components=1)
  assert np.isscalar(train_llik)
  assert np.isscalar(test_llik)
  assert np.isfinite(train_llik)
  assert np.isfinite(test_llik)
  assert train_llik < 0
  assert test_llik < 0

def test_generalization_score_pvae(simulate_train_test):
  train, test, eta = simulate_train_test
  train_llik, test_llik = scmodes.benchmark.generalization_score_pvae(train, test, n_components=1)
  assert np.isscalar(train_llik)
  assert np.isscalar(test_llik)
  assert np.isfinite(train_llik)
  assert np.isfinite(test_llik)
  assert train_llik < 0
  assert test_llik < 0

def test_evaluate_lra_generalization(simulate):
  x, eta = simulate
  x = pd.DataFrame(x)
  res = scmodes.benchmark.evaluate_lra_generalization(x, methods=['nmf'], n_components=1)
  assert not res.empty
  assert np.isfinite(res.values).all()
  assert res.shape == (1, 2)
