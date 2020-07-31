import numpy as np
import pandas as pd
import pytest
import scmodes.benchmark.llik

from .fixtures import test_data

def test__llik_point(test_data):
  x = test_data
  gene = 'ENSG00000116251'
  xj = x[gene]
  size = x.values.sum(axis=1)
  k, llik = scmodes.benchmark.llik._llik_point(gene, xj, size)
  assert k == gene
  assert np.isscalar(llik)
  assert np.isfinite(llik)
  assert llik <= 0

def test_llik_point(test_data):
  x = test_data
  res = scmodes.benchmark.llik.llik_point(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.columns).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test_llik_gamma(test_data):
  x = test_data
  res = scmodes.benchmark.llik.llik_gamma(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.columns).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test_llik_point_gamma(test_data):
  x = test_data
  res = scmodes.benchmark.llik.llik_point_gamma(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.columns).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test__llik_unimodal(test_data):
  x = test_data
  gene = 'ENSG00000116251'
  xj = x[gene]
  size = x.values.sum(axis=1)
  k, llik = scmodes.benchmark.llik._llik_unimodal(gene, xj, size)
  assert k == gene
  assert np.isscalar(llik)
  assert np.isfinite(llik)
  assert llik <= 0

def test_llik_unimodal(test_data):
  x = test_data
  res = scmodes.benchmark.llik.llik_unimodal(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.columns).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test__llik_npmle(test_data):
  x = test_data
  gene = 'ENSG00000116251'
  xj = x[gene]
  size = x.values.sum(axis=1)
  k, llik = scmodes.benchmark.llik._llik_npmle(gene, xj, size)
  assert k == gene
  assert np.isscalar(llik)
  assert np.isfinite(llik)
  assert llik <= 0

def test_llik_npmle(test_data):
  x = test_data
  res = scmodes.benchmark.llik.llik_npmle(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.columns).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test_evaluate_llik(test_data):
  x = test_data
  res = scmodes.benchmark.llik.evaluate_llik(x, methods=['point', 'gamma', 'point_gamma', 'unimodal', 'npmle'])
  assert res.shape == (5 * x.shape[1], 3)
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()
