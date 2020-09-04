import numpy as np
import pandas as pd
import pytest
import scmodes.benchmark.llik

from .fixtures import *

def test__llik_point(test_adata_one_gene):
  gene, xj, size = test_adata_one_gene
  k, llik = scmodes.benchmark.llik._llik_point(gene, xj, size)
  assert k == gene
  assert np.isscalar(llik)
  assert np.isfinite(llik)
  assert llik <= 0

def test_llik_point(test_adata):
  x = test_adata
  res = scmodes.benchmark.llik.llik_point(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.var[0]).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test__llik_gamma(test_adata_one_gene):
  gene, xj, size = test_adata_one_gene
  k, llik = scmodes.benchmark.llik._llik_gamma(gene, xj, size, max_iters=1000, tol=1e-3, extrapolate=True)
  assert k == gene
  assert np.isscalar(llik)
  assert np.isfinite(llik)
  assert llik <= 0

def test_llik_gamma(test_adata):
  x = test_adata
  res = scmodes.benchmark.llik.llik_gamma(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.var[0]).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test__llik_point_gamma(test_adata_one_gene):
  gene, xj, size = test_adata_one_gene
  k, llik = scmodes.benchmark.llik._llik_point_gamma(gene, xj, size, max_iters=1000, tol=1e-3, extrapolate=True)
  assert k == gene
  assert np.isscalar(llik)
  assert np.isfinite(llik)
  assert llik <= 0

def test_llik_point_gamma(test_adata):
  x = test_adata
  res = scmodes.benchmark.llik.llik_point_gamma(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.var[0]).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test__llik_unimodal(test_adata_one_gene):
  gene, xj, size = test_adata_one_gene
  k, llik = scmodes.benchmark.llik._llik_unimodal(gene, xj, size)
  assert k == gene
  assert np.isscalar(llik)
  assert np.isfinite(llik)
  assert llik <= 0

def test_llik_unimodal(test_adata):
  x = test_adata
  res = scmodes.benchmark.llik.llik_unimodal(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.var[0]).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test__llik_npmle(test_adata_one_gene):
  gene, xj, size = test_adata_one_gene
  k, llik = scmodes.benchmark.llik._llik_npmle(
    gene, xj, size, K=512, max_grid_updates=20, tol=1e-5, thresh=1e-8,
    verbose=False)
  assert k == gene
  assert np.isscalar(llik)
  assert np.isfinite(llik)
  assert llik <= 0

def test_llik_npmle(test_adata):
  x = test_adata
  res = scmodes.benchmark.llik.llik_npmle(x)
  assert res.shape == (x.shape[1], 1)
  assert (res.index == x.var[0]).all()
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()

def test_evaluate_llik(test_adata):
  x = test_adata
  res = scmodes.benchmark.llik.evaluate_llik(x, methods=['point', 'gamma', 'point_gamma', 'unimodal', 'npmle'])
  assert res.shape == (5 * x.shape[1], 3)
  assert np.isfinite(res['llik']).all()
  assert (res['llik'] <= 0).all()
