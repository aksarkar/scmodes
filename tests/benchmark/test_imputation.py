import scmodes

from .fixtures import *

def test_imputation_score_oracle(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_oracle(x)
  assert np.isfinite(loss)
  assert loss > 0

def test_imputation_score_ebpm_point(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_ebpm_point(x)
  assert np.isfinite(loss)
  assert loss > 0

def test_imputation_score_nmf(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_nmf(x)
  assert np.isfinite(loss)
  assert loss > 0

def test_imputation_score_nbmf(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_nbmf(x, rank=1)
  assert np.isfinite(loss)
  assert loss > 0

def test_imputation_score_glmpca(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_glmpca(x, rank=1)
  assert np.isfinite(loss)
  assert loss > 0

def test_imputation_score_pvae(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_pvae(x, rank=1)
  assert np.isfinite(loss)
  assert loss > 0
  
def test_evaluate_imputation(simulate):
  x, eta = simulate
  result = scmodes.benchmark.evaluate_imputation(x, methods=['oracle', 'nmf', 'glmpca'], rank=1, n_trials=1)
  assert result.shape == (3, 3)
  assert np.isfinite(result['loss']).all()
