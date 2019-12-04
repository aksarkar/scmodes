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

def test_imputation_score_wnmf(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_wnmf(x)
  assert np.isfinite(loss)
  assert loss > 0

def test_imputation_score_wnbmf(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_wnbmf(x)
  assert np.isfinite(loss)
  assert loss > 0

def test_imputation_score_wglmpca(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_wglmpca(x)
  assert np.isfinite(loss)
  assert loss > 0
  
def test_evaluate_imputation(simulate):
  x, eta = simulate
  result = scmodes.benchmark.evaluate_imputation(x, methods=['oracle', 'wnmf', 'wglmpca'], rank=1, n_trials=1)
  assert result.shape == (3, 3)
  assert np.isfinite(result['loss']).all()
