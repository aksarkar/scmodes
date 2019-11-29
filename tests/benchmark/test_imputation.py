import scmodes

from .fixtures import *

def test_imputation_score_nmf(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_nmf(x)
  assert np.isfinite(loss)
  assert loss > 0

def test_imputation_score_glmpca(simulate):
  x, eta = simulate
  loss = scmodes.benchmark.imputation_score_glmpca(x)
  assert np.isfinite(loss)
  assert loss > 0
  
def test_evaluate_imputation(simulate):
  x, eta = simulate
  result = scmodes.benchmark.evaluate_imputation(x, methods=['nmf', 'glmpca'], rank=1, n_trials=1)
  assert result.shape == (2, 3)
  assert np.isfinite(result['loss']).all()
