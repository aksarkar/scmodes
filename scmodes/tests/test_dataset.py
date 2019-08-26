import numpy as np
import scmodes

def test_simulate_pois_rank1():
  x, eta = scmodes.dataset.simulate_pois(n=30, p=60, rank=1)
  assert x.shape == (30, 60)
  assert eta.shape == (30, 60)
  assert (x >= 0).all()
  assert (~np.isclose(np.linalg.svd(eta, compute_uv=False, full_matrices=False), 0)).sum() == 1

def test_simulate_pois_rank2():
  x, eta = scmodes.dataset.simulate_pois(n=30, p=60, rank=2)
  assert x.shape == (30, 60)
  assert eta.shape == (30, 60)
  assert (x >= 0).all()
  assert (~np.isclose(np.linalg.svd(eta, compute_uv=False, full_matrices=False), 0)).sum() == 2

def test_simulate_pois_masked():
  x, eta = scmodes.dataset.simulate_pois(n=30, p=60, rank=2, holdout=.25)
  assert np.ma.is_masked(x)

def test_simulate_pois_size():
  x, mu = scmodes.dataset.simulate_pois_size(n=30, p=60, s=1000, rank=1, seed=0)
  assert x.shape == (30, 60)
  assert mu.shape == (30, 60)
  assert (x >= 0).all()
  assert np.isclose(mu.sum(axis=0), 1).all()
