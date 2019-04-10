import scaa
import numpy as np
import pytest

@pytest.fixture
def dims():
  # Data (n, p); latent representation (n, d)
  n = 50
  p = 1000
  d = 20
  stoch_samples = 10
  return n, p, d, stoch_samples

@pytest.fixture
def simulate():
  return scaa.benchmark.simulate_pois(n=30, p=60, rank=1, eta_max=3)

@pytest.fixture
def simulate_holdout():
  return scaa.benchmark.simulate_pois(n=200, p=300, rank=1, eta_max=3, holdout=.1)
  
@pytest.fixture
def simulate_train_test():
  x, eta = scaa.benchmark.simulate_pois(n=200, p=300, rank=1, eta_max=3)
  train, test = scaa.benchmark.train_test_split(x)
  return train, test, eta
