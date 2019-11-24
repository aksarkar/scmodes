import scmodes
import numpy as np
import pandas as pd
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
  return scmodes.dataset.simulate_pois(n=30, p=60, rank=1, eta_max=3)

@pytest.fixture
def simulate_holdout():
  return scmodes.dataset.simulate_pois(n=200, p=300, rank=1, eta_max=3, holdout=.1)
  
@pytest.fixture
def simulate_train_test():
  x, eta = scmodes.dataset.simulate_pois(n=200, p=300, rank=1, eta_max=3)
  train, test = scmodes.benchmark.train_test_split(x)
  return pd.DataFrame(train), pd.DataFrame(test), eta
