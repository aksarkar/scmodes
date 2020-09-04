import anndata
import numpy as np
import os
import pandas as pd
import pytest
import scipy.sparse as ss
import scmodes

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
  return train, test, eta

@pytest.fixture
def test_data():
  x = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'counts.txt.gz'), sep='\t', index_col=0)
  return x

@pytest.fixture
def test_adata():
  x = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'counts.txt.gz'), sep='\t', index_col=0)
  y = anndata.AnnData(ss.csr_matrix(x.values), obs=pd.DataFrame(x.index), var=pd.DataFrame(x.columns))
  return y

@pytest.fixture
def test_adata_one_gene():
  x = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'counts.txt.gz'), sep='\t', index_col=0)
  y = anndata.AnnData(ss.csr_matrix(x.values), obs=pd.DataFrame(x.index), var=pd.DataFrame(x.columns))
  gene = 'ENSG00000116251'
  yj = y[:,0].X
  size = y.X.sum(axis=1).A.ravel()
  return gene, yj, size
  
