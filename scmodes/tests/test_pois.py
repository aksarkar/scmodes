import numpy as np
import scmodes.pois
import scipy.stats as st

def test_pois_fa_glm_rank1():
  n = 30
  p = 60
  x, mu = scmodes.dataset.simulate_pois_size(n=n, p=p, rank=1, s=1000, seed=0)
  L, F = scmodes.pois.pois_fa_glm(x, n_components=1)
  assert L.shape == (n, 1)
  assert F.shape == (p, 1)

def test_pois_fa_glm_rank1_oracle():
  n = 30
  p = 60
  x, mu = scmodes.dataset.simulate_pois_size(n=n, p=p, rank=1, s=1000, seed=0)
  s = x.sum(axis=1, keepdims=True)
  oracle_llik = st.poisson(mu=s * mu).logpmf(x).sum()
  L, F = scmodes.pois.pois_fa_glm(x, n_components=1, verbose=True)
  fit_llik = st.poisson(mu=s * np.exp(L.dot(F.T))).logpmf(x).sum()
  assert fit_llik > oracle_llik

def test_pois_fa_glm_rank10_oracle():
  n = 30
  p = 60
  x, mu = scmodes.dataset.simulate_pois_size(n=n, p=p, rank=10, s=1000, seed=0)
  s = x.sum(axis=1, keepdims=True)
  oracle_llik = st.poisson(mu=s * mu).logpmf(x).sum()
  L, F = scmodes.pois.pois_fa_glm(x, n_components=10, verbose=True)
  fit_llik = st.poisson(mu=s * np.exp(L.dot(F.T))).logpmf(x).sum()
  assert fit_llik > oracle_llik
