import numpy as np
import scmodes.pois
import scipy.stats as st

def test_pois_fa_glm_rank1():
  n = 30
  p = 60
  x, eta = scmodes.dataset.simulate_pois(n=n, p=p, rank=1)
  L, F = scmodes.pois.pois_fa_glm(x, n_components=1)
  assert L.shape == (n, 1)
  assert F.shape == (p, 1)

def test_pois_fa_glm_rank1_oracle():
  n = 30
  p = 60
  x, eta = scmodes.dataset.simulate_pois(n=n, p=p, rank=1)
  oracle_llik = st.poisson(mu=np.exp(eta)).logpmf(x).sum()
  L, F = scmodes.pois.pois_fa_glm(x, n_components=1, verbose=True)
  fit_llik = st.poisson(mu=np.exp(L.dot(F.T))).logpmf(x).sum()
  assert fit_llik > oracle_llik

def test_pois_fa_glm_rank10_oracle():
  n = 30
  p = 60
  # Important: this needs to be scaled, otherwise exp blows up
  x, eta = scmodes.dataset.simulate_pois(n=n, p=p, rank=10, eta_max=3)
  oracle_llik = st.poisson(mu=np.exp(eta)).logpmf(x).sum()
  # Important: initialization matters
  np.random.seed(4)
  L, F = scmodes.pois.pois_fa_glm(x, n_components=7, verbose=True)
  fit_llik = st.poisson(mu=np.exp(L.dot(F.T))).logpmf(x).sum()
  assert fit_llik > oracle_llik
  
