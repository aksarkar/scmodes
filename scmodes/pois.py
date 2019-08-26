"""Python ports of Poisson factor analysis methods"""

import numpy as np
import scipy.stats as st

def pois_fa_glm(x, n_components=10, max_iters=100, init=None, verbose=False):
  """Fit Poisson factor model using Fisher scoring

  x_ij ~ Pois(s_i exp(eta_ij))
  eta_ij = \sum_k l_ik f_jk'

  This algorithm is called GLM-PCA by Townes et al. 2019.

  """
  n, p = x.shape
  s = x.sum(axis=1, keepdims=True)
  if init is None:
    L = np.random.normal(scale=1 / n_components, size=(n, n_components))
    F = np.random.normal(scale=1 / n_components, size=(p, n_components))
  else:
    L, F = init
    assert L.shape[0] == n
    assert F.shape[0] == p
    assert L.shape[1] == F.shape[1] == n_components
  llik = st.poisson(mu=_mu(s, L, F)).logpmf(x).sum()
  print(f'(init) {llik:.4g}')
  for i in range(max_iters):
    L1 = _pois_fa_glm_update(x, s, L, F, update_l=True)
    update = st.poisson(mu=_mu(s, L1, F)).logpmf(x).sum()
    print(f'({i} L) {update:.4g}')
    F1 = _pois_fa_glm_update(x, s, L1, F, update_l=False)
    update = st.poisson(mu=_mu(s, L1, F1)).logpmf(x).sum()
    print(f'({i} F) {update:.4g}')
    if update < llik:
      pass
    elif np.isclose(llik, update):
      return L, F
    else:
      L = L1
      F = F1
      llik = update
  raise RuntimeError('failed to converge: exceeded max_iters')

def _mu(s, L, F):
  return s * np.exp(L.dot(F.T))

def _pois_fa_glm_update(x, s, L, F, update_l=True):
  """Fisher scoring updates for Poisson factor model"""
  lam = _mu(s, L, F)
  assert np.isfinite(lam).all()
  J = x - lam  # pre-compute part of the score
  if update_l:
    res = np.zeros(L.shape)
    for k in range(L.shape[1]):
      res[:,k] = L[:,k] + J.dot(F[:,k]) / lam.dot(np.square(F[:,k]))
  else:
    res = np.zeros(F.shape)
    for k in range(F.shape[1]):
      res[:,k] = F[:,k] + J.T.dot(L[:,k]) / lam.T.dot(np.square(L[:,k]))
  return res
