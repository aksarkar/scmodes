"""Python ports of Poisson factor analysis methods"""

import numpy as np
import scipy.stats as st

def pois_fa_glm(x, n_components=10, max_iters=1000, init=None, verbose=False):
  """Fit Poisson factor model using IRLS

  This is called GLM-PCA by Townes et al. 2019.

  """
  n, p = x.shape
  # Pre-compute the saturated model llik
  llik_saturated = st.poisson(mu=x).logpmf(x).sum()
  if init is None:
    L = np.random.normal(scale=1 / n_components, size=(n, n_components))
    F = np.random.normal(scale=1 / n_components, size=(p, n_components))
  else:
    L, F = init
    assert L.shape[0] == n
    assert F.shape[0] == p
    assert L.shape[1] == F.shape[1] == n_components
  obj = np.inf
  for i in range(max_iters):
    L1 = _pois_fa_glm_update(x, L, F, update_l=True)
    F1 = _pois_fa_glm_update(x, L1, F, update_l=False)
    # Don't follow Townes et al. here. The size factors will be included in L
    llik = st.poisson(mu=np.exp(L1.dot(F1.T))).logpmf(x).sum()
    update = 2 * (llik_saturated - llik)
    trace = np.exp(L1.dot(F1.T)).max()
    if verbose:
      print(f'({i}) {llik:.3g} {update:.3g} {trace:.3g}')
    if update > obj:
      pass
    elif np.isclose(obj, update):
      return L, F
    else:
      L = L1
      F = F1
      obj = update
  raise RuntimeError('failed to converge: exceeded max_iters')

def _pois_fa_glm_update(x, L, F, penalty=0, update_l=True):
  """IRLS updates for Poisson factor model"""
  H = np.exp(L.dot(F.T))  # expected information
  assert np.isfinite(H).all()
  J = x - H  # gradient
  if update_l:
    for k in range(L.shape[1]):
      L[:,k] = L[:,k] + (J.dot(F[:,k]) - penalty * L[:,k]) / (H.dot(np.square(F[:,k])) + penalty)
    return L
  else:
    for k in range(F.shape[1]):
      F[:,k] = F[:,k] + (J.T.dot(L[:,k]) - penalty * F[:,k]) / (H.T.dot(np.square(L[:,k])) + penalty)
    return F
