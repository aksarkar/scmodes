import numpy as np
import scipy.stats as st
import scmodes.ebnbm
from ebpm.fixtures import *

@pytest.fixture
def simulate_nb_gamma():
  np.random.seed(1)
  n = 100
  p = 5
  s = 1e5 * np.ones((n, 1))
  theta = 0.2
  log_mu = np.random.uniform(-12, -6, size=(1, p))
  log_phi = np.random.uniform(-6, 0, size=(1, p))
  G = st.gamma(a=np.exp(-log_phi), scale=np.exp(log_mu + log_phi))
  lam = G.rvs(size=(n, p))
  x = st.nbinom(n=1 / theta, p=1 / (1 + s * lam * theta)).rvs()
  return x, s, log_mu, log_phi, theta

def test_ebnbm_gamma_em(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  init = np.hstack([np.exp(-log_phi).ravel(), np.exp(-log_phi - log_mu).ravel(), 1e-3])
  log_mu_hat, log_phi_hat, log_theta_hat, alpha, beta, gamma, delta, elbo = scmodes.ebnbm.ebnbm_gamma(
    x, s, init=init, tol=1e-5, extrapolate=False)
  assert log_mu_hat.shape == log_mu.shape
  assert np.isfinite(log_mu_hat).all()
  assert log_phi_hat.shape == log_phi.shape
  assert np.isfinite(log_phi_hat).all()
  assert log_theta_hat.shape == ()
  assert np.isfinite(log_theta_hat)
  assert alpha.shape == x.shape
  assert np.isfinite(alpha).all()
  assert (alpha > 0).all()
  assert beta.shape == x.shape
  assert np.isfinite(beta).all()
  assert (beta > 0).all()
  assert gamma.shape == x.shape
  assert np.isfinite(gamma).all()
  assert (gamma > 0).all()
  assert delta.shape == x.shape
  assert np.isfinite(delta).all()
  assert (delta > 0).all()
  assert elbo.shape == ()
  assert np.isfinite(elbo)
  assert elbo <= 0

def test_ebnbm_gamma_em_nb_measurement(simulate_nb_gamma):
  x, s, log_mu, log_phi, theta = simulate_nb_gamma
  init = np.hstack([np.exp(-log_phi).ravel(), np.exp(-log_phi - log_mu).ravel(), theta])
  log_mu_hat, log_phi_hat, log_theta_hat, alpha, beta, gamma, delta, elbo = scmodes.ebnbm.ebnbm_gamma(
    x, s, init=init, tol=1e-5, extrapolate=False)
  assert log_mu_hat.shape == log_mu.shape
  assert np.isfinite(log_mu_hat).all()
  assert log_phi_hat.shape == log_phi.shape
  assert np.isfinite(log_phi_hat).all()
  assert log_theta_hat.shape == ()
  assert np.isfinite(log_theta_hat)
  assert alpha.shape == x.shape
  assert np.isfinite(alpha).all()
  assert (alpha > 0).all()
  assert beta.shape == x.shape
  assert np.isfinite(beta).all()
  assert (beta > 0).all()
  assert gamma.shape == x.shape
  assert np.isfinite(gamma).all()
  assert (gamma > 0).all()
  assert delta.shape == x.shape
  assert np.isfinite(delta).all()
  assert (delta > 0).all()
  assert elbo.shape == ()
  assert np.isfinite(elbo)
  assert elbo <= 0

def test_ebnbm_gamma_em_default_init(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  log_mu_hat, log_phi_hat, log_theta_hat, alpha, beta, gamma, delta, elbo = scmodes.ebnbm.ebnbm_gamma(
    x, s, tol=100, extrapolate=False)
  assert log_mu_hat.shape == log_mu.shape
  assert np.isfinite(log_mu_hat).all()
  assert log_phi_hat.shape == log_phi.shape
  assert np.isfinite(log_phi_hat).all()
  assert log_theta_hat.shape == ()
  assert np.isfinite(log_theta_hat)
  assert alpha.shape == x.shape
  assert np.isfinite(alpha).all()
  assert (alpha > 0).all()
  assert beta.shape == x.shape
  assert np.isfinite(beta).all()
  assert (beta > 0).all()
  assert gamma.shape == x.shape
  assert np.isfinite(gamma).all()
  assert (gamma > 0).all()
  assert delta.shape == x.shape
  assert np.isfinite(delta).all()
  assert (delta > 0).all()
  assert elbo.shape == ()
  assert np.isfinite(elbo)
  assert elbo <= 0

def test_ebnbm_gamma_squarem(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  init = np.hstack([np.exp(-log_phi).ravel(), np.exp(-log_phi - log_mu).ravel(), 1e-3])
  log_mu_hat, log_phi_hat, log_theta_hat, alpha, beta, gamma, delta, elbo = scmodes.ebnbm.ebnbm_gamma(
    x, s, init=init, tol=1e-5, extrapolate=True)
  assert log_mu_hat.shape == log_mu.shape
  assert np.isfinite(log_mu_hat).all()
  assert log_phi_hat.shape == log_phi.shape
  assert np.isfinite(log_phi_hat).all()
  assert log_theta_hat.shape == ()
  assert np.isfinite(log_theta_hat)
  assert alpha.shape == x.shape
  assert np.isfinite(alpha).all()
  assert (alpha > 0).all()
  assert beta.shape == x.shape
  assert np.isfinite(beta).all()
  assert (beta > 0).all()
  assert gamma.shape == x.shape
  assert np.isfinite(gamma).all()
  assert (gamma > 0).all()
  assert delta.shape == x.shape
  assert np.isfinite(delta).all()
  assert (delta > 0).all()
  assert elbo.shape == ()
  assert np.isfinite(elbo)
  assert elbo <= 0

def test_ebnbm_gamma_squarem_nb_measurement(simulate_nb_gamma):
  x, s, log_mu, log_phi, theta = simulate_nb_gamma
  init = np.hstack([np.exp(-log_phi).ravel(), np.exp(-log_phi - log_mu).ravel(), theta])
  log_mu_hat, log_phi_hat, log_theta_hat, alpha, beta, gamma, delta, elbo = scmodes.ebnbm.ebnbm_gamma(
    x, s, init=init, max_iters=20_000, tol=1e-4, extrapolate=True)
  assert log_mu_hat.shape == log_mu.shape
  assert np.isfinite(log_mu_hat).all()
  assert log_phi_hat.shape == log_phi.shape
  assert np.isfinite(log_phi_hat).all()
  assert log_theta_hat.shape == ()
  assert np.isfinite(log_theta_hat)
  assert alpha.shape == x.shape
  assert np.isfinite(alpha).all()
  assert (alpha > 0).all()
  assert beta.shape == x.shape
  assert np.isfinite(beta).all()
  assert (beta > 0).all()
  assert gamma.shape == x.shape
  assert np.isfinite(gamma).all()
  assert (gamma > 0).all()
  assert delta.shape == x.shape
  assert np.isfinite(delta).all()
  assert (delta > 0).all()
  assert elbo.shape == ()
  assert np.isfinite(elbo)
  assert elbo <= 0

def test_ebnbm_gamma_squarem_default_init(simulate_gamma):
  x, s, log_mu, log_phi, _ = simulate_gamma
  log_mu_hat, log_phi_hat, log_theta_hat, alpha, beta, gamma, delta, elbo = scmodes.ebnbm.ebnbm_gamma(
    x, s, tol=10, extrapolate=True)
  assert log_mu_hat.shape == log_mu.shape
  assert np.isfinite(log_mu_hat).all()
  assert log_phi_hat.shape == log_phi.shape
  assert np.isfinite(log_phi_hat).all()
  assert log_theta_hat.shape == ()
  assert np.isfinite(log_theta_hat)
  assert alpha.shape == x.shape
  assert np.isfinite(alpha).all()
  assert (alpha > 0).all()
  assert beta.shape == x.shape
  assert np.isfinite(beta).all()
  assert (beta > 0).all()
  assert gamma.shape == x.shape
  assert np.isfinite(gamma).all()
  assert (gamma > 0).all()
  assert delta.shape == x.shape
  assert np.isfinite(delta).all()
  assert (delta > 0).all()
  assert elbo.shape == ()
  assert np.isfinite(elbo)
  assert elbo <= 0
