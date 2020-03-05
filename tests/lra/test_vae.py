import numpy as np
import torch
import pytest
import scipy.stats as st
import scmodes
import scmodes.lra.vae

@pytest.fixture
def simulate():
  np.random.seed(0)
  l = np.random.normal(size=(100, 3))
  f = np.random.normal(size=(3, 200))
  eta = l.dot(f)
  eta *= 5 / eta.max()
  x = np.random.poisson(lam=np.exp(eta))
  return x, eta

@pytest.fixture
def dims():
  # Data (n, p); latent representation (n, d)
  n = 50
  p = 1000
  d = 20
  n_samples = 10
  return n, p, d, n_samples

def test_encoder(dims):
  n, p, d, n_samples = dims
  enc = scmodes.lra.vae.Encoder(p, d)
  x = torch.tensor(np.random.normal(size=(n, p)), dtype=torch.float)
  mean, scale = enc.forward(x)
  assert mean.shape == (n, d)
  assert scale.shape == (n, d)

def test_decoder(dims):
  n, p, d, n_samples = dims
  dec = scmodes.lra.vae.Pois(d, p)
  x = torch.tensor(np.random.normal(size=(n, d)), dtype=torch.float)
  lam = dec.forward(x)
  assert lam.shape == (n, p)

def _fit_pvae(x, w=None, latent_dim=10, lr=1e-2, max_epochs=100, n_samples=10, verbose=False):
  n, p = x.shape
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.PVAE(p, latent_dim).fit(x, w=w, lr=lr, n_samples=n_samples, max_epochs=max_epochs, verbose=verbose)
  return model, x

def test_pvae(simulate):
  x, eta = simulate
  _fit_pvae(x, max_epochs=1)

def test_pvae_denoise(simulate):
  x, eta = simulate
  model, xt = _fit_pvae(x, max_epochs=1)
  lam = model.denoise(xt)
  assert lam.shape == x.shape
  assert np.isfinite(lam).all()
  assert (lam >= 0).all()

def test_pvae_predict(simulate):
  x, eta = simulate
  model, xt = _fit_pvae(x, max_epochs=1)
  lam = model.predict(xt)
  assert lam.shape == x.shape
  assert np.isfinite(lam).all()
  assert (lam >= 0).all()

@pytest.mark.skip('unfair comparison')
def test_pvae_oracle(simulate):
  x, eta = simulate
  l0 = st.poisson(mu=np.exp(eta)).logpmf(x).sum()
  model, xt = _fit_pvae(x, lr=1e-2, max_epochs=1000)
  lam = model.denoise(xt)
  l1 = st.poisson(mu=lam).logpmf(x).sum()
  assert l1 > l0

def test_pvae_test_set(simulate):
  x, eta = simulate
  n, p = x.shape
  latent_dim = 10
  xt = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.PVAE(p, latent_dim).fit(xt, test_size=0.1, lr=1e-3, n_samples=10, max_epochs=100, trace=True)
  t = np.array(model.trace)
  assert t.shape == (100, 2)

def test_wpvae(simulate):
  x, eta = simulate
  w = torch.ones(x.shape)
  _fit_pvae(x, w=w, max_epochs=1)

def test_wpvae_n_samples_1(simulate):
  x, eta = simulate
  w = torch.ones(x.shape)
  _fit_pvae(x, w=w, n_samples=1, max_epochs=10)

def test_wpvae_0_weights(simulate):
  x, eta = simulate
  w = (np.random.uniform(size=x.shape) < 0.9).astype(np.float32)
  w = torch.tensor(w)
  _fit_pvae(x, w=w, n_samples=1, max_epochs=10)
  
def test_nbvae_params():
  m0 = scmodes.lra.PVAE(100, 10)
  m1 = scmodes.lra.NBVAE(100, 10)
  assert len(list(m1.parameters())) == len(list(m0.parameters())) + 1

def test_nbvae(simulate):
  x, eta = simulate
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.NBVAE(input_dim=x.shape[1], latent_dim=10).fit(x, lr=1e-2, n_samples=10, max_epochs=1)

def test_nbvae_denoise(simulate):
  x, eta = simulate
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.NBVAE(input_dim=x.shape[1], latent_dim=10).fit(x, lr=1e-2, n_samples=10, max_epochs=1)
  lam = model.denoise(x)
  assert lam.shape == x.shape
  assert np.isfinite(lam).all()
  assert (lam >= 0).all()

def test_nbvae_predict(simulate):
  x, eta = simulate
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.NBVAE(input_dim=x.shape[1], latent_dim=10).fit(x, lr=1e-2, n_samples=10, max_epochs=1)
  mu = model.predict(x)
  assert mu.shape == x.shape
  assert np.isfinite(mu).all()
  assert (mu >= 0).all()

def test_nbvae_predict_samples(simulate):
  x, eta = simulate
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.NBVAE(input_dim=x.shape[1], latent_dim=10).fit(x, lr=1e-2, n_samples=10, max_epochs=1)
  mu = model.predict(x, n_samples=100)
  assert mu.shape == x.shape
  assert np.isfinite(mu).all()
  assert (mu >= 0).all()

def test_zinbvae_params():
  m0 = scmodes.lra.PVAE(100, 10)
  m1 = scmodes.lra.ZINBVAE(100, 10)
  assert len(list(m1.parameters())) == len(list(m0.parameters())) + 2

def test_zinbvae(simulate):
  x, eta = simulate
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.ZINBVAE(input_dim=x.shape[1], latent_dim=10).fit(x, lr=1e-2, n_samples=10, max_epochs=1)

def test_zinbvae_denoise(simulate):
  x, eta = simulate
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.ZINBVAE(input_dim=x.shape[1], latent_dim=10).fit(x, lr=1e-2, n_samples=10, max_epochs=1)
  lam = model.denoise(x)
  assert lam.shape == x.shape
  assert np.isfinite(lam).all()
  assert (lam >= 0).all()

def test_zinbvae_predict(simulate):
  x, eta = simulate
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.ZINBVAE(input_dim=x.shape[1], latent_dim=10).fit(x, lr=1e-2, n_samples=10, max_epochs=1)
  mu = model.predict(x)
  assert mu.shape == x.shape
  assert np.isfinite(mu).all()
  assert (mu >= 0).all()

def test_zinbvae_predict_samples(simulate):
  x, eta = simulate
  x = torch.tensor(x, dtype=torch.float)
  model = scmodes.lra.ZINBVAE(input_dim=x.shape[1], latent_dim=10).fit(x, lr=1e-2, n_samples=10, max_epochs=1)
  mu = model.predict(x, n_samples=100)
  assert mu.shape == x.shape
  assert np.isfinite(mu).all()
  assert (mu >= 0).all()
