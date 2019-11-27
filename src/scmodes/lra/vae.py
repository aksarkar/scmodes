"""Variational autoencoder models for count data

Under the VAE (Kingma and Welling 2014; Rezende and Mohamed 2014; Lopez et
al. 2018),

p(x_ij | λ_ij) ~ Pois(λ_ij)

In the simplest case, λ_ij = (λ(z_i))_j, where λ(⋅) is a neural
network R^k -> R^p. We also consider

λ_ij = (μ_ij(z_i))_j u_ij
u_ij ~ g(⋅)

where μ(⋅) is the neural network, and g is a Gamma distribution. The goal is
to learn p(z_i | x_i), which is achieved by variational inference assuming

q(z_i | x_i) = N(m(x_i), diag(S(x_i)))

where m, S are neural network outputs R^p -> R^k. To optimize the ELBO, we use
the reparameterization gradient. Crucially, these implementations support
incomplete data associated with weights w_ij ∈ {0, 1} (Nazabal et al. 2018;
Mattei and Frellsen 2018).

"""
import torch

class Encoder(torch.nn.Module):
  """Encoder q(z | x) = N(m(x), diag(S(x)))"""
  def __init__(self, input_dim, output_dim, hidden_dim=128):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.BatchNorm1d(hidden_dim),
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.BatchNorm1d(hidden_dim),
    )
    self.mean = torch.nn.Linear(hidden_dim, output_dim)
    self.scale = torch.nn.Sequential(torch.nn.Linear(hidden_dim, output_dim), torch.nn.Softplus())

  def forward(self, x):
    q = self.net(x)
    return self.mean(q), self.scale(q)

class Pois(torch.nn.Module):
  """Decoder p(x | z) ~ Poisson(s_i λ(z))"""
  def __init__(self, input_dim, output_dim, hidden_dim=128):
    super().__init__()
    self.lam = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, output_dim),
      torch.nn.Softplus(),
    )

  def forward(self, x):
    return self.lam(x) + 1e-8

def kl_term(mean, scale):
  """KL divergence between N(mean, scale) and N(0, 1)"""
  return .5 * (1 - 2 * torch.log(scale) + (mean * mean + scale * scale))

def pois_llik(x, mean):
  """Log likelihood of x distributed as Poisson"""
  return x * torch.log(mean) - mean - torch.lgamma(x + 1)

class PVAE(torch.nn.Module):
  def __init__(self, input_dim, latent_dim):
    super().__init__()
    self.encoder = Encoder(input_dim, latent_dim)
    self.decoder = Pois(latent_dim, input_dim)

  def loss(self, x, s, stoch_samples):
    mean, scale = self.encoder.forward(x)
    # [batch_size]
    # Important: this is analytic
    kl = torch.sum(kl_term(mean, scale), dim=1)
    # [stoch_samples, batch_size, latent_dim]
    qz = torch.distributions.Normal(mean, scale).rsample(stoch_samples)
    # [stoch_samples, batch_size, input_dim]
    lam = self.decoder.forward(qz)
    error = torch.mean(torch.sum(pois_llik(x, lam), dim=2), dim=0)
    # Important: optim minimizes
    loss = -torch.sum(error - kl)
    return loss

  def fit(self, x, s, max_epochs, verbose=False, stoch_samples=10, **kwargs):
    """Fit the model

    :param x: torch.tensor [n_cells, n_genes]
    :param s: torch.tensor [n_cells, 1]

    """
    if torch.cuda.is_available():
      # Move the model and data to the GPU
      self.cuda()
      x = x.cuda()
      s = s.cuda()
    stoch_samples = torch.Size([stoch_samples])
    opt = torch.optim.RMSprop(self.parameters(), **kwargs)
    for epoch in range(max_epochs):
      opt.zero_grad()
      loss = self.loss(x, s, stoch_samples)
      if torch.isnan(loss):
        raise RuntimeError('nan loss')
      loss.backward()
      opt.step()
      if verbose and not epoch % 10:
        print(f'[epoch={epoch}] elbo={-loss}')
    return self

  @torch.no_grad()
  def denoise(self, x):
    if torch.cuda.is_available():
      x = x.cuda()
    # Plug E[z | x] into the decoder
    lam = self.decoder.forward(self.encoder.forward(x)[0])
    if torch.cuda.is_available():
      lam = lam.cpu()
    return lam.numpy()
