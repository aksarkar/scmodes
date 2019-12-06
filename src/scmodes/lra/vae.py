"""Variational autoencoder models for count data, supporting missing values

Under the VAE (Kingma and Welling 2014; Rezende and Mohamed 2014; Lopez et
al. 2018),

p(x_ij | λ_ij) ~ Pois(λ_ij)

In the simplest case, λ_ij = (λ(z_i))_j, where λ(⋅) is a neural
network R^k -> R^p. We also consider

λ_ij = (μ_ij(z_i))_j u_ij
u_ij ~ g(⋅)

where μ(⋅) is the neural network, and g is a (point-)Gamma distribution. The goal is
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
  """Decoder p(x | z) ~ Poisson(λ(z))"""
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
    return self.lam(x) + 1e-15

def kl_term(mean, scale):
  """KL divergence between N(mean, scale) and N(0, 1)"""
  return .5 * (1 - 2 * torch.log(scale) + (mean * mean + scale * scale))

def pois_llik(x, mean):
  """Return the log likelihood of x distributed as Poisson"""
  return x * torch.log(mean) - mean - torch.lgamma(x + 1)

def nb_llik(x, mean, inv_disp):
  """Return the log likelihood of x distributed as NB

  See Hilbe 2012, eq. 8.10

  mean - mean (> 0)
  inv_disp - inverse dispersion (> 0)

  """
  return (x * torch.log(mean / inv_disp) -
          x * torch.log(1 + mean / inv_disp) -
          inv_disp * torch.log(1 + mean / inv_disp) +
          torch.lgamma(x + inv_disp) -
          torch.lgamma(inv_disp) -
          torch.lgamma(x + 1))

def zinb_llik(x, mean, inv_disp, logodds):
  """Return the log likelihood of x distributed as ZINB

  See Hilbe 2012, eq. 11.12, 11.13

  mean - mean (> 0)
  inv_disp - inverse dispersion (> 0)
  logodds - logit point mass weight

  """
  # Important identities:
  # log(x + y) = log(x) + softplus(log(y) - log(x))
  # log(sigmoid(x)) = -softplus(-x)
  softplus = torch.nn.functional.softplus
  case_zero = -softplus(-logodds) + softplus(nb_llik(x, mean, inv_disp) - logodds)
  case_non_zero = -softplus(logodds) + nb_llik(x, mean, inv_disp)
  return torch.where(torch.lt(x, 1), case_zero, case_non_zero)

class PVAE(torch.nn.Module):
  def __init__(self, input_dim, latent_dim):
    super().__init__()
    self.encoder = Encoder(input_dim, latent_dim)
    self.decoder = Pois(latent_dim, input_dim)

  def loss(self, x, w, n_samples):
    # Important: if w ∈ {0, 1}, then we need to mask entries going into the
    # decoder
    mean, scale = self.encoder.forward(w * x)
    # [batch_size]
    # Important: this is analytic
    kl = torch.sum(kl_term(mean, scale), dim=1)
    # [n_samples, batch_size, latent_dim]
    qz = torch.distributions.Normal(mean, scale).rsample(n_samples)
    # [n_samples, batch_size, input_dim]
    lam = self.decoder.forward(qz)
    error = torch.mean(torch.sum(torch.reshape(w, [1, w.shape[0], w.shape[1]]) * pois_llik(x, lam), dim=2), dim=0)
    # Important: optim minimizes
    loss = -torch.sum(error - kl)
    return loss

  def fit(self, x, max_epochs, w=None, verbose=False, n_samples=10, **kwargs):
    """Fit the model

    :param x: torch.tensor [n_cells, n_genes]
    :param w: torch.tensor [n_cells, n_genes]

    """
    if w is None:
      w = torch.tensor([[1]], dtype=torch.float)
    if torch.cuda.is_available():
      # Move the model and data to the GPU
      self.cuda()
      x = x.cuda()
      w = w.cuda()
    n_samples = torch.Size([n_samples])
    opt = torch.optim.RMSprop(self.parameters(), **kwargs)
    for epoch in range(max_epochs):
      opt.zero_grad()
      loss = self.loss(x, w=w, n_samples=n_samples)
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

class NBVAE(PVAE):
  def __init__(self, input_dim, latent_dim, disp_by_gene=False):
    """Initialize the VAE parameters

    disp_by_gene - if True, model one dispersion parameter per gene

    """
    # Important: only μ is a neural network output, so we can actually reuse
    # the entire PVAE.
    super().__init__(input_dim, latent_dim)
    if disp_by_gene:
      # Important: shape needs to be correct to broadcast
      self.log_inv_disp = torch.nn.Parameter(torch.zeros([1, input_dim]))
    else:
      self.log_inv_disp = torch.nn.Parameter(torch.zeros([1]))

  def loss(self, x, w, n_samples):
    mean, scale = self.encoder.forward(w * x)
    kl = torch.sum(kl_term(mean, scale), dim=1)
    qz = torch.distributions.Normal(mean, scale).rsample(n_samples)
    mu = self.decoder.forward(qz)
    error = torch.mean(torch.sum(torch.reshape(w, [1, w.shape[0], w.shape[1]]) * nb_llik(x, mu, torch.exp(self.log_inv_disp)), dim=2), dim=0)
    loss = -torch.sum(error - kl)
    return loss
    
  @torch.no_grad()
  def denoise(self, x):
    if torch.cuda.is_available():
      x = x.cuda()
    # Plug E[z | x] into the decoder
    mu = self.decoder.forward(self.encoder.forward(x)[0])
    # Expected posterior mean
    lam = (x + torch.exp(self.log_inv_disp)) / (mu + torch.exp(self.log_inv_disp))
    if torch.cuda.is_available():
      lam = lam.cpu()
    return lam.numpy()

class ZINBVAE(NBVAE):
  def __init__(self, input_dim, latent_dim, disp_by_gene=False, logodds_by_gene=False):
    # Important: only μ is a neural network output, and we still need
    # log_inv_disp, so we can actually reuse the entire NBVAE.
    super().__init__(input_dim, latent_dim, disp_by_gene=disp_by_gene)
    if logodds_by_gene:
      self.logodds = torch.nn.Parameter(torch.zeros([1, input_dim]))
    else:
      self.logodds = torch.nn.Parameter(torch.zeros([1]))

  def loss(self, x, w, n_samples):
    mean, scale = self.encoder.forward(w * x)
    kl = torch.sum(kl_term(mean, scale), dim=1)
    qz = torch.distributions.Normal(mean, scale).rsample(n_samples)
    mu = self.decoder.forward(qz)
    error = torch.mean(torch.sum(torch.reshape(w, [1, w.shape[0], w.shape[1]]) * zinb_llik(x, mu, torch.exp(self.log_inv_disp), self.logodds), dim=2), dim=0)
    loss = -torch.sum(error - kl)
    return loss
    
  @torch.no_grad()
  def denoise(self, x):
    if torch.cuda.is_available():
      x = x.cuda()
    # Plug E[z | x] into the decoder
    mu = self.decoder.forward(self.encoder.forward(x)[0])
    # Expected posterior mean
    lam = torch.sigmoid(-self.logodds) * (x + torch.exp(self.log_inv_disp)) / (mu + torch.exp(self.log_inv_disp))
    if torch.cuda.is_available():
      lam = lam.cpu()
    return lam.numpy()
