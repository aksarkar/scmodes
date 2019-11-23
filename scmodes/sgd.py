"""Poisson--(point-)Gamma model of scRNA-seq data at a single gene

Under this model, the marginal distribution of x_ij is (ZI)NB. We estimate
(pi_j, mu_j, phi_j) by maximizing the log likelihood using accelerated SGD.

"""
import torch

def _nb_llik(x, s, log_mean, log_inv_disp):
  """Return ln p(x_i | s_i, g)

  x_i ~ Poisson(s_i lambda_i)
  lambda_i ~ g = Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

  x - [n, p] tensor
  s - [n, 1] tensor
  log_mean - [1, p] tensor
  log_inv_disp - [1, p] tensor

  """
  mean = torch.matmul(s, torch.exp(log_mean))
  inv_disp = torch.exp(log_inv_disp)
  return (x * torch.log(mean / inv_disp)
          - x * torch.log(1 + mean / inv_disp)
          - inv_disp * torch.log(1 + mean / inv_disp)
          # Important: these terms are why we use inverse dispersion
          + torch.lgamma(x + inv_disp)
          - torch.lgamma(inv_disp)
          - torch.lgamma(x + 1))

class PoissonGamma():
  def __init__(self, p):
    self.log_mean = torch.zeros([1, p], dtype=torch.float, requires_grad=True)
    self.log_inv_disp = torch.zeros([1, p], dtype=torch.float, requires_grad=True)
    self.trace = []

  def fit(self, data, max_epochs=10, verbose=False, trace=False, **kwargs):
    """Fit the model.

    data - torch.utils.data.DataLoader
    kwargs - arguments to torch.optim.RMSprop

    """
    if torch.cuda.is_available():
      # Move the model to the GPU
      self.cuda()
    opt = torch.optim.RMSprop([self.log_mean, self.log_inv_disp], **kwargs)
    for epoch in range(max_epochs):
      for (x, s) in data:
        opt.zero_grad()
        if torch.cuda.is_available():
          # Move the data to the GPU
          x.cuda()
          s.cuda()
        loss = -_nb_llik(x, s, self.log_mean, self.log_inv_disp).sum()
        loss.backward()
        opt.step()
        if trace:
          self.trace.append([self.log_mean.item(), self.log_inv_disp.item(), loss.item()])
      if verbose:
        print(f'Epoch {epoch}:', loss.item())

  @torch.no_grad()
  def opt(self):
    """Return ln mu_j, ln phi_j"""
    return self.log_mean.detach().numpy(), -self.log_inv_disp.detach().numpy()

class PoissonPointGamma(PoissonGamma):
  def __init__(self, p):
    super().__init__(p)
    self.logodds = torch.zeros([p, 1])

  def forward(self, x, s):
    """Return sum of ln p(x_i | s_i g)

    x_i ~ Poisson(s_i lambda_i)
    lambda_i ~ g = logit(logodds) delta_0() + logit(-logodds) Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

    """
    nb_llik = _nb_llik(x, s, self.log_mean, self.log_inv_disp)
    case_zero = -torch.nn.softplus(-logodds) + torch.nn.softplus(nb_llik - logodds)
    case_non_zero = -torch.nn.softplus(logodds) + nb_llik
    return torch.where(torch.lt(x, 1), case_zero, case_non_zero).sum()
