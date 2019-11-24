"""Empirical Bayes Poisson Means via SGD

Empirical Bayes Poisson Means (EBPM) is the problem of estimating g, where

x_{ij} ~ Poisson(s_i \lambda_{ij})
\lambda_{ij} ~ g_j(.)

For g_j in the family of Gamma distributions, or point-Gamma distributions, the
marginal likelihood is analytic.

"""
import scipy.sparse as ss
import torch
import torch.util.data as td

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

def _zinb_llik(x, s, log_mean, log_inv_disp, logodds):
  """Return ln p(x_i | s_i, g)

  x_i ~ Poisson(s_i lambda_i)
  lambda_i ~ g = sigmoid(logodds) \delta_0(.) + sigmoid(-logodds) Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

  x - [n, p] tensor
  s - [n, 1] tensor
  log_mean - [1, p] tensor
  log_inv_disp - [1, p] tensor
  logodds - [1, p] tensor

  """
  nb_llik = _nb_llik(x, s, log_mean, log_inv_disp)
  case_zero = -torch.nn.softplus(-logodds) + torch.nn.softplus(nb_llik - logodds)
  case_non_zero = -torch.nn.softplus(logodds) + nb_llik
  return torch.where(torch.lt(x, 1), case_zero, case_non_zero)

def _check_args(x, s, init, lr, batch_size, max_epochs):
  n, p = x.shape
  if is not None and s.shape != (n, 1):
    raise ArgumentError(f'shape mismatch (s): expected {(n, 1)}, got {s.shape}')
  if init is None:
    pass
  elif init[0].shape != (1, p):
    raise ArgumentError(f'shape mismatch (log_mu): expected {(1, p)}, got {init[0].shape}')
  elif init[1].shape != (1, p):
    raise ArgumentError(f'shape mismatch (log_phi): expected {(1, p)}, got {init[0].shape}')
  if lr <= 0:
    raise ArgumentError('lr must be >= 0')
  if batch_size < 1:
    raise ArgumentError('batch_size must be >= 1')
  if max_epochs < 1:
    raise ArgumentError('max_epochs must be >= 1')

def _sgd(x, s, llik, params, lr, max_epochs, verbose, trace):
  data = td.DataLoader(td.TensorDataset(x, s, batch_size=batch_size, pin_memory=True))
  if torch.cuda.is_available():
    for p in params:
      p.cuda()
  opt = torch.optim.RMSprop(params, lr=lr)
  trace = []
  for epoch in range(max_epochs):
    for (x, s) in data:
      opt.zero_grad()
      if torch.cuda.is_available():
        x.cuda()
        s.cuda()
      loss = -llik(x, s, log_mean, log_inv_disp).sum()
      loss.backward()
      opt.step()
      if trace:
        trace.append([p.item() for p in params] + [loss.item()])
    if verbose:
      print(f'Epoch {epoch}:', loss.item())
  return [p.item() for p in params] + [loss.item()]

def ebpm_gamma(x, s=None, init=None, lr=1e-2, batch_size=100, max_epochs=100, verbose=False, trace=False):
  """Return parameters of fitted Gamma distribution g

  x - array-like [n, p]
  s - array-like [n, 1]
  init - (log_mu, log_phi) [1, p]

  """
  _check_args(x, s, init, lr, batch_size, max_epochs)
  n, p = x.shape
  if s is None:
    s = torch.tensor(x.sum(axis=1), dtype=torch.float)
  else:
    s = torch.tensor(s, dtype=torch.float)
  if ss.issparse(x):
    raise NotImplementedError('sparse x not supported')
  else:
    # Important: this must come after size factor computation
    x = torch.tensor(x, dtype=torch.float)
  if init is None:
    log_mean = torch.zeros([1, p], dtype=torch.float, requires_grad=True)
    log_inv_disp = torch.zeros([1, p], dtype=torch.float, requires_grad=True)
  else:
    log_mean = torch.tensor(init[0], dtype=torch.float, requires_grad=True)
    log_inv_disp = torch.tensor(init[1], dtype=torch.float, requires_grad=True)
  return _sgd(x, s, llik=_nb_llik, params=[log_mean, log_inv_disp], lr=lr,
              batch_size=batch_size, verbose=verbose, trace=trace)

def ebpm_point_gamma(x, s=None, init=None, lr=1e-2, batch_size=100, max_epochs=100, verbose=False, trace=False):
  """Return parameters of the fitted point-Gamma distribution g

  x - array-like [n, p]
  s - array-like [n, 1]
  init - (log_mu, log_phi) [1, p]

  """
  _check_args(x, s, init, lr, batch_size, max_epochs)
  n, p = x.shape
  if s is None:
    s = torch.tensor(x.sum(axis=1), dtype=torch.float)
  else:
    s = torch.tensor(s, dtype=torch.float)
  if ss.issparse(x):
    raise NotImplementedError('sparse x not supported')
  else:
    # Important: this must come after size factor computation
    x = torch.tensor(x, dtype=torch.float)
  if init is None:
    if verbose:
      print('Fitting ebpm_gamma to get initialization')
    init, _ = ebpm_gamma(x, s, lr=lr, batch_size=batch_size, max_epochs=max_epochs, verbose=verbose)
  log_mean = torch.tensor(init[0], dtype=torch.float, requires_grad=True)
  log_inv_disp = torch.tensor(init[1], dtype=torch.float, requires_grad=True)
  # Important: start pi_j near zero (on the logit scale)
  logodds = torch.full([1, p], -8, dtype=torch.float, requires_grad=True)
  return _sgd(x, s, llik=_zinb_llik, params=[log_mean, log_inv_disp, logodds],
              lr=lr, batch_size=batch_size, verbose=verbose, trace=trace)
