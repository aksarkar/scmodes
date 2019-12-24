"""Empirical Bayes Poisson Means via SGD

These implementations are specialized for fitting p EBPM problems on n samples
in parallel, where n, p may be large.

"""
import scipy.sparse as ss
import torch
import torch.utils.data as td

class EBPMDataset(td.Dataset):
  """Specialized dataset for sparse count matrix and scale factors"""
  def __init__(self, x, s):
    super().__init__()
    if ss.issparse(x) and not ss.isspmatrix_csr(x):
      x = x.tocsr()
    else:
      x = ss.csr_matrix(x)
    self.n, self.p = x.shape
    if torch.cuda.is_available():
      device = 'cuda'
    else:
      device = 'cpu'
    self.data = torch.tensor(x.data, dtype=torch.float, device=device)
    self.indices = torch.tensor(x.indices, dtype=torch.long, device=device)
    self.indptr = torch.tensor(x.indptr, dtype=torch.long, device=device)
    self.s = torch.tensor(s, dtype=torch.float, device=device)

  def __getitem__(self, index):
    start = self.indptr[index]
    end = self.indptr[index + 1]
    return (
      torch.sparse.FloatTensor(
        # Important: sparse indices are long in Torch
        torch.stack([torch.zeros(end - start, dtype=torch.long, device=self.indices.device), self.indices[start:end]]),
        # Important: this needs to be 1d before collate_fn
        self.data[start:end], size=[1, self.p]).to_dense().squeeze(),
      self.s[index]
    )
    
  def __len__(self):
    return self.n

  def collate_fn(self, indices):
    """Return a minibatch of items"""
    # Important: this is *much* faster than [data[i] for i in indices]
    return (
      torch.sparse.FloatTensor(
        torch.cat([
          torch.stack([torch.full(((self.indptr[i + 1] - self.indptr[i]).item(),), i - indices[0],
                                  dtype=torch.long, device=self.indices.device),
                       self.indices[self.indptr[i]:self.indptr[i + 1]]]) for i in indices], dim=1),
        torch.cat([self.data[self.indptr[i]:self.indptr[i + 1]] for i in indices]),
        size=[len(indices), self.p]).to_dense().squeeze(),
      self.s[indices]
    )

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
  lambda_i ~ g = sigmoid(logodds) \\delta_0(.) + sigmoid(-logodds) Gamma(exp(log_inv_disp), exp(log_mean - log_inv_disp))

  x - [n, p] tensor
  s - [n, 1] tensor
  log_mean - [1, p] tensor
  log_inv_disp - [1, p] tensor
  logodds - [1, p] tensor

  """
  nb_llik = _nb_llik(x, s, log_mean, log_inv_disp)
  softplus = torch.nn.functional.softplus
  case_zero = -softplus(-logodds) + softplus(nb_llik - logodds)
  case_non_zero = -softplus(logodds) + nb_llik
  return torch.where(torch.lt(x, 1), case_zero, case_non_zero)

def _check_args(x, s, init, lr, batch_size, max_epochs):
  """Check x, s and return a DataLoader"""
  n, p = x.shape
  if s is None:
    s = x.sum(axis=1)
  elif  s.shape != (n, 1):
    raise ValueError(f'shape mismatch (s): expected {(n, 1)}, got {s.shape}')
  elif not isinstance(s, torch.FloatTensor):
    s = torch.tensor(s, dtype=torch.float)
  else:
    s = s
  if ss.issparse(x):
    data = EBPMDataset(x, s)
  elif not isinstance(x, torch.Tensor):
    x = torch.tensor(x, dtype=torch.float)
    if torch.cuda.is_available():
      x = x.cuda()
      s = s.cuda()
    data = td.TensorDataset(x, s)
  if init is None:
    pass
  elif init[0].shape != (1, p):
    raise ValueError(f'shape mismatch (log_mu): expected {(1, p)}, got {init[0].shape}')
  elif init[1].shape != (1, p):
    raise ValueError(f'shape mismatch (log_phi): expected {(1, p)}, got {init[1].shape}')
  elif len(init) > 2:
    raise ValueError('expected two values in init, got {len(init)}')
  if lr <= 0:
    raise ValueError('lr must be >= 0')
  if batch_size < 1:
    raise ValueError('batch_size must be >= 1')
  if max_epochs < 1:
    raise ValueError('max_epochs must be >= 1')
  return data, n, p

def _sgd(data, llik, params, lr=1e-2, batch_size=100, max_epochs=100, num_workers=0, verbose=False, trace=False):
  """SGD subroutine

  x - [n, p] tensor
  s - [n, 1] tensor
  llik - function returning [n, p] tensor
  params - list of tensor [1, p]

  """
  data = td.DataLoader(data, batch_size=batch_size, num_workers=num_workers,
                       # Important: only CPU memory can be pinned
                       pin_memory=not torch.cuda.is_available())
  opt = torch.optim.RMSprop(params, lr=lr)
  param_trace = []
  loss = None
  for epoch in range(max_epochs):
    for (x, s) in data:
      opt.zero_grad()
      # Important: params are assumed to be provided in the order assumed by llik
      loss = -llik(x, s, *params).sum()
      if torch.isnan(loss):
        raise RuntimeError('nan loss')
      loss.backward()
      opt.step()
      if trace:
        # Important: this only works for scalar params
        param_trace.append([p.item() for p in params] + [loss.item()])
    if verbose:
      print(f'Epoch {epoch}:', loss.item())
  if torch.cuda.is_available:
    result = [p.cpu().detach().numpy() for p in params]
  else:
    result = [p.detach().numpy() for p in params]
  # Clean up GPU memory
  del params[:]
  result.append(loss.item())
  if trace:
    result.append(param_trace)
  return result

def ebpm_gamma(x, s=None, init=None, lr=1e-2, batch_size=100, max_epochs=100, verbose=False, trace=False):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
distribution

  Important: returns log mu and -log phi

  x - array-like [n, p]
  s - array-like [n, 1]
  init - (log_mu, log_phi) [1, p]

  """
  data, n, p = _check_args(x, s, init, lr, batch_size, max_epochs)
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  if init is None:
    log_mean = torch.zeros([1, p], dtype=torch.float, requires_grad=True, device=device)
    log_inv_disp = torch.zeros([1, p], dtype=torch.float, requires_grad=True, device=device)
  else:
    log_mean = torch.tensor(init[0], dtype=torch.float, requires_grad=True, device=device)
    log_inv_disp = torch.tensor(init[1], dtype=torch.float, requires_grad=True, device=device)
  return _sgd(data, llik=_nb_llik, params=[log_mean, log_inv_disp], lr=lr,
              batch_size=batch_size, max_epochs=max_epochs, verbose=verbose,
              trace=trace)

def ebpm_point_gamma(x, s=None, init=None, lr=1e-2, batch_size=100, max_epochs=100, verbose=False, trace=False):
  """Return fitted parameters and marginal log likelihood assuming g is a Gamma
distribution

  Important: returns log mu, -log phi, logit pi

  x - array-like [n, p]
  s - array-like [n, 1]
  init - (log_mu, -log_phi) [1, p]

  """
  data, n, p = _check_args(x, s, init, lr, batch_size, max_epochs)
  if init is None:
    if verbose:
      print('Fitting ebpm_gamma to get initialization')
    res = ebpm_gamma(x, s, lr=lr, batch_size=batch_size, max_epochs=max_epochs, verbose=verbose)
    init = res[:-1]
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  log_mean = torch.tensor(init[0], dtype=torch.float, requires_grad=True, device=device)
  log_inv_disp = torch.tensor(init[1], dtype=torch.float, requires_grad=True, device=device)
  # Important: start pi_j near zero (on the logit scale)
  logodds = torch.full([1, p], -8, dtype=torch.float, requires_grad=True, device=device)
  return _sgd(data, llik=_zinb_llik, params=[log_mean, log_inv_disp, logodds],
              lr=lr, batch_size=batch_size, max_epochs=max_epochs,
              verbose=verbose, trace=trace)
