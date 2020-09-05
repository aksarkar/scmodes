"""Generic routines for EM, SQUAREM, DAAREM"""

import numpy as np

def em(init, objective_fn, update_fn, max_iters, tol, *args, **kwargs):
  theta = init.copy()
  obj = objective_fn(theta, *args, **kwargs)
  for i in range(max_iters):
    theta = update_fn(theta, *args, **kwargs)
    update = objective_fn(theta, *args, **kwargs)
    diff = update - obj
    if i == 0 and diff < 0:
      # Hack: this is needed for numerical reasons, because in e.g.,
      # ebpm_gamma, a point mass is the limit as b → ∞
      return init, obj
    elif not np.isfinite(update):
      raise RuntimeError('Non-finite objective')
    elif diff < 0:
      raise RuntimeError('llik decreased')
    elif diff < tol:
      return theta, update
    else:
      obj = update
  else:
    raise RuntimeError(f'failed to converge in max_iters ({diff:.4g} > {tol:.4g})')

def squarem(init, objective_fn, update_fn, max_iters, tol, par_tol=1e-8, max_step_updates=10, *args, **kwargs):
  """Squared extrapolation scheme for accelerated EM

  Reference: 

    Varadhan, R. and Roland, C. (2008), Simple and Globally Convergent Methods
    for Accelerating the Convergence of Any EM Algorithm. Scandinavian Journal
    of Statistics, 35: 335-353. doi:10.1111/j.1467-9469.2007.00585.x

  """
  theta = init
  obj = objective_fn(theta, *args, **kwargs)
  for i in range(max_iters):
    x1 = update_fn(theta, *args, **kwargs)
    r = x1 - theta
    if i == 0 and objective_fn(x1, *args, **kwargs) < obj:
      # Hack: this is needed for numerical reasons, because in e.g.,
      # ebpm_gamma, a point mass is the limit as a = 1/φ → ∞
      return init, obj
    x2 = update_fn(x1, *args, **kwargs)
    v = (x2 - x1) - r
    if np.linalg.norm(v) < par_tol:
      return x2, objective_fn(x2, *args, **kwargs)
    step = -np.sqrt(r @ r) / np.sqrt(v @ v)
    if step > -1:
      step = -1
      theta += -2 * step * r + step * step * v
      update = objective_fn(theta, *args, **kwargs)
      diff = update - obj
    else:
      # Step length = -1 is EM; use as large a step length as is feasible to
      # maintain monotonicity
      for j in range(max_step_updates):
        candidate = theta - 2 * step * r + step * step * v
        update = objective_fn(candidate, *args, **kwargs)
        diff = update - obj
        if np.isfinite(update) and diff > 0:
          theta = candidate
          break
        else:
          step = (step - 1) / 2
      else:
        step = -1
        theta += -2 * step * r + step * step * v
        update = objective_fn(theta, *args, **kwargs)
        diff = update - obj
    if diff < tol:
      return theta, update
    else:
      obj = update
  else:
    raise RuntimeError(f'failed to converge in max_iters ({diff:.3g} > {tol:.3g})')
