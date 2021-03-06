import anndata
import numpy as np
import pandas as pd
import scipy.io
import scipy.special as sp

def simulate_pois(n, p, rank, eta_max=None, holdout=None, seed=0):
  np.random.seed(seed)
  l = np.random.normal(size=(n, rank))
  f = np.random.normal(size=(rank, p))
  eta = l.dot(f)
  if eta_max is not None:
    # Scale the maximum value
    eta *= eta_max / eta.max()
  x = np.random.poisson(lam=np.exp(eta))
  if holdout is not None:
    mask = np.random.uniform(size=(n, p)) < holdout
    x = np.ma.masked_array(x, mask=mask)
  return x, eta

def simulate_pois_size(n, p, rank, s, seed=0):
  np.random.seed(seed)
  l = np.random.normal(size=(n, rank))
  f = np.random.normal(size=(rank, p))
  eta = l.dot(f)
  eta -= sp.logsumexp(eta, axis=0)
  mu = np.exp(eta)
  x = np.random.poisson(lam=s * mu)
  return x, mu

def read_10x(prefix, min_detect=0.25, return_adata=False, return_df=False):
  if return_adata and return_df:
    raise ValueError('only one of return_adata and return_df must be True')
  counts = scipy.io.mmread(f'{prefix}/matrix.mtx.gz').tocsr()
  samples = pd.read_csv(f'{prefix}/barcodes.tsv.gz', sep='\t', header=None)
  genes = pd.read_csv(f'{prefix}/genes.tsv.gz', sep='\t', header=None)
  # Important: counts is genes x samples
  if min_detect >= 0:
    keep_genes = ((counts > 0).mean(axis=1) >= min_detect).A.ravel()
    counts = counts[keep_genes]
    genes = genes.loc[keep_genes]
  else:
    raise ValueError('min_detect must be >= 0')
  if return_adata:
    return anndata.AnnData(counts.T.tocsr(), obs=samples, var=genes)
  elif return_df:
    return pd.DataFrame(counts.A.T, index=samples[0].values, columns=genes[0].values)
  else:
    return counts.T

def ipsc(prefix, return_df=False, query=None, p=None, seed=0):
  annotations = pd.read_csv(f'{prefix}/scqtl-annotation.txt', sep='\t')
  keep_samples = pd.read_csv(f'{prefix}/quality-single-cells.txt', sep='\t', index_col=0, header=None)
  keep_genes = pd.read_csv(f'{prefix}/genes-pass-filter.txt', sep='\t', index_col=0, header=None)
  if query is not None:
    keep_genes = keep_genes[keep_genes.index.isin(query)]
  if p is not None:
    keep_genes = keep_genes[keep_genes.values].sample(n=p, random_state=seed)
  annotations = annotations.loc[keep_samples.values.ravel()]
  result = []
  for chunk in pd.read_csv(f'{prefix}/scqtl-counts.txt.gz', sep='\t', index_col=0, chunksize=100):
     x = (chunk
          .loc[:,keep_samples.values.ravel()]
          .filter(items=keep_genes[keep_genes.values.ravel()].index, axis='index'))
     if not x.empty:
       result.append(x)
  result = pd.concat(result)
  # Return samples x genes
  if return_df:
    return result.T
  else:
    return result.values.T

def synthetic_mix(x1, x2, min_detect=None):
  x = pd.concat([x1, x2], axis='index', join='inner')
  y = np.zeros(x.shape[0]).astype(int)
  y[:x1.shape[0]] = 1
  if min_detect is not None:
    keep_genes = (x > 0).mean(axis=0) >= min_detect
    x = x.loc[:,keep_genes]
  return x, y

def cortex(path, return_df=False):
  counts = pd.read_csv(path, index_col=0, sep='\t')
  # Follow scVI here
  subset = counts.loc[counts.var(axis=1).sort_values(ascending=False).head(n=500).index].T
  if return_df:
    return subset
  else:
    return subset.values
