import numpy as np
import pandas as pd
import scipy.io

def read_10x(prefix, min_detect=0.25, return_df=False):
  counts = scipy.io.mmread(f'{prefix}/matrix.mtx.gz').tocsr()
  keep = ((counts > 0).mean(axis=1) >= min_detect).A.ravel()
  counts = counts[keep].T.A.astype(np.int)
  if return_df:
    genes = pd.read_csv(f'{prefix}/genes.tsv.gz', sep='\t', header=None)
    return pd.DataFrame(counts, columns=genes.loc[keep, 0])
  else:
    return counts

def cd8_cytotoxic_t_cells(prefix, **kwargs):
  return read_10x('/project2/mstephens/aksarkar/projects/singlecell-ideas/data/10xgenomics/cytotoxic_t/filtered_matrices_mex/hg19', **kwargs)

def cd19_b_cells(**kwargs):
  return read_10x('/project2/mstephens/aksarkar/projects/singlecell-ideas/data/10xgenomics/b_cells/filtered_matrices_mex/hg19/', **kwargs)

def ipsc(prefix='/project2/mstephens/aksarkar/projects/singlecell-qtl/data', return_df=False, query=None, n=None, seed=0):
  annotations = pd.read_csv(f'{prefix}/scqtl-annotation.txt', sep='\t')
  keep_samples = pd.read_csv(f'{prefix}/quality-single-cells.txt', sep='\t', index_col=0, header=None)
  keep_genes = pd.read_csv(f'{prefix}/genes-pass-filter.txt', sep='\t', index_col=0, header=None)
  if query is not None:
    keep_genes = keep_genes[keep_genes.index.isin(query)]
  if n is not None:
    keep_genes = keep_genes.sample(n=n, random_state=seed)
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

def cd8_cd19_mix():
  cd8 = cd8_cytotoxic_t_cells(return_df=True)
  cd19 = cd19_b_cells(return_df=True)
  x = pd.concat([cd8, cd19], axis='index', join='inner')
  y = np.zeros(x.shape[0]).astype(int)
  y[:cd8.shape[0]] = 1
  return x, y

def cyto_naive_t_mix():
  cyto = read_10x(prefix='/project2/mstephens/aksarkar/projects/singlecell-ideas/data/10xgenomics/cytotoxic_t/filtered_matrices_mex/hg19', return_df=True)
  naive = read_10x(prefix='/project2/mstephens/aksarkar/projects/singlecell-ideas/data/10xgenomics/naive_cytotoxic/filtered_matrices_mex/hg19', return_df=True)
  x = pd.concat([cyto, naive], axis='index', join='inner')
  y = np.zeros(x.shape[0]).astype(int)
  y[:cyto.shape[0]] = 1
  return x, y

def pbmcs_68k(**kwargs):
  return read_10x('/project2/mstephens/aksarkar/projects/singlecell-ideas/data/10xgenomics/fresh_68k_pbmc_donor_a/filtered_matrices_mex/hg19', **kwargs)

def cortex(return_df=False):
  counts = pd.read_csv('/project2/mstephens/aksarkar/projects/singlecell-ideas/data/zeisel-2015/GSE60361_C1-3005-Expression.txt.gz', index_col=0, sep='\t')
  # Follow scVI here
  subset = counts.loc[counts.var(axis=1).sort_values(ascending=False).head(n=500).index].T
  if return_df:
    return subset
  else:
    return subset.values
