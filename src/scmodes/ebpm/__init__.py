"""Algorithms for solving Empirical Bayes Poisson Means (EBPM)

EBPM is the problem of estimating g, where

x_{ij} ~ Poisson(s_i \lambda_{ij})
\lambda_{ij} ~ g_j(.)

For g_j a point mass, or a member of the family of Gamma distributions, or
point-Gamma distributions, the marginal likelihood is analytic. Other choices
of g_j require specialized inference algorithms.

We provide simple implementations for analyzing a single gene at a time under
scmodes.ebpm. We additionally provide specialized implementations of ebpm_gamma
and ebpm_point_gamma for fitting many EBPM problems in parallel on a GPU in
scmodes.ebpm.sgd (which requires an explicit import).

"""
from .wrappers import *
