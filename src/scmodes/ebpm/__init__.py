"""Algorithms for solving Empirical Bayes Poisson Means (EBPM)

EBPM is the problem of estimating g, where

x_{ij} ~ Poisson(s_i \lambda_{ij})
\lambda_{ij} ~ g_j(.)

For g_j a point mass, or a member of the family of Gamma distributions, or
point-Gamma distributions, the marginal likelihood is analytic.

"""
from .wrappers import *
