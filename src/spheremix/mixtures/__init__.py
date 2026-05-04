"""
Package for mixture models of spherical probability distributions.
"""

from .initializer import MeanInitStrategy as MeanInitStrategy
from .initializer import create_mean_initializer as create_mean_initializer
from .spherical_cauchy import SphericalCauchyMixture as SphericalCauchyMixture

__all__ = []
