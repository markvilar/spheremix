"""
Package for spherical probability distributions and mixture models.
"""

from importlib.metadata import version

from . import distributions as distributions
from . import mixtures as mixtures
from .protocols import SphericalClusterer as SphericalClusterer
from .protocols import SphericalMixtureModel as SphericalMixtureModel

__version__ = version("spheremix")

__all__ = ["__version__"]
