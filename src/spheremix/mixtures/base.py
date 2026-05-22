"""
Base protocol and serialization mixin for spherical mixture models.
"""

from abc import ABC, abstractmethod
from importlib.metadata import version as _get_version
from typing import ClassVar, Protocol, Self, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------


@runtime_checkable
class SphericalMixtureBase(Protocol):
    """
    Structural interface for spherical mixture models.

    Any class that implements the methods below with the expected signatures
    satisfies this protocol, without requiring explicit inheritance.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Fit the model to unit-vector data X."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to its most probable component."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample posterior probabilities over components."""
        ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the model and return hard component assignments."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        """Return the mean per-sample log-likelihood of X under the fitted model."""
        ...


# ---------------------------------------------------------------------------
# Serialization mixin
# ---------------------------------------------------------------------------


class SerializableMixin(ABC):
    """
    Mixin that provides ``to_dict`` / ``from_dict`` serialization.

    Subclasses must implement ``_params_to_dict`` and ``_params_from_dict``
    to handle their own parameters. The mixin writes and validates the
    envelope (``type``, ``spheremix_version``, ``version``).

    Class attributes
    ----------------
    _SERIAL_VERSION:
        Schema version integer. Bump this when the serialized format changes
        in a backward-incompatible way.
    """

    _SERIAL_VERSION: ClassVar[int] = 1

    def to_dict(self) -> dict:
        """
        Serialize the model to a JSON-compatible dict.

        Returns
        -------
        dict
            Envelope keys (``type``, ``spheremix_version``, ``version``) plus
            the subclass-specific parameters from ``_params_to_dict``.
        """
        return {
            "type": self.__class__.__name__,
            "spheremix_version": _get_version("spheremix"),
            "version": self._SERIAL_VERSION,
            **self._params_to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """
        Restore a model from a dict produced by ``to_dict``.

        Parameters
        ----------
        data:
            Dict as returned by ``to_dict``.

        Returns
        -------
        SerializableMixin
            Restored model instance.

        Raises
        ------
        ValueError
            If ``data["type"]`` does not match the class name or
            ``data["version"]`` does not match ``_SERIAL_VERSION``.
        """
        if data.get("type") != cls.__name__:
            raise ValueError(
                f"Expected type {cls.__name__!r}, got {data.get('type')!r}."
            )
        if data.get("version") != cls._SERIAL_VERSION:
            raise ValueError(
                f"Unsupported version {data.get('version')!r}. Expected {cls._SERIAL_VERSION}."
            )
        return cls._params_from_dict(data)

    @abstractmethod
    def _params_to_dict(self) -> dict:
        """Return a dict of the subclass-specific parameters."""
        ...

    @classmethod
    @abstractmethod
    def _params_from_dict(cls, data: dict) -> Self:
        """Reconstruct an instance from the subclass-specific parameters in ``data``."""
        ...
