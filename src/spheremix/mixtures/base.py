"""
Serialization mixin for spherical mixture models.
"""

from abc import ABC, abstractmethod
from importlib.metadata import version as _get_version
from typing import ClassVar, Self


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
