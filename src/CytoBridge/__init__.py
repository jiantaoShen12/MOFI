"""CytoBridge top-level package.

Keep imports lightweight so submodule entry scripts can run in environments where
optional dependencies (for example `scanpy`) are unavailable.
"""

from importlib import import_module

__version__ = "0.2.0"

__all__ = ["pp", "tl", "pl", "utils", "Map", "reproducibility", "manuscript"]


def __getattr__(name: str):
    """Lazily expose the historical ``CytoBridge.<submodule>`` API."""
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
