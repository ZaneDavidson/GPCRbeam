from importlib.metadata import PackageNotFoundError, version

__version__: str
try:
    __version__ = version("gpcrbeam")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
