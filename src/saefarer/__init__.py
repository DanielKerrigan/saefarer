import importlib.metadata

try:
    __version__ = importlib.metadata.version("saefarer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
