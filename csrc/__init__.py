import importlib.metadata as _m

def _require(pkg):
    try:
        _m.version(pkg)
    except _m.PackageNotFoundError:
        raise RuntimeError(f"Package '{pkg}' must be installed before importing ops3d")

_require("torch")
_require("numpy")
del _require, _m