"""
Load the compiled ops3d._C extension. Tries the in-package load first, then
the installed .so from site-packages (so running from repo with a wheel installed
still finds _C even if the wheel only had the extension).
"""
from __future__ import absolute_import

import importlib.util
import os
import sys
import types

try:
    from . import _C
except ImportError:
    _C = None
    # Running from repo while wheel is installed: load .so directly from site-packages
    for p in sys.path:
        if "site-packages" not in p and "dist-packages" not in p:
            continue
        ops3d_dir = os.path.join(p, "ops3d")
        if not os.path.isdir(ops3d_dir):
            continue
        for name in os.listdir(ops3d_dir):
            if name.startswith("_C") and name.endswith(".so"):
                so_path = os.path.join(ops3d_dir, name)
                try:
                    spec = importlib.util.spec_from_file_location(
                        "ops3d._C", so_path, submodule_search_locations=[]
                    )
                    if spec is None or spec.loader is None:
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    # Stub ops3d so the extension can load as ops3d._C
                    stub = types.ModuleType("ops3d")
                    stub.__path__ = [ops3d_dir]
                    saved = {
                        k: sys.modules.pop(k, None)
                        for k in ("ops3d", "ops3d._C")
                        if k in sys.modules
                    }
                    sys.modules["ops3d"] = stub
                    sys.modules["ops3d._C"] = mod
                    spec.loader.exec_module(mod)
                    _C = mod  # noqa: F811
                    break
                except Exception:
                    _C = None
                finally:
                    for k, v in saved.items():
                        if v is not None:
                            sys.modules[k] = v
                break
        if _C is not None:
            break
