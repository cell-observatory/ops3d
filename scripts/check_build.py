from __future__ import print_function

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getcwd() != ROOT:
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

def main():
    import torch
    from torch.utils.cpp_extension import load

    check_src = os.path.join(ROOT, "csrc", "check_build.cpp")
    if not os.path.isfile(check_src):
        print("check_build.py: missing", check_src, file=sys.stderr)
        return 1

    print("Building template extension (fast, no CUDA)...", flush=True)
    try:
        mod = load(
            name="check_build",
            sources=[check_src],
            verbose=False,
        )
    except Exception as e:
        print("Template build failed:", e, file=sys.stderr)
        return 1

    if not getattr(mod, "check_build_ok", None):
        print("Template module missing check_build_ok", file=sys.stderr)
        return 1

    ok = mod.check_build_ok()
    if not ok:
        print("check_build_ok() returned False", file=sys.stderr)
        return 1

    print("Template build OK. Proceed with full compile (e.g. pip wheel .).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
