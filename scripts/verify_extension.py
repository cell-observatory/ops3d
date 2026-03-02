from __future__ import print_function

import sys


def main():
    try:
        from ops3d import _C
    except ImportError as e:
        print("ops3d._C is not installed or failed to import:", e, file=sys.stderr)
        return 1

    if _C is None:
        print("ops3d._C is None (extension not built).", file=sys.stderr)
        return 1

    required = [
        "nms_3d",
        "roi_align_3d_forward",
        "roi_align_3d_backward",
        "flash_deform_attn_forward",
        "flash_deform_attn_backward",
    ]
    missing = [name for name in required if not hasattr(_C, name)]
    if missing:
        print("ops3d._C missing symbols:", missing, file=sys.stderr)
        return 1

    print("ops3d._C is installed and has required symbols:", required)
    return 0


if __name__ == "__main__":
    sys.exit(main())
