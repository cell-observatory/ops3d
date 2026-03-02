import pytest

try:
    from ops3d import _C
except ImportError:
    _C = None


@pytest.mark.skipif(_C is None, reason="ops3d._C not built (run pip install -e . or install from wheel)")
def test_C_extension_importable():
    """ops3d._C must be importable when the package is installed from a wheel or after build."""
    assert _C is not None


@pytest.mark.skipif(_C is None, reason="ops3d._C not built")
def test_C_extension_has_nms_3d():
    assert hasattr(_C, "nms_3d")


@pytest.mark.skipif(_C is None, reason="ops3d._C not built")
def test_C_extension_has_roi_align_3d():
    assert hasattr(_C, "roi_align_3d_forward")
    assert hasattr(_C, "roi_align_3d_backward")


@pytest.mark.skipif(_C is None, reason="ops3d._C not built")
def test_C_extension_has_flash_deform_attn():
    assert hasattr(_C, "flash_deform_attn_forward")
    assert hasattr(_C, "flash_deform_attn_backward")
