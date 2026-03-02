import os
import glob
import subprocess
import sys

from setuptools import setup, find_packages

import numpy
import torch

from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, CppExtension, CUDA_HOME


def run_check_build():
    """Run template build check before full CUDA compile. Exits on failure."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "check_build.py")
    if not os.path.isfile(script):
        return
    rc = subprocess.call([sys.executable, script])
    if rc != 0:
        print("Template build check failed; aborting full build.", file=sys.stderr)
        sys.exit(rc)


class BuildExtWithCheck(cpp_extension.BuildExtension):
    """Run template extension check before building the real CUDA extension."""

    def run(self):
        run_check_build()
        super().run()


def load_requires():
    here = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(here, "requirements.txt")
    with open(req_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return lines


def print_compile_env():
    import subprocess
    print('torch :', torch.__version__)
    print('torch.cuda :', torch.version.cuda)
    print("CUDA_HOME :", CUDA_HOME)
    try:
        with open(os.devnull, 'w') as devnull:
            gcc = subprocess.check_output(['gcc', '--version'],
                                          stderr=devnull).decode().rstrip('\r\n').split('\n')[0]
        print('gcc :', gcc)
    except Exception as e:
        pass


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'csrc')

    main_files = [
        f for f in glob.glob(os.path.join(extensions_dir, '*.cpp'))
        if os.path.basename(f) != 'check_build.cpp'
    ]
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = [os.path.relpath(f, start=this_dir) for f in main_files]
    source_cuda = [os.path.relpath(f, start=this_dir) for f in source_cuda]

    sources = sources + source_cuda

    define_macros = []
    extra_compile_args = {}

    extension = CUDAExtension
    define_macros += [('WITH_CUDA', None)]
    nvcc_flags = os.getenv('NVCC_FLAGS', '').split() if os.getenv('NVCC_FLAGS') else []
    nvcc_deform_attn_flags = [
                            "-DCUDA_HAS_FP16=1", 
                            "-D__CUDA_NO_HALF_OPERATORS__", 
                            "-D__CUDA_NO_HALF_CONVERSIONS__", 
                            "-D__CUDA_NO_HALF2_OPERATORS__",
                                ]

    extra_compile_args = {
        'cxx': ['-O2'],  # or '-O0' for debugging
        'nvcc': nvcc_flags + ['-allow-unsupported-compiler'] + nvcc_deform_attn_flags,
    }

    # Use PyTorch's CUDA include directories to match PyTorch's CUDA version
    # This ensures we compile against the same CUDA version that PyTorch uses
    torch_include_dirs = cpp_extension.include_paths()
    
    include_dirs = [
        os.path.relpath(extensions_dir, start=this_dir), 
        numpy.get_include(),
    ] + torch_include_dirs
    
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    # Link against PyTorch's CUDA libraries to avoid version mismatches
    # BuildExtension will automatically add PyTorch's library paths
    extra_link_args = [f"-Wl,-rpath,{torch_lib}"]
    
    # Ensure we use PyTorch's CUDA runtime libraries, not system CUDA_HOME
    # The curand library should come from PyTorch's CUDA installation
    library_dirs = [torch_lib]

    ext_modules = [
        extension(
            'ops3d._C',  
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            library_dirs=library_dirs,
            libraries=['curand'],  
        )
    ]

    return ext_modules


if __name__ == "__main__":
    print_compile_env()
    setup(
        name="ops3d",
        version='0.1.0',
        description='Custom 3D Operations for PyTorch',
        packages=find_packages(exclude=["tests", "tests.*"]),
        package_data={"ops3d": ["*.py"]},
        install_requires=load_requires(),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtWithCheck},
        classifiers=[
            "Programming Language :: Python :: 3",
        ],
        zip_safe=False,
    )