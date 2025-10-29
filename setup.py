import os
import glob

from setuptools import setup

import numpy
import torch

from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, CppExtension, CUDA_HOME


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

    # find all .cpp and .cu files in csrc directory:
    main_files = glob.glob(os.path.join(extensions_dir, '*.cpp'))
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

    # sources = [os.path.join(extensions_dir, s) for s in sources]
    # include_dirs = [extensions_dir, numpy.get_include()]

    include_dirs = [
        os.path.relpath(extensions_dir, start=this_dir), 
        numpy.get_include(),
        os.path.join(CUDA_HOME or "/usr/local/cuda", "include"),
        ]
    
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    extra_link_args = [f"-Wl,-rpath,{torch_lib}"]

    ext_modules = [
        extension(
            'ops3d._C',  
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
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
        packages=["ops3d"],
        package_dir={"ops3d": "."},
        # install_requires=load_requires(),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': cpp_extension.BuildExtension},
        classifiers=[
            "Programming Language :: Python :: 3",
        ],
        zip_safe=False,
    )