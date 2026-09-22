import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# On this server nvcc is installed under /usr/bin and its CUDA headers are
# installed separately in /usr/include.  Make the path explicit so that a
# conda host compiler (which uses an isolated sysroot) can find cuda_runtime.h.
cuda_include = os.environ.get('SHIFT_CUDA_INCLUDE', '/usr/include')

setup(
    name='shift_cuda_linear_cpp',
    ext_modules=[
        CUDAExtension('shift_cuda', [
            'shift_cuda.cpp',
            'shift_cuda_kernel.cu',
        ],
        include_dirs=[cuda_include],
        # ``include_dirs`` may be omitted from the nvcc host-compiler
        # invocation when the CUDA toolkit is split from the system compiler.
        # Pass it explicitly to both compilers as well.
        extra_compile_args={
            'cxx': [f'-I{cuda_include}'],
            'nvcc': [f'-I{cuda_include}'],
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
