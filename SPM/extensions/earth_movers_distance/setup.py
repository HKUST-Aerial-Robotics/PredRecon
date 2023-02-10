from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='emd_cuda',
    ext_modules=[
        CUDAExtension(
            name='emd_cuda',
            sources=[
                'emd.cpp',
                'emd_kernel.cu',
            ],
            # extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
