from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lmKAN',
    packages=['lmKAN'],
    ext_modules=[
        CUDAExtension(
            name='lmKAN_kernels',
            sources=[
                'cuda_kernels/kernels.cu',
                'cuda_kernels/utilities.cu',
                'cuda_kernels/kernels_dkan_full.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    #'-gencode', 'arch=compute_90,code=sm_90',
                    # Include line info for better profiling and debugging
                    '-lineinfo',
                    # Pass optimization flags to the PTX assembler
                    '-Xptxas', '-O3',
                    # Disable L1 cache for global memory accesses
                    '-Xptxas', '-dlcm=cg',
                ],
            },
        )
    ],
    install_requires=[
        'torch',
        'numpy'
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
