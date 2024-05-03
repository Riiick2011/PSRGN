from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='Guass1D',
    version="2.2.0",
    author="",
    author_email="",
    description="A small package for 1d aligment in cuda",
    long_description="I will write a longer description here :)",
    long_description_content_type="",
    url="",
    ext_modules=[
        CUDAExtension(
            name = 'Guass1D',
            sources = [
              'Guass1D_cuda.cpp',
              'Guass1D_cuda_kernal.cu',
            ]
         )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
