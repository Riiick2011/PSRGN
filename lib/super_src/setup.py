from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='Super1D',
    version="2.2.0",
    author="",
    author_email="",
    description="A small package for 1d aligment in cuda",
    long_description="I will write a longer description here :)",
    long_description_content_type="",
    url="",
    ext_modules=[
        CppExtension(
            name = 'Super1D',
            sources = [
              'Super1D_cuda.cpp',
              'Super1D_cuda_kernal.cu',
            ],
            extra_compile_args={'cxx': [],
              'nvcc': ['--expt-relaxed-constexpr']}
         )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
