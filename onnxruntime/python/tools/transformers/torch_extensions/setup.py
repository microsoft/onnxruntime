from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name='longformer_attention',
      ext_modules=[
          CppExtension(name='longformer_attention',
                       sources=['longformer_attention.cpp'],
                       include_dirs=[],
                       extra_compile_args=['-g'])
      ],
      cmdclass={'build_ext': BuildExtension})
