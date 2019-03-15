#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from setuptools import setup, find_packages
from os import path, getcwd
import platform
import sys

package_name = 'onnxruntime'
if '--use_tensorrt' in sys.argv:
    package_name = 'onnxruntime-gpu-tensorrt'
    sys.argv.remove('--use_tensorrt')
elif '--use_cuda' in sys.argv:
    package_name = 'onnxruntime-gpu'
    sys.argv.remove('--use_cuda')

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

# Additional binaries
if platform.system() == 'Linux':
  libs = ['onnxruntime_pybind11_state.so', 'libmkldnn.so.0', 'libmklml_intel.so', 'libiomp5.so']
elif platform.system() == "Darwin":
  libs = ['onnxruntime_pybind11_state.so', 'libmkldnn.0.dylib'] # TODO add libmklml and libiomp5 later.
else:
  libs = ['onnxruntime_pybind11_state.pyd', 'mkldnn.dll', 'mklml.dll', 'libiomp5md.dll']

data = [path.join('capi', x) for x in libs if path.isfile(path.join('onnxruntime', 'capi', x))]

# Additional examples
examples_names = ["mul_1.pb", "logreg_iris.onnx", "sigmoid.onnx"]
examples = [path.join('datasets', x) for x in examples_names]

# Extra files such as EULA and ThirdPartyNotices
extra = ["LICENSE", "ThirdPartyNotices.txt"]

# Description
README = path.join(getcwd(), "docs/python/README.rst")
if not path.exists(README):
    this = path.dirname(__file__)
    README = path.join(this, "docs/python/README.rst")
if not path.exists(README):
    raise FileNotFoundError("Unable to find 'README.rst'")
with open(README) as f:
    long_description = f.read()


# Setup
setup(
    name=package_name,
    version='0.3.0',
    description='ONNX Runtime Python bindings',
    long_description=long_description,
    author='Microsoft Corporation',
    author_email='onnx@microsoft.com',
    cmdclass={'bdist_wheel': bdist_wheel},
    license="MIT License",
    packages=['onnxruntime',
              'onnxruntime.backend',
              'onnxruntime.capi',
              'onnxruntime.datasets',
              'onnxruntime.tools',
              ],
    package_data={
        'onnxruntime': data + examples + extra,
    },
    extras_require={
        'backend': ['onnx>=1.2.3'],
        'numpy': ['numpy>=1.15.0']
    },
    entry_points= {
        'console_scripts': [
            'onnxruntime_test = onnxruntime.tools.onnxruntime_test:main',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'],
    )
