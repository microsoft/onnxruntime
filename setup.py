#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from setuptools import setup, find_packages, Extension
from distutils import log as logger
from distutils.command.build_ext import build_ext as _build_ext
from glob import glob
from os import path, getcwd, environ, remove
from shutil import copyfile
import platform
import subprocess
import sys
import datetime

nightly_build = False
package_name = 'onnxruntime'

if '--use_tensorrt' in sys.argv:
    package_name = 'onnxruntime-gpu-tensorrt'
    sys.argv.remove('--use_tensorrt')
    if '--nightly_build' in sys.argv:
        package_name = 'ort-trt-nightly'
        nightly_build = True
        sys.argv.remove('--nightly_build')
elif '--use_cuda' in sys.argv:
    package_name = 'onnxruntime-gpu'
    sys.argv.remove('--use_cuda')
    if '--nightly_build' in sys.argv:
        package_name = 'ort-gpu-nightly'
        nightly_build = True
        sys.argv.remove('--nightly_build')
elif '--use_ngraph' in sys.argv:
    package_name = 'onnxruntime-ngraph'
    sys.argv.remove('--use_ngraph')

elif '--use_openvino' in sys.argv:
    package_name = 'onnxruntime-openvino'

elif '--use_nuphar' in sys.argv:
    package_name = 'onnxruntime-nuphar'
    sys.argv.remove('--use_nuphar')

if '--nightly_build' in sys.argv:
    package_name = 'ort-nightly'
    nightly_build = True
    sys.argv.remove('--nightly_build')

is_manylinux1 = False
if environ.get('AUDITWHEEL_PLAT', None) == 'manylinux1_x86_64' or environ.get('AUDITWHEEL_PLAT', None) == 'manylinux2010_x86_64' :
    is_manylinux1 = True


class build_ext(_build_ext):
    def build_extension(self, ext):
        dest_file = self.get_ext_fullpath(ext.name)
        logger.info('copying %s -> %s', ext.sources[0], dest_file)
        copyfile(ext.sources[0], dest_file)


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            if not is_manylinux1:
                self.root_is_pure = False

        def _rewrite_ld_preload(self, to_preload):
            with open('onnxruntime/capi/_ld_preload.py', 'rt') as f:
                ld_preload = f.read().splitlines()
            with open('onnxruntime/capi/_ld_preload.py', 'wt') as f:
                for line in ld_preload:
                    f.write(line)
                    f.write('\n')
                    if 'LD_PRELOAD_BEGIN_MARK' in line:
                        break
                if len(to_preload) > 0:
                    f.write('from ctypes import CDLL, RTLD_GLOBAL\n')
                    for library in to_preload:
                        f.write('_{} = CDLL("{}", mode=RTLD_GLOBAL)\n'.format(library.split('.')[0], library))

        def run(self):
            if is_manylinux1:
                source = 'onnxruntime/capi/onnxruntime_pybind11_state.so'
                dest = 'onnxruntime/capi/onnxruntime_pybind11_state_manylinux1.so'
                logger.info('copying %s -> %s', source, dest)
                copyfile(source, dest)
                result = subprocess.run(['patchelf', '--print-needed', dest], check=True, stdout=subprocess.PIPE, universal_newlines=True)
                cuda_dependencies = ['libcublas.so', 'libcudnn.so', 'libcudart.so']
                to_preload = []
                args = ['patchelf', '--debug']
                for line in result.stdout.split('\n'):
                    for dependency in cuda_dependencies:
                        if dependency in line:
                            to_preload.append(line)
                            args.extend(['--remove-needed', line])
                args.append(dest)
                if len(to_preload) > 0:
                    subprocess.run(args, check=True, stdout=subprocess.PIPE)
                self._rewrite_ld_preload(to_preload)
            _bdist_wheel.run(self)
            if is_manylinux1:
                file = glob(path.join(self.dist_dir, '*linux*.whl'))[0]
                logger.info('repairing %s for manylinux1', file)
                try:
                    subprocess.run(['auditwheel', 'repair', '-w', self.dist_dir, file], check=True, stdout=subprocess.PIPE)
                finally:
                    logger.info('removing %s', file)
                    remove(file)

except ImportError:
    bdist_wheel = None

# Additional binaries
if platform.system() == 'Linux':
  libs = ['onnxruntime_pybind11_state.so', 'libmkldnn.so.1', 'libmklml_intel.so', 'libiomp5.so', 'mimalloc.so']
  # nGraph Libs
  libs.extend(['libngraph.so', 'libcodegen.so', 'libcpu_backend.so', 'libmkldnn.so', 'libtbb_debug.so', 'libtbb_debug.so.2', 'libtbb.so', 'libtbb.so.2'])
  # Nuphar Libs
  libs.extend(['libtvm.so.0.5.1'])
  # Openvino Libs
  libs.extend(['libcpu_extension.so'])
  if nightly_build:
    libs.extend(['libonnxruntime_pywrapper.so'])
elif platform.system() == "Darwin":
  libs = ['onnxruntime_pybind11_state.so', 'libmkldnn.1.dylib', 'mimalloc.so'] # TODO add libmklml and libiomp5 later.
  if nightly_build:
    libs.extend(['libonnxruntime_pywrapper.dylib'])
else:
  libs = ['onnxruntime_pybind11_state.pyd', 'mkldnn.dll', 'mklml.dll', 'libiomp5md.dll']
  libs.extend(['ngraph.dll', 'cpu_backend.dll', 'tbb.dll', 'mimalloc-override.dll', 'mimalloc-redirect.dll', 'mimalloc-redirect32.dll'])
  # Nuphar Libs
  libs.extend(['tvm.dll'])
  # Openvino Libs
  libs.extend(['cpu_extension.dll'])
  if nightly_build:
    libs.extend(['onnxruntime_pywrapper.dll'])

if is_manylinux1:
    data = ['capi/libonnxruntime_pywrapper.so'] if nightly_build else []
    ext_modules = [
        Extension(
            'onnxruntime.capi.onnxruntime_pybind11_state',
            ['onnxruntime/capi/onnxruntime_pybind11_state_manylinux1.so'],
        ),
    ]
else:
    data = [path.join('capi', x) for x in libs if path.isfile(path.join('onnxruntime', 'capi', x))]
    ext_modules = []


python_modules_list = list()
if '--use_openvino' in sys.argv:
  #Adding python modules required for openvino ep
  python_modules_list.extend(['openvino_mo', 'openvino_emitter'])
  sys.argv.remove('--use_openvino')

# Additional examples
examples_names = ["mul_1.onnx", "logreg_iris.onnx", "sigmoid.onnx"]
examples = [path.join('datasets', x) for x in examples_names]

# Extra files such as EULA and ThirdPartyNotices
extra = ["LICENSE", "ThirdPartyNotices.txt", "Privacy.md"]

# Description
README = path.join(getcwd(), "docs/python/README.rst")
if not path.exists(README):
    this = path.dirname(__file__)
    README = path.join(this, "docs/python/README.rst")
if not path.exists(README):
    raise FileNotFoundError("Unable to find 'README.rst'")
with open(README) as f:
    long_description = f.read()


version_number = ''
with open('VERSION_NUMBER') as f:
    version_number = f.readline().strip()
if nightly_build:
    date_suffix = str(datetime.datetime.now().date().strftime("%m%d"))
    version_number = version_number + ".dev" + date_suffix

# Setup
setup(
    name=package_name,
    version=version_number,
    description='ONNX Runtime Python bindings',
    long_description=long_description,
    author='Microsoft Corporation',
    author_email='onnx@microsoft.com',
    cmdclass={'bdist_wheel': bdist_wheel, 'build_ext': build_ext},
    license="MIT License",
    packages=['onnxruntime',
              'onnxruntime.backend',
              'onnxruntime.capi',
              'onnxruntime.datasets',
              'onnxruntime.tools',
              ] + (['onnxruntime.nuphar'] if package_name == 'onnxruntime-nuphar' else []),
    ext_modules=ext_modules,
    package_data={
        'onnxruntime': data + examples + extra,
    },
    py_modules=python_modules_list,
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
