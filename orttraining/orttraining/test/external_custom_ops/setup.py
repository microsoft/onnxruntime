import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from subprocess import CalledProcessError

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake",
             "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir), 
             ext.sourcedir], cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", "."], cwd=self.build_temp
        )

setup(name='custom_ops',
    version='0.1',
    author='',
    author_email='',
    description='Apollo onnxruntime custom ops',
    long_description='',
    ext_modules=[CMakeExtension('custom_ops')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False
)
