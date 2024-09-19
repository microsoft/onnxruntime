# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import subprocess
import sys
from subprocess import CalledProcessError  # noqa: F401

import onnx
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import onnxruntime


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
            [
                "cmake",
                f"-DPYBIND11_PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
                f"-DONNX_INCLUDE={os.path.dirname(os.path.dirname(onnx.__file__))}",
                "-DONNXRUNTIME_EXTERNAL_INCLUDE={}".format(
                    os.path.join(os.path.join(os.path.dirname(onnxruntime.__file__), "external"), "include")
                ),
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                ext.sourcedir,
            ],
            cwd=self.build_temp,
        )
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)


setup(
    name="orttraining_external_custom_ops",
    version="0.1",
    author="",
    author_email="",
    description="External custom ops example",
    long_description="",
    ext_modules=[CMakeExtension("orttrainng_external_custom_ops")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
