# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def get_extra_deps(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return json.load(fp)["extra_dependencies"]


# use techniques described at https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
# Don't use technique 6 since it needs extra dependencies.
VERSION = get_version("olive/version.py")
EXTRAS = get_extra_deps("olive/olive_config.json")

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")) as req_file:
    requirements = req_file.read().splitlines()


CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]

long_description = (
    "Olive: Simplify ML Model Finetuning, Conversion, Quantization, and Optimization for CPUs, GPUs and NPUs"
)

description = long_description.split(".", maxsplit=1)[0] + "."

setup(
    name="olive-ai",
    version=VERSION,
    description=description,
    long_description=long_description,
    author="Microsoft Corporation",
    author_email="olivedevteam@microsoft.com",
    license="MIT License",
    classifiers=CLASSIFIERS,
    url="https://microsoft.github.io/Olive/",
    download_url="https://github.com/microsoft/Olive/tags",
    packages=find_packages(include=["olive*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require=EXTRAS,
    include_package_data=False,
    package_data={
        "olive": ["olive_config.json"],
        "olive.assets.io_configs": ["*.yaml"],
        "olive.auto_optimizer": ["config_template/*.yaml", "workflow_template/*.yaml"],
        "olive.platform_sdk.qualcomm": ["create_python_env.sh", "create_python_env.ps1", "copy_libcdsprpc.ps1"],
        "olive.systems.python_environment": ["common_requirements.txt"],
    },
    data_files=[],
    entry_points={
        "console_scripts": ["olive=olive.cli.launcher:main"],
    },
)
