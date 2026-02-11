# ------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
# pylint: disable=C0103

import datetime
import logging
import platform
import shlex
import subprocess
import sys
from glob import glob
from os import environ, getcwd, path, remove
from shutil import copyfile

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as InstallCommandBase

nightly_build = False
wheel_name_suffix = None
logger = logging.getLogger()


def parse_arg_remove_boolean(argv, arg_name):
    arg_value = False
    if arg_name in sys.argv:
        arg_value = True
        argv.remove(arg_name)

    return arg_value


def parse_arg_remove_string(argv, arg_name_equal):
    arg_value = None
    for arg in sys.argv[1:]:
        if arg.startswith(arg_name_equal):
            arg_value = arg[len(arg_name_equal) :]
            sys.argv.remove(arg)
            break

    return arg_value


# Any combination of the following arguments can be applied

if parse_arg_remove_boolean(sys.argv, "--nightly_build"):
    nightly_build = True

wheel_name_suffix = parse_arg_remove_string(sys.argv, "--wheel_name_suffix=")

is_qnn = True
package_name = "onnxruntime-qnn"
qnn_version = parse_arg_remove_string(sys.argv, "--qnn_version=")

# PEP 513 defined manylinux1_x86_64 and manylinux1_i686
# PEP 571 defined manylinux2010_x86_64 and manylinux2010_i686
# PEP 599 defines the following platform tags:
# manylinux2014_x86_64
# manylinux2014_i686
# manylinux2014_aarch64
# manylinux2014_armv7l
# manylinux2014_ppc64
# manylinux2014_ppc64le
# manylinux2014_s390x
manylinux_tags = [
    "manylinux1_x86_64",
    "manylinux1_i686",
    "manylinux2010_x86_64",
    "manylinux2010_i686",
    "manylinux2014_x86_64",
    "manylinux2014_i686",
    "manylinux2014_aarch64",
    "manylinux2014_armv7l",
    "manylinux2014_ppc64",
    "manylinux2014_ppc64le",
    "manylinux2014_s390x",
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "manylinux_2_34_x86_64",
    "manylinux_2_34_aarch64",
]
is_manylinux = environ.get("AUDITWHEEL_PLAT", None) in manylinux_tags


class build_ext(_build_ext):  # noqa: N801
    def build_extension(self, ext):
        dest_file = self.get_ext_fullpath(ext.name)
        logger.info("copying %s -> %s", ext.sources[0], dest_file)
        copyfile(ext.sources[0], dest_file)


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):  # noqa: N801
        """Helper functions to create wheel package"""

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            if not is_manylinux:
                self.root_is_pure = False

        def get_tag(self):
            # Override to use py3 tag instead of specific Python versions
            # This makes the wheel work with any Python 3.10+
            # Since there are no .pyd files, only native DLLs/SOs
            impl, abi, plat = super().get_tag()
            # Use 'py3' instead of specific version like 'cp310', 'cp311', etc.
            impl = "py3"
            # Use 'none' for ABI since we have no Python C extensions
            abi = "none"
            return impl, abi, plat

        def run(self):
            _bdist_wheel.run(self)
            if is_manylinux and not disable_auditwheel_repair and not is_qnn:
                assert self.dist_dir is not None
                file = glob(path.join(self.dist_dir, "*linux*.whl"))[0]
                logger.info("repairing %s for manylinux1", file)
                auditwheel_cmd = ["auditwheel", "-v", "repair", "-w", self.dist_dir, file]
                logger.info("Running %s", " ".join([shlex.quote(arg) for arg in auditwheel_cmd]))
                try:
                    subprocess.run(auditwheel_cmd, check=True, stdout=subprocess.PIPE)
                finally:
                    logger.info("removing %s", file)
                    remove(file)

except ImportError as error:
    print("Error importing dependencies:")
    print(error)
    bdist_wheel = None


class InstallCommand(InstallCommandBase):
    def finalize_options(self):
        ret = InstallCommandBase.finalize_options(self)
        self.install_lib = self.install_platlib
        return ret


providers_qnn = "onnxruntime_providers_qnn"
if platform.system() == "Linux":
    providers_qnn = "lib" + providers_qnn + ".so"
elif platform.system() == "Windows":
    providers_qnn = providers_qnn + ".dll"

# Additional binaries
dl_libs = []
libs = []

if platform.system() == "Linux" or platform.system() == "AIX":
    dl_libs.append(providers_qnn)
    # QNN-EP is built as shared libs
    libs.append(providers_qnn)
    # QNN
    qnn_deps = [
        "libGenie.so",
        "libQnnCpu.so",
        "libQnnGpu.so",
        "libQnnHtp.so",
        "libQnnHtpPrepare.so",
        "libQnnHtpV68Skel.so",
        "libQnnHtpV68Stub.so",
        "libQnnIr.so",
        "libQnnSaver.so",
        "libQnnSystem.so",
        "libHtpPrepare.so",
    ]
    dl_libs.extend(qnn_deps)
else:
    # QNN-EP is built as shared libs
    libs = [providers_qnn]
    # QNN V68/V73/V81 dependencies
    qnn_deps = [
        "Genie.dll",
        "QnnCpu.dll",
        "QnnGpu.dll",
        "QnnHtp.dll",
        "QnnIr.dll",
        "QnnSaver.dll",
        "QnnSystem.dll",
        "QnnHtpPrepare.dll",
        "QnnHtpV81Stub.dll",
        "libQnnHtpV81Skel.so",
        "libqnnhtpv81.cat",
        "QnnHtpV73Stub.dll",
        "libQnnHtpV73Skel.so",
        "libqnnhtpv73.cat",
        "QnnHtpV68Stub.dll",
        "libQnnHtpV68Skel.so",
    ]
    libs.extend(qnn_deps)

if is_manylinux:
    data = list(dl_libs)
else:
    data = list(libs)
ext_modules = []

# Extra files such as EULA and ThirdPartyNotices (and Qualcomm License, only for QNN release packages)
extra = ["LICENSE", "ThirdPartyNotices.txt", "Privacy.md", "Qualcomm_LICENSE.pdf"]

# Description
readme_file = "docs/python/README.rst"
README = path.join(getcwd(), readme_file)
if not path.exists(README):
    this = path.dirname(__file__)
    README = path.join(this, readme_file)

if not path.exists(README):
    raise FileNotFoundError("Unable to find 'README.rst'")
with open(README, encoding="utf-8") as fdesc:
    long_description = fdesc.read()

data_files = []
requirements_file = "requirements.txt"

local_version = None
disable_auditwheel_repair = parse_arg_remove_boolean(sys.argv, "--disable_auditwheel_repair")

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
]

packages = ["onnxruntime_qnn"]
package_data = {"onnxruntime_qnn": data + extra}

version_number = ""
with open("VERSION_NUMBER") as f:
    version_number = f.readline().strip()
if nightly_build:
    # https://docs.microsoft.com/en-us/azure/devops/pipelines/build/variables
    build_suffix = environ.get("BUILD_BUILDNUMBER")
    if build_suffix is None:
        # The following line is only for local testing
        build_suffix = str(datetime.datetime.now().date().strftime("%Y%m%d"))
    else:
        build_suffix = build_suffix.replace(".", "")

    if len(build_suffix) > 8 and len(build_suffix) < 12:
        # we want to format the build_suffix to avoid (the 12th run on 20210630 vs the first run on 20210701):
        # 2021063012 > 202107011
        # in above 2021063012 is treated as the latest which is incorrect.
        # we want to convert the format to:
        # 20210630012 < 20210701001
        # where the first 8 digits are date. the last 3 digits are run count.
        # as long as there are less than 1000 runs per day, we will not have the problem.
        # to test this code locally, run:
        # NIGHTLY_BUILD=1 BUILD_BUILDNUMBER=202107011 python tools/ci_build/build.py --config RelWithDebInfo \
        #   --enable_training --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        #   --nccl_home /usr/lib/x86_64-linux-gnu/ --build_dir build/Linux --build --build_wheel --skip_tests \
        #   --cuda_version 11.1
        def check_date_format(date_str):
            try:
                datetime.datetime.strptime(date_str, "%Y%m%d")
                return True
            except Exception:
                return False

        def reformat_run_count(count_str):
            try:
                count = int(count_str)
                if count >= 0 and count < 1000:
                    return f"{count:03}"
                elif count >= 1000:
                    raise RuntimeError(f"Too many builds for the same day: {count}")
                return ""
            except Exception:
                return ""

        build_suffix_is_date_format = check_date_format(build_suffix[:8])
        build_suffix_run_count = reformat_run_count(build_suffix[8:])
        if build_suffix_is_date_format and build_suffix_run_count:
            build_suffix = build_suffix[:8] + build_suffix_run_count
    elif len(build_suffix) >= 12:
        raise RuntimeError(f'Incorrect build suffix: "{build_suffix}"')

    version_number = version_number + ".dev" + build_suffix

if local_version:
    version_number = version_number + local_version

cmd_classes = {}
if bdist_wheel is not None:
    cmd_classes["bdist_wheel"] = bdist_wheel
cmd_classes["install"] = InstallCommand
cmd_classes["build_ext"] = build_ext

requirements_path = path.join(getcwd(), requirements_file)
if not path.exists(requirements_path):
    this = path.dirname(__file__)
    requirements_path = path.join(this, requirements_file)
if not path.exists(requirements_path):
    raise FileNotFoundError("Unable to find " + requirements_file)
with open(requirements_path) as f:
    install_requires = f.read().splitlines()


def save_build_and_package_info(package_name, version_number, qnn_version):
    sys.path.append(path.join(path.dirname(__file__), "onnxruntime", "python"))

    version_path = path.join("onnxruntime_qnn", "build_and_package_info.py")
    with open(version_path, "w") as f:
        f.write(f"package_name = '{package_name}'\n")
        f.write(f"__version__ = '{version_number}'\n")
        if qnn_version:
            f.write(f"qnn_version = '{qnn_version}'\n")


save_build_and_package_info(package_name, version_number, qnn_version)

extras_require = {}
setup(
    name=package_name,
    version=version_number,
    description="ONNX Runtime is a runtime accelerator for Machine Learning models",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Microsoft Corporation",
    author_email="onnxruntime@microsoft.com",
    cmdclass=cmd_classes,
    license="MIT License",
    packages=packages,
    ext_modules=ext_modules,
    package_data=package_data,
    url="https://onnxruntime.ai",
    download_url="https://github.com/onnxruntime/onnxruntime-qnn/tags",
    data_files=data_files,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10",
    keywords="onnx machine learning",
    entry_points={
        "console_scripts": [
            "onnxruntime_test = onnxruntime.tools.onnxruntime_test:main",
        ]
    },
    classifiers=classifiers,
)
