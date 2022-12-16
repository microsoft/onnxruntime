# ------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import datetime
import platform
import subprocess
import sys
from distutils import log as logger
from distutils.command.build_ext import build_ext as _build_ext
from glob import glob, iglob
from os import environ, getcwd, path, popen, remove
from pathlib import Path
from shutil import copyfile

from packaging.tags import sys_tags
from setuptools import Extension, setup
from setuptools.command.install import install as InstallCommandBase

nightly_build = False
package_name = "onnxruntime"
wheel_name_suffix = None


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
    package_name = "ort-nightly"
    nightly_build = True

wheel_name_suffix = parse_arg_remove_string(sys.argv, "--wheel_name_suffix=")

cuda_version = None
rocm_version = None
is_rocm = False
is_openvino = False
# The following arguments are mutually exclusive
if wheel_name_suffix == "gpu":
    # TODO: how to support multiple CUDA versions?
    cuda_version = parse_arg_remove_string(sys.argv, "--cuda_version=")
elif parse_arg_remove_boolean(sys.argv, "--use_rocm"):
    is_rocm = True
    package_name = "onnxruntime-rocm" if not nightly_build else "ort-rocm-nightly"
    rocm_version = parse_arg_remove_string(sys.argv, "--rocm_version=")
elif parse_arg_remove_boolean(sys.argv, "--use_openvino"):
    is_openvino = True
    package_name = "onnxruntime-openvino"
elif parse_arg_remove_boolean(sys.argv, "--use_dnnl"):
    package_name = "onnxruntime-dnnl"
elif parse_arg_remove_boolean(sys.argv, "--use_tvm"):
    package_name = "onnxruntime-tvm"
elif parse_arg_remove_boolean(sys.argv, "--use_vitisai"):
    package_name = "onnxruntime-vitisai"
elif parse_arg_remove_boolean(sys.argv, "--use_acl"):
    package_name = "onnxruntime-acl"
elif parse_arg_remove_boolean(sys.argv, "--use_armnn"):
    package_name = "onnxruntime-armnn"
elif parse_arg_remove_boolean(sys.argv, "--use_cann"):
    package_name = "onnxruntime-cann"

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
    "manylinux_2_27_x86_64",
    "manylinux_2_27_aarch64",
]
is_manylinux = environ.get("AUDITWHEEL_PLAT", None) in manylinux_tags


class build_ext(_build_ext):
    def build_extension(self, ext):
        dest_file = self.get_ext_fullpath(ext.name)
        logger.info("copying %s -> %s", ext.sources[0], dest_file)
        copyfile(ext.sources[0], dest_file)


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        """Helper functions to create wheel package"""

        if is_openvino and is_manylinux:

            def get_tag(self):
                _, _, plat = _bdist_wheel.get_tag(self)
                if platform.system() == "Linux":
                    # Get the right platform tag by querying the linker version
                    glibc_major, glibc_minor = popen("ldd --version | head -1").read().split()[-1].split(".")
                    """# See https://github.com/mayeut/pep600_compliance/blob/master/
                    pep600_compliance/tools/manylinux-policy.json"""
                    if glibc_major == "2" and glibc_minor == "17":
                        plat = "manylinux_2_17_x86_64.manylinux2014_x86_64"
                    else:  # For manylinux2014 and above, no alias is required
                        plat = "manylinux_%s_%s_x86_64" % (glibc_major, glibc_minor)
                tags = next(sys_tags())
                return (tags.interpreter, tags.abi, plat)

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            if not is_manylinux:
                self.root_is_pure = False

        def _rewrite_ld_preload(self, to_preload):
            with open("onnxruntime/capi/_ld_preload.py", "a") as f:
                if len(to_preload) > 0:
                    f.write("from ctypes import CDLL, RTLD_GLOBAL\n")
                    for library in to_preload:
                        f.write('_{} = CDLL("{}", mode=RTLD_GLOBAL)\n'.format(library.split(".")[0], library))

        def _rewrite_ld_preload_cuda(self, to_preload):
            with open("onnxruntime/capi/_ld_preload.py", "a") as f:
                if len(to_preload) > 0:
                    f.write("from ctypes import CDLL, RTLD_GLOBAL\n")
                    f.write("try:\n")
                    for library in to_preload:
                        f.write('    _{} = CDLL("{}", mode=RTLD_GLOBAL)\n'.format(library.split(".")[0], library))
                    f.write("except OSError:\n")
                    f.write("    import os\n")
                    f.write('    os.environ["ORT_CUDA_UNAVAILABLE"] = "1"\n')

        def _rewrite_ld_preload_tensorrt(self, to_preload):
            with open("onnxruntime/capi/_ld_preload.py", "a") as f:
                if len(to_preload) > 0:
                    f.write("from ctypes import CDLL, RTLD_GLOBAL\n")
                    f.write("try:\n")
                    for library in to_preload:
                        f.write('    _{} = CDLL("{}", mode=RTLD_GLOBAL)\n'.format(library.split(".")[0], library))
                    f.write("except OSError:\n")
                    f.write("    import os\n")
                    f.write('    os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"\n')

        def run(self):
            if is_manylinux:
                source = "onnxruntime/capi/onnxruntime_pybind11_state.so"
                dest = "onnxruntime/capi/onnxruntime_pybind11_state_manylinux1.so"
                logger.info("copying %s -> %s", source, dest)
                copyfile(source, dest)
                result = subprocess.run(
                    ["patchelf", "--print-needed", dest], check=True, stdout=subprocess.PIPE, universal_newlines=True
                )
                dependencies = [
                    "librccl.so",
                    "libamdhip64.so",
                    "librocblas.so",
                    "libMIOpen.so",
                    "libhsa-runtime64.so",
                    "libhsakmt.so",
                ]
                to_preload = []
                to_preload_cuda = []
                to_preload_tensorrt = []
                to_preload_cann = []
                cuda_dependencies = []
                args = ["patchelf", "--debug"]
                for line in result.stdout.split("\n"):
                    for dependency in dependencies:
                        if dependency in line:
                            to_preload.append(line)
                            args.extend(["--remove-needed", line])
                args.append(dest)
                if len(args) > 3:
                    subprocess.run(args, check=True, stdout=subprocess.PIPE)

                dest = "onnxruntime/capi/libonnxruntime_providers_" + ("rocm.so" if is_rocm else "cuda.so")
                if path.isfile(dest):
                    result = subprocess.run(
                        ["patchelf", "--print-needed", dest],
                        check=True,
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )
                    cuda_dependencies = [
                        "libcublas.so",
                        "libcublasLt.so",
                        "libcudnn.so",
                        "libcudart.so",
                        "libcurand.so",
                        "libcufft.so",
                        "libnvToolsExt.so",
                        "libcupti.so",
                    ]
                    rocm_dependencies = [
                        "librccl.so",
                        "libamdhip64.so",
                        "librocblas.so",
                        "libMIOpen.so",
                        "libhsa-runtime64.so",
                        "libhsakmt.so",
                    ]
                    args = ["patchelf", "--debug"]
                    for line in result.stdout.split("\n"):
                        for dependency in cuda_dependencies + rocm_dependencies:
                            if dependency in line:
                                if dependency not in to_preload:
                                    to_preload_cuda.append(line)
                                args.extend(["--remove-needed", line])
                    args.append(dest)
                    if len(args) > 3:
                        subprocess.run(args, check=True, stdout=subprocess.PIPE)

                dest = "onnxruntime/capi/libonnxruntime_providers_" + ("migraphx.so" if is_rocm else "tensorrt.so")
                if path.isfile(dest):
                    result = subprocess.run(
                        ["patchelf", "--print-needed", dest],
                        check=True,
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )
                    tensorrt_dependencies = ["libnvinfer.so", "libnvinfer_plugin.so", "libnvonnxparser.so"]
                    args = ["patchelf", "--debug"]
                    for line in result.stdout.split("\n"):
                        for dependency in cuda_dependencies + tensorrt_dependencies:
                            if dependency in line:
                                if dependency not in (to_preload + to_preload_cuda):
                                    to_preload_tensorrt.append(line)
                                args.extend(["--remove-needed", line])
                    args.append(dest)
                    if len(args) > 3:
                        subprocess.run(args, check=True, stdout=subprocess.PIPE)

                dest = "onnxruntime/capi/libonnxruntime_providers_cann.so"
                if path.isfile(dest):
                    result = subprocess.run(
                        ["patchelf", "--print-needed", dest],
                        check=True,
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )
                    cann_dependencies = ["libascendcl.so", "libacl_op_compiler.so", "libfmk_onnx_parser.so"]
                    args = ["patchelf", "--debug"]
                    for line in result.stdout.split("\n"):
                        for dependency in cann_dependencies:
                            if dependency in line:
                                if dependency not in to_preload:
                                    to_preload_cann.append(line)
                                args.extend(["--remove-needed", line])
                    args.append(dest)
                    if len(args) > 3:
                        subprocess.run(args, check=True, stdout=subprocess.PIPE)

                dest = "onnxruntime/capi/libonnxruntime_providers_openvino.so"
                if path.isfile(dest):
                    subprocess.run(
                        ["patchelf", "--set-rpath", "$ORIGIN", dest, "--force-rpath"],
                        check=True,
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )

                self._rewrite_ld_preload(to_preload)
                self._rewrite_ld_preload_cuda(to_preload_cuda)
                self._rewrite_ld_preload_tensorrt(to_preload_tensorrt)
                self._rewrite_ld_preload(to_preload_cann)
            _bdist_wheel.run(self)
            if is_manylinux and not disable_auditwheel_repair and not is_openvino:
                assert self.dist_dir is not None
                file = glob(path.join(self.dist_dir, "*linux*.whl"))[0]
                logger.info("repairing %s for manylinux1", file)
                try:
                    subprocess.run(
                        ["auditwheel", "repair", "-w", self.dist_dir, file], check=True, stdout=subprocess.PIPE
                    )
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


providers_cuda_or_rocm = "libonnxruntime_providers_" + ("rocm.so" if is_rocm else "cuda.so")
providers_tensorrt_or_migraphx = "libonnxruntime_providers_" + ("migraphx.so" if is_rocm else "tensorrt.so")
providers_openvino = "libonnxruntime_providers_openvino.so"
providers_cann = "libonnxruntime_providers_cann.so"

# Additional binaries
dl_libs = []
libs = []

if platform.system() == "Linux":
    libs = [
        "onnxruntime_pybind11_state.so",
        "libdnnl.so.2",
        "libmklml_intel.so",
        "libmklml_gnu.so",
        "libiomp5.so",
        "mimalloc.so",
    ]
    dl_libs = ["libonnxruntime_providers_shared.so"]
    dl_libs.append(providers_cuda_or_rocm)
    dl_libs.append(providers_tensorrt_or_migraphx)
    dl_libs.append(providers_cann)
    # DNNL, TensorRT & OpenVINO EPs are built as shared libs
    libs.extend(["libonnxruntime_providers_shared.so"])
    libs.extend(["libonnxruntime_providers_dnnl.so"])
    libs.extend(["libonnxruntime_providers_openvino.so"])
    libs.append(providers_cuda_or_rocm)
    libs.append(providers_tensorrt_or_migraphx)
    libs.append(providers_cann)
    if nightly_build:
        libs.extend(["libonnxruntime_pywrapper.so"])
elif platform.system() == "Darwin":
    libs = ["onnxruntime_pybind11_state.so", "libdnnl.2.dylib", "mimalloc.so"]  # TODO add libmklml and libiomp5 later.
    # DNNL & TensorRT EPs are built as shared libs
    libs.extend(["libonnxruntime_providers_shared.dylib"])
    libs.extend(["libonnxruntime_providers_dnnl.dylib"])
    libs.extend(["libonnxruntime_providers_tensorrt.dylib"])
    libs.extend(["libonnxruntime_providers_cuda.dylib"])
    if nightly_build:
        libs.extend(["libonnxruntime_pywrapper.dylib"])
else:
    libs = ["onnxruntime_pybind11_state.pyd", "dnnl.dll", "mklml.dll", "libiomp5md.dll"]
    # DNNL, TensorRT & OpenVINO EPs are built as shared libs
    libs.extend(["onnxruntime_providers_shared.dll"])
    libs.extend(["onnxruntime_providers_dnnl.dll"])
    libs.extend(["onnxruntime_providers_tensorrt.dll"])
    libs.extend(["onnxruntime_providers_openvino.dll"])
    libs.extend(["onnxruntime_providers_cuda.dll"])
    # DirectML Libs
    libs.extend(["DirectML.dll"])
    if nightly_build:
        libs.extend(["onnxruntime_pywrapper.dll"])

if is_manylinux:
    if is_openvino:
        ov_libs = [
            "libopenvino_intel_cpu_plugin.so",
            "libopenvino_intel_gpu_plugin.so",
            "libopenvino_intel_myriad_plugin.so",
            "libopenvino_auto_plugin.so",
            "libopenvino_hetero_plugin.so",
            "libtbb.so.2",
            "libtbbmalloc.so.2",
            "libopenvino.so",
            "libopenvino_c.so",
            "libopenvino_onnx_frontend.so",
        ]
        for x in ov_libs:
            y = "onnxruntime/capi/" + x
            subprocess.run(
                ["patchelf", "--set-rpath", "$ORIGIN", y, "--force-rpath"],
                check=True,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )
            dl_libs.append(x)
        dl_libs.append(providers_openvino)
        dl_libs.append("plugins.xml")
        dl_libs.append("usb-ma2x8x.mvcmd")
    data = ["capi/libonnxruntime_pywrapper.so"] if nightly_build else []
    data += [path.join("capi", x) for x in dl_libs if path.isfile(path.join("onnxruntime", "capi", x))]
    ext_modules = [
        Extension(
            "onnxruntime.capi.onnxruntime_pybind11_state",
            ["onnxruntime/capi/onnxruntime_pybind11_state_manylinux1.so"],
        ),
    ]
else:
    data = [path.join("capi", x) for x in libs if path.isfile(path.join("onnxruntime", "capi", x))]
    ext_modules = []

# Additional examples
examples_names = ["mul_1.onnx", "logreg_iris.onnx", "sigmoid.onnx"]
examples = [path.join("datasets", x) for x in examples_names]

# Extra files such as EULA and ThirdPartyNotices
extra = ["LICENSE", "ThirdPartyNotices.txt", "Privacy.md"]

# Description
readme_file = "docs/python/ReadMeOV.rst" if is_openvino else "docs/python/README.rst"
README = path.join(getcwd(), readme_file)
if not path.exists(README):
    this = path.dirname(__file__)
    README = path.join(this, readme_file)

if not path.exists(README):
    raise FileNotFoundError("Unable to find 'README.rst'")
with open(README) as f:
    long_description = f.read()

# Include files in onnxruntime/external if --enable_external_custom_op_schemas build.sh command
# line option is specified.
# If the options is not specified this following condition fails as onnxruntime/external folder is not created in the
# build flow under the build binary directory.
if path.isdir(path.join("onnxruntime", "external")):
    # Gather all files under onnxruntime/external directory.
    extra.extend(
        list(
            str(Path(*Path(x).parts[1:]))
            for x in list(iglob(path.join(path.join("onnxruntime", "external"), "**/*.*"), recursive=True))
        )
    )

packages = [
    "onnxruntime",
    "onnxruntime.backend",
    "onnxruntime.capi",
    "onnxruntime.capi.training",
    "onnxruntime.datasets",
    "onnxruntime.tools",
    "onnxruntime.tools.mobile_helpers",
    "onnxruntime.tools.ort_format_model",
    "onnxruntime.tools.ort_format_model.ort_flatbuffers_py",
    "onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs",
    "onnxruntime.tools.qdq_helpers",
    "onnxruntime.quantization",
    "onnxruntime.quantization.operators",
    "onnxruntime.quantization.CalTableFlatBuffers",
    "onnxruntime.transformers",
    "onnxruntime.transformers.models.gpt2",
    "onnxruntime.transformers.models.longformer",
    "onnxruntime.transformers.models.t5",
]

package_data = {"onnxruntime.tools.mobile_helpers": ["*.md", "*.config"]}
data_files = []

requirements_file = "requirements.txt"

local_version = None
enable_training = parse_arg_remove_boolean(sys.argv, "--enable_training")
enable_training_on_device = parse_arg_remove_boolean(sys.argv, "--enable_training_on_device")
enable_rocm_profiling = parse_arg_remove_boolean(sys.argv, "--enable_rocm_profiling")
disable_auditwheel_repair = parse_arg_remove_boolean(sys.argv, "--disable_auditwheel_repair")
default_training_package_device = parse_arg_remove_boolean(sys.argv, "--default_training_package_device")

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

if not enable_training:
    classifiers.extend(["Operating System :: Microsoft :: Windows", "Operating System :: MacOS"])

if enable_training:
    packages.extend(
        [
            "onnxruntime.training",
            "onnxruntime.training.amp",
            "onnxruntime.training.experimental",
            "onnxruntime.training.experimental.gradient_graph",
            "onnxruntime.training.optim",
            "onnxruntime.training.torchdynamo",
            "onnxruntime.training.ortmodule",
            "onnxruntime.training.ortmodule.experimental",
            "onnxruntime.training.ortmodule.experimental.json_config",
            "onnxruntime.training.ortmodule.experimental.hierarchical_ortmodule",
            "onnxruntime.training.ortmodule.torch_cpp_extensions",
            "onnxruntime.training.ortmodule.torch_cpp_extensions.cpu.aten_op_executor",
            "onnxruntime.training.ortmodule.torch_cpp_extensions.cpu.torch_interop_utils",
            "onnxruntime.training.ortmodule.torch_cpp_extensions.cuda.torch_gpu_allocator",
            "onnxruntime.training.ortmodule.torch_cpp_extensions.cuda.fused_ops",
            "onnxruntime.training.utils.data",
        ]
    )
    if enable_training_on_device:
        packages.append("onnxruntime.training.api")
        packages.append("onnxruntime.training.onnxblock")
        packages.append("onnxruntime.training.onnxblock.loss")
        packages.append("onnxruntime.training.onnxblock.optim")
    package_data["onnxruntime.training.ortmodule.torch_cpp_extensions.cpu.aten_op_executor"] = ["*.cc"]
    package_data["onnxruntime.training.ortmodule.torch_cpp_extensions.cpu.torch_interop_utils"] = ["*.cc"]
    package_data["onnxruntime.training.ortmodule.torch_cpp_extensions.cuda.torch_gpu_allocator"] = ["*.cc"]
    package_data["onnxruntime.training.ortmodule.torch_cpp_extensions.cuda.fused_ops"] = [
        "*.cpp",
        "*.cu",
        "*.cuh",
        "*.h",
    ]
    requirements_file = "requirements-training.txt"
    # with training, we want to follow this naming convention:
    # stable:
    # onnxruntime-training-1.7.0+cu111-cp36-cp36m-linux_x86_64.whl
    # nightly:
    # onnxruntime-training-1.7.0.dev20210408+cu111-cp36-cp36m-linux_x86_64.whl
    # this is needed immediately by pytorch/ort so that the user is able to
    # install an onnxruntime training package with matching torch cuda version.
    if not is_openvino:
        # To support the package consisting of both openvino and training modules part of it
        package_name = "onnxruntime-training"

    disable_local_version = environ.get("ORT_DISABLE_PYTHON_PACKAGE_LOCAL_VERSION", "0")
    disable_local_version = (
        disable_local_version == "1"
        or disable_local_version.lower() == "true"
        or disable_local_version.lower() == "yes"
    )
    # local version should be disabled for internal feeds.
    if not disable_local_version:
        # we want put default training packages to pypi. pypi does not accept package with a local version.
        if not default_training_package_device or nightly_build:
            if cuda_version:
                # removing '.' to make Cuda version number in the same form as Pytorch.
                local_version = "+cu" + cuda_version.replace(".", "")
            elif rocm_version:
                # removing '.' to make Rocm version number in the same form as Pytorch.
                local_version = "+rocm" + rocm_version.replace(".", "")
            else:
                # cpu version for documentation
                local_version = "+cpu"

if package_name == "onnxruntime-tvm":
    packages += ["onnxruntime.providers.tvm"]

package_data["onnxruntime"] = data + examples + extra

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
            except:  # noqa
                return False

        def reformat_run_count(count_str):
            try:
                count = int(count_str)
                if count >= 0 and count < 1000:
                    return "{:03}".format(count)
                elif count >= 1000:
                    raise RuntimeError(f"Too many builds for the same day: {count}")
                return ""
            except:  # noqa
                return ""

        build_suffix_is_date_format = check_date_format(build_suffix[:8])
        build_suffix_run_count = reformat_run_count(build_suffix[8:])
        if build_suffix_is_date_format and build_suffix_run_count:
            build_suffix = build_suffix[:8] + build_suffix_run_count
    elif len(build_suffix) >= 12:
        raise RuntimeError(f'Incorrect build suffix: "{build_suffix}"')

    if enable_training:
        from packaging import version
        from packaging.version import Version

        # with training package, we need to bump up version minor number so that
        # nightly releases take precedence over the latest release when --pre is used during pip install.
        # eventually this shall be the behavior of all onnxruntime releases.
        # alternatively we may bump up version number right after every release.
        ort_version = version.parse(version_number)
        if isinstance(ort_version, Version):
            # TODO: this is the last time we have to do this!!!
            # We shall bump up release number right after release cut.
            if ort_version.major == 1 and ort_version.minor == 8 and ort_version.micro == 0:
                version_number = "{major}.{minor}.{macro}".format(
                    major=ort_version.major, minor=ort_version.minor + 1, macro=ort_version.micro
                )

    version_number = version_number + ".dev" + build_suffix

if local_version:
    version_number = version_number + local_version
    if is_rocm and enable_rocm_profiling:
        version_number = version_number + ".profiling"

if wheel_name_suffix:
    if not (enable_training and wheel_name_suffix == "gpu"):
        # for training packages, local version is used to indicate device types
        package_name = "{}-{}".format(package_name, wheel_name_suffix)

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


if enable_training:

    def save_build_and_package_info(package_name, version_number, cuda_version, rocm_version):
        sys.path.append(path.join(path.dirname(__file__), "onnxruntime", "python"))
        from onnxruntime_collect_build_info import find_cudart_versions

        version_path = path.join("onnxruntime", "capi", "build_and_package_info.py")
        with open(version_path, "w") as f:
            f.write("package_name = '{}'\n".format(package_name))
            f.write("__version__ = '{}'\n".format(version_number))

            if cuda_version:
                f.write("cuda_version = '{}'\n".format(cuda_version))

                # cudart_versions are integers
                cudart_versions = find_cudart_versions(build_env=True)
                if cudart_versions and len(cudart_versions) == 1:
                    f.write("cudart_version = {}\n".format(cudart_versions[0]))
                else:
                    print(
                        "Error getting cudart version. ",
                        "did not find any cudart library"
                        if not cudart_versions or len(cudart_versions) == 0
                        else "found multiple cudart libraries",
                    )
            elif rocm_version:
                f.write("rocm_version = '{}'\n".format(rocm_version))

    save_build_and_package_info(package_name, version_number, cuda_version, rocm_version)

# Setup
setup(
    name=package_name,
    version=version_number,
    description="ONNX Runtime is a runtime accelerator for Machine Learning models",
    long_description=long_description,
    author="Microsoft Corporation",
    author_email="onnxruntime@microsoft.com",
    cmdclass=cmd_classes,
    license="MIT License",
    packages=packages,
    ext_modules=ext_modules,
    package_data=package_data,
    url="https://onnxruntime.ai",
    download_url="https://github.com/microsoft/onnxruntime/tags",
    data_files=data_files,
    install_requires=install_requires,
    keywords="onnx machine learning",
    entry_points={
        "console_scripts": [
            "onnxruntime_test = onnxruntime.tools.onnxruntime_test:main",
        ]
    },
    classifiers=classifiers,
)
