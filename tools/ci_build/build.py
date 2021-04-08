#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import hashlib
from logger import log


class BaseError(Exception):
    """Base class for errors originating from build.py."""
    pass


class BuildError(BaseError):
    """Error from running build steps."""

    def __init__(self, *messages):
        super().__init__("\n".join(messages))


class UsageError(BaseError):
    """Usage related error."""

    def __init__(self, message):
        super().__init__(message)


def _check_python_version():
    # According to the BUILD.md, python 3.5+ is required:
    # Python 2 is definitely not supported and it should be safer to consider
    # it won't run with python 4:
    if sys.version_info[0] != 3:
        raise BuildError(
            "Bad python major version: expecting python 3, found version "
            "'{}'".format(sys.version))
    if sys.version_info[1] < 5:
        raise BuildError(
            "Bad python minor version: expecting python 3.5+, found version "
            "'{}'".format(sys.version))


_check_python_version()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNXRuntime CI build driver.",
        usage="""  # noqa
        Default behavior is --update --build --test for native architecture builds.
        Default behavior is --update --build for cross-compiled builds.

        The Update phase will update git submodules, and run cmake to generate makefiles.
        The Build phase will build all projects.
        The Test phase will run all unit tests, and optionally the ONNX tests.

        Use the individual flags to only run the specified stages.
        """)
    # Main arguments
    parser.add_argument(
        "--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument(
        "--config", nargs="+", default=["Debug"],
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build.")
    parser.add_argument(
        "--update", action='store_true', help="Update makefiles.")
    parser.add_argument("--build", action='store_true', help="Build.")
    parser.add_argument(
        "--clean", action='store_true',
        help="Run 'cmake --build --target clean' for the selected config/s.")
    parser.add_argument(
        "--parallel", action='store_true', help="""Use parallel build.
    The build setup doesn't get all dependencies right, so --parallel
    only works if you're just rebuilding ONNXRuntime code. If you've
    done an update that fetched external dependencies you have to build
    without --parallel the first time. Once that's done , run with
    "--build --parallel --test" to just build in
    parallel and run tests.""")
    parser.add_argument("--test", action='store_true', help="Run unit tests.")
    parser.add_argument(
        "--skip_tests", action='store_true', help="Skip all tests.")

    # Training options
    parser.add_argument(
        "--enable_nvtx_profile", action='store_true', help="Enable NVTX profile in ORT.")
    parser.add_argument(
        "--enable_training", action='store_true', help="Enable training in ORT.")
    parser.add_argument(
        "--enable_training_python_frontend_e2e_tests", action="store_true",
        help="Enable the pytorch frontend training tests.")
    parser.add_argument(
        "--use_horovod", action='store_true', help="Enable Horovod.")
    parser.add_argument(
        "--mpi_home", help="Path to MPI installation dir")
    parser.add_argument(
        "--nccl_home", help="Path to NCCL installation dir")
    parser.add_argument(
        "--use_torch", action='store_true', help="Build with libtorch C++ APIs")
    parser.add_argument(
        "--torch_home", help="Path to Pytorch python package or libtorch dir")

    # enable ONNX tests
    parser.add_argument(
        "--enable_onnx_tests", action='store_true',
        help="""When running the Test phase, run onnx_test_running against
        available test data directories.""")
    parser.add_argument("--path_to_protoc_exe", help="Path to protoc exe.")
    parser.add_argument(
        "--fuzz_testing", action='store_true', help="Enable Fuzz testing of the onnxruntime.")

    # generate documentaiton
    parser.add_argument(
        "--gen_doc", action='store_true',
        help="Generate documentation on contrib ops")

    # CUDA related
    parser.add_argument("--use_cuda", action='store_true', help="Enable CUDA.")
    parser.add_argument(
        "--cuda_version", help="The version of CUDA toolkit to use. "
        "Auto-detect if not specified. e.g. 9.0")
    parser.add_argument(
        "--cuda_home", help="Path to CUDA home."
        "Read from CUDA_HOME environment variable if --use_cuda is true and "
        "--cuda_home is not specified.")
    parser.add_argument(
        "--cudnn_home", help="Path to CUDNN home. "
        "Read from CUDNN_HOME environment variable if --use_cuda is true and "
        "--cudnn_home is not specified.")

    # Python bindings
    parser.add_argument(
        "--enable_pybind", action='store_true', help="Enable Python Bindings.")
    parser.add_argument(
        "--build_wheel", action='store_true', help="Build Python Wheel.")
    parser.add_argument(
        "--wheel_name_suffix", help="Suffix to append to created wheel names. "
        "This value is currently only used for nightly builds.")
    parser.add_argument(
        "--numpy_version", help="Installs a specific version of numpy "
        "before building the python binding.")
    parser.add_argument(
        "--skip-keras-test", action='store_true',
        help="Skip tests with Keras if keras is installed")

    # C-Sharp bindings
    parser.add_argument(
        "--build_csharp", action='store_true',
        help="Build C#.Net DLL and NuGet package. This should be only used in CI pipelines. "
        "For building C# bindings and packaging them into nuget package use --build_nuget arg.")

    parser.add_argument(
        "--build_nuget", action='store_true',
        help="Build C#.Net DLL and NuGet package on the local machine. "
        "Currently only Windows and Linux platforms are supported.")

    # Java bindings
    parser.add_argument(
        "--build_java", action='store_true', help="Build Java bindings.")

    # Node.js binding
    parser.add_argument(
        "--build_nodejs", action='store_true',
        help="Build Node.js binding and NPM package.")

    # Build a shared lib
    parser.add_argument(
        "--build_shared_lib", action='store_true',
        help="Build a shared library for the ONNXRuntime.")

    # Build options
    parser.add_argument(
        "--cmake_extra_defines", nargs="+",
        help="Extra definitions to pass to CMake during build system "
        "generation. These are just CMake -D options without the leading -D.")
    parser.add_argument(
        "--target",
        help="Build a specific target, e.g. winml_dll")
    parser.add_argument(
        "--x86", action='store_true',
        help="Create x86 makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed")
    parser.add_argument(
        "--arm", action='store_true',
        help="Create ARM makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed")
    parser.add_argument(
        "--arm64", action='store_true',
        help="Create ARM64 makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed")
    parser.add_argument(
        "--msvc_toolset", help="MSVC toolset to use. e.g. 14.11")
    parser.add_argument("--android", action='store_true', help='Build for Android')
    parser.add_argument("--android_abi", type=str, default='arm64-v8a', help='')
    parser.add_argument("--android_api", type=int, default=27, help='Android API Level, e.g. 21')
    parser.add_argument("--android_sdk_path", type=str, help='Path to the Android SDK')
    parser.add_argument("--android_ndk_path", default="", help="Path to the Android NDK")
    parser.add_argument("--android_cpp_shared", action="store_true",
                        help="Build with shared libc++ instead of the default static libc++.")

    parser.add_argument("--ios", action='store_true', help="build for ios")
    parser.add_argument(
        "--ios_sysroot", default="",
        help="Specify the location name of the macOS platform SDK to be used")
    parser.add_argument(
        "--ios_toolchain_dir", default="",
        help="Path to ios toolchain binaries")
    parser.add_argument(
        "--ios_toolchain_file", default="",
        help="Path to ios toolchain file, "
        "or cmake/onnxruntime_ios.toolchain.cmake will be used")
    parser.add_argument(
        "--use_xcode", action='store_true',
        help="Use Xcode as cmake generator, this is only supported on MacOS.")
    parser.add_argument(
        "--osx_arch", type=str,
        help="Specify the Target specific architectures for macOS and iOS"
        "This is only supported on MacOS")
    parser.add_argument(
        "--apple_deploy_target", type=str,
        help="Specify the minimum version of the target platform "
        "(e.g. macOS or iOS)"
        "This is only supported on MacOS")

    # Arguments needed by CI
    parser.add_argument(
        "--cmake_path", default="cmake", help="Path to the CMake program.")
    parser.add_argument(
        "--ctest_path", default="ctest", help="Path to the CTest program.")
    parser.add_argument(
        "--skip_submodule_sync", action='store_true', help="Don't do a "
        "'git submodule update'. Makes the Update phase faster.")
    parser.add_argument(
        "--use_vstest", action='store_true',
        help="Use use_vstest for running unitests.")
    parser.add_argument(
        "--use_jemalloc", action='store_true', help="Use jemalloc.")
    parser.add_argument(
        "--use_mimalloc", default=['none'],
        choices=['none', 'stl', 'arena', 'all'], help="Use mimalloc.")
    parser.add_argument(
        "--use_openblas", action='store_true', help="Build with OpenBLAS.")
    parser.add_argument(
        "--use_dnnl", action='store_true', help="Build with DNNL.")
    parser.add_argument(
        "--use_mklml", action='store_true', help="Build with MKLML.")
    parser.add_argument(
        "--use_featurizers", action='store_true',
        help="Build with ML Featurizer support.")
    parser.add_argument(
        "--use_ngraph", action='store_true', help="Build with nGraph.")
    parser.add_argument(
        "--use_openvino", nargs="?", const="CPU_FP32",
        choices=["CPU_FP32", "GPU_FP32", "GPU_FP16", "VAD-M_FP16",
                 "MYRIAD_FP16", "VAD-F_FP32"],
        help="Build with OpenVINO for specific hardware.")
    parser.add_argument(
        "--use_nnapi", action='store_true', help="Build with NNAPI support.")
    parser.add_argument(
        "--use_rknpu", action='store_true', help="Build with RKNPU.")
    parser.add_argument(
        "--use_preinstalled_eigen", action='store_true',
        help="Use pre-installed Eigen.")
    parser.add_argument("--eigen_path", help="Path to pre-installed Eigen.")
    parser.add_argument(
        "--use_openmp", action='store_true', help="Build with OpenMP")
    parser.add_argument(
        "--enable_msinternal", action="store_true",
        help="Enable for Microsoft internal builds only.")
    parser.add_argument("--llvm_path", help="Path to llvm dir")
    parser.add_argument(
        "--use_vitisai", action='store_true', help="Build with Vitis-AI")
    parser.add_argument(
        "--use_nuphar", action='store_true', help="Build with nuphar")
    parser.add_argument(
        "--use_tensorrt", action='store_true', help="Build with TensorRT")
    parser.add_argument(
        "--tensorrt_home", help="Path to TensorRT installation dir")
    parser.add_argument(
        "--use_migraphx", action='store_true', help="Build with MIGraphX")
    parser.add_argument(
        "--migraphx_home", help="Path to MIGraphX installation dir")
    parser.add_argument(
        "--use_full_protobuf", action='store_true',
        help="Use the full protobuf library")

    parser.add_argument(
        "--skip_onnx_tests", action='store_true', help="Explicitly disable "
        "all onnx related tests. Note: Use --skip_tests to skip all tests.")
    parser.add_argument(
        "--skip_winml_tests", action='store_true',
        help="Explicitly disable all WinML related tests")
    parser.add_argument(
        "--skip_nodejs_tests", action='store_true',
        help="Explicitly disable all Node.js binding tests")
    parser.add_argument(
        "--enable_msvc_static_runtime", action='store_true',
        help="Enable static linking of MSVC runtimes.")
    parser.add_argument(
        "--enable_language_interop_ops", action='store_true',
        help="Enable operator implemented in language other than cpp")
    parser.add_argument(
        "--cmake_generator",
        choices=['Visual Studio 15 2017', 'Visual Studio 16 2019', 'Ninja'],
        default='Visual Studio 15 2017' if is_windows() else None,
        help="Specify the generator that CMake invokes. "
        "This is only supported on Windows")
    parser.add_argument(
        "--enable_multi_device_test", action='store_true',
        help="Test with multi-device. Mostly used for multi-device GPU")
    parser.add_argument(
        "--use_dml", action='store_true', help="Build with DirectML.")
    parser.add_argument(
        "--use_winml", action='store_true', help="Build with WinML.")
    parser.add_argument(
        "--winml_root_namespace_override", type=str,
        help="Specify the namespace that WinML builds into.")
    parser.add_argument(
        "--use_telemetry", action='store_true',
        help="Only official builds can set this flag to enable telemetry.")
    parser.add_argument(
        "--enable_wcos", action='store_true',
        help="Build for Windows Core OS.")
    parser.add_argument(
        "--enable_windows_store", action='store_true',
        help="Build for Windows Store")
    parser.add_argument(
        "--enable_lto", action='store_true',
        help="Enable Link Time Optimization")
    parser.add_argument(
        "--use_acl", nargs="?", const="ACL_1905",
        choices=["ACL_1902", "ACL_1905", "ACL_1908", "ACL_2002"],
        help="Build with ACL for ARM architectures.")
    parser.add_argument(
        "--use_armnn", action='store_true',
        help="Enable ArmNN Execution Provider.")
    parser.add_argument(
        "--armnn_relu", action='store_true',
        help="Use the Relu operator implementation from the ArmNN EP.")
    parser.add_argument(
        "--armnn_bn", action='store_true',
        help="Use the Batch Normalization operator implementation from the ArmNN EP.")
    parser.add_argument(
        "--build_micro_benchmarks", action='store_true',
        help="Build ONNXRuntime micro-benchmarks.")

    # options to reduce binary size
    parser.add_argument("--minimal_build", action='store_true',
                        help="Create a build that only supports ORT format models. "
                        "See /docs/ONNX_Runtime_Format_Model_Usage.md for more information. "
                        "RTTI is automatically disabled in a minimal build.")
    parser.add_argument("--include_ops_by_model", type=str, help="include ops from model(s) under designated path.")
    parser.add_argument("--include_ops_by_config", type=str,
                        help="include ops from config file. "
                        "See /docs/Reduced_Operator_Kernel_build.md for more information.")

    parser.add_argument("--disable_contrib_ops", action='store_true',
                        help="Disable contrib ops (reduces binary size)")
    parser.add_argument("--disable_ml_ops", action='store_true',
                        help="Disable traditional ML ops (reduces binary size)")
    parser.add_argument("--disable_rtti", action='store_true', help="Disable RTTI (reduces binary size)")
    parser.add_argument("--disable_exceptions", action='store_true',
                        help="Disable exceptions to reduce binary size. Requires --minimal_build.")
    parser.add_argument("--disable_ort_format_load", action='store_true',
                        help='Disable support for loading ORT format models in a non-minimal build.')

    return parser.parse_args()


def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    executable_path = shutil.which(command_or_path)
    if executable_path is None:
        raise BuildError("Failed to resolve executable path for "
                         "'{}'.".format(command_or_path))
    return os.path.realpath(executable_path)


def is_windows():
    return sys.platform.startswith("win")


def is_macOS():
    return sys.platform.startswith("darwin")


def is_linux():
    return sys.platform.startswith("linux")


def get_linux_distro():
    try:
        with open('/etc/os-release', 'r') as f:
            dist_info = dict(
                line.strip().split('=', 1) for line in f.readlines())
        return dist_info.get('NAME', '').strip('"'), dist_info.get(
            'VERSION', '').strip('"')
    except (IOError, ValueError):
        return '', ''


def is_ubuntu_1604():
    dist, ver = get_linux_distro()
    return dist == 'Ubuntu' and ver.startswith('16.04')


def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)


def run_subprocess(args, cwd=None, capture=False, dll_path=None,
                   shell=False, env={}):
    log.debug("Running subprocess in '{0}'\n{1}".format(
        cwd or os.getcwd(), args))
    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path

    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (
        None, None)
    my_env.update(env)
    completed_process = subprocess.run(
        args, cwd=cwd, check=True, stdout=stdout, stderr=stderr,
        env=my_env, shell=shell)
    log.debug("Subprocess completed. Return code=" +
              str(completed_process.returncode))
    return completed_process


def update_submodules(source_dir):
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=source_dir)
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"],
                   cwd=source_dir)


def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )


def is_sudo():
    return 'SUDO_UID' in os.environ.keys()


def install_apt_package(package):
    have = package in str(run_subprocess(
        ["apt", "list", "--installed", package], capture=True).stdout)
    if not have:
        if is_sudo():
            run_subprocess(['apt-get', 'install', '-y', package])
        else:
            raise BuildError(package + " APT package missing. Please re-run "
                             "this script using sudo to install.")


def install_ubuntu_deps(args):
    """Check if the necessary Ubuntu dependencies are installed.
    Not required on docker. Provide help output if missing."""

    # check we need the packages first
    if not (args.enable_pybind or args.use_openblas):
        return

    # not needed on docker as packages are pre-installed
    if not is_docker():
        try:
            if args.enable_pybind:
                install_apt_package("python3")

            if args.use_openblas:
                install_apt_package("libopenblas-dev")

        except Exception as e:
            raise BuildError("Error setting up required APT packages. "
                             "{}".format(str(e)))


def install_python_deps(numpy_version=""):
    dep_packages = ['setuptools', 'wheel', 'pytest']
    dep_packages.append('numpy=={}'.format(numpy_version) if numpy_version
                        else 'numpy>=1.16.6')
    dep_packages.append('sympy>=1.1')
    dep_packages.append('packaging')
    dep_packages.append('cerberus')
    run_subprocess([sys.executable, '-m', 'pip', 'install', '--trusted-host',
                    'files.pythonhosted.org'] + dep_packages)


# We need to install Torch to test certain functionalities of the ORT Python package
def install_torch():
    # Command works for both Windows
    run_subprocess([sys.executable, '-m', 'pip', 'install', '--trusted-host',
                    'files.pythonhosted.org', 'torch===1.5.1+cu101', 'torchvision===0.6.1+cu101',
                    '-f', 'https://download.pytorch.org/whl/torch_stable.html'])


def check_md5(filename, expected_md5):
    if not os.path.exists(filename):
        return False
    hash_md5 = hashlib.md5()
    BLOCKSIZE = 1024*64
    with open(filename, "rb") as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hash_md5.update(buf)
            buf = f.read(BLOCKSIZE)
    hex = hash_md5.hexdigest()
    if hex != expected_md5:
        log.info('md5 mismatch, expect %s, got %s' % (expected_md5, hex))
        os.remove(filename)
        return False
    return True


def setup_test_data(build_dir, configs):
    # create a shortcut for test models if there is a 'models'
    # folder in build_dir
    if is_windows():
        src_model_dir = os.path.join(build_dir, 'models')
        if os.path.exists('C:\\local\\models') and not os.path.exists(
                src_model_dir):
            log.debug("creating shortcut %s -> %s" % (
                'C:\\local\\models', src_model_dir))
            run_subprocess(['mklink', '/D', '/J', src_model_dir,
                            'C:\\local\\models'], shell=True)
        for config in configs:
            config_build_dir = get_config_build_dir(build_dir, config)
            os.makedirs(config_build_dir, exist_ok=True)
            dest_model_dir = os.path.join(config_build_dir, 'models')
            if os.path.exists('C:\\local\\models') and not os.path.exists(
                    dest_model_dir):
                log.debug("creating shortcut %s -> %s" % (
                    'C:\\local\\models', dest_model_dir))
                run_subprocess(['mklink', '/D', '/J', dest_model_dir,
                                'C:\\local\\models'], shell=True)
            elif os.path.exists(src_model_dir) and not os.path.exists(
                    dest_model_dir):
                log.debug("creating shortcut %s -> %s" % (
                    src_model_dir, dest_model_dir))
                run_subprocess(['mklink', '/D', '/J', dest_model_dir,
                                src_model_dir], shell=True)


def generate_build_tree(cmake_path, source_dir, build_dir, cuda_home, cudnn_home,
                        mpi_home, nccl_home, tensorrt_home, migraphx_home, torch_home,
                        path_to_protoc_exe, configs, cmake_extra_defines, args, cmake_extra_args):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    # TODO: fix jemalloc build so it does not conflict with onnxruntime
    # shared lib builds. (e.g. onnxuntime_pybind)
    # for now, disable jemalloc if pybind is also enabled.
    cmake_args = [
        cmake_path, cmake_dir,
        "-Donnxruntime_RUN_ONNX_TESTS=" + (
            "ON" if args.enable_onnx_tests else "OFF"),
        "-Donnxruntime_BUILD_WINML_TESTS=" + (
            "OFF" if args.skip_winml_tests else "ON"),
        "-Donnxruntime_GENERATE_TEST_REPORTS=ON",
        "-Donnxruntime_DEV_MODE=" + (
            "OFF" if args.use_acl or args.use_armnn or args.use_winml or
            (args.ios and is_macOS()) else "ON"),
        "-DPYTHON_EXECUTABLE=" + sys.executable,
        "-Donnxruntime_USE_CUDA=" + ("ON" if args.use_cuda else "OFF"),
        "-Donnxruntime_CUDNN_HOME=" + (cudnn_home if args.use_cuda else ""),
        "-Donnxruntime_USE_FEATURIZERS=" + (
            "ON" if args.use_featurizers else "OFF"),
        "-Donnxruntime_CUDA_HOME=" + (cuda_home if args.use_cuda else ""),
        "-Donnxruntime_USE_JEMALLOC=" + ("ON" if args.use_jemalloc else "OFF"),
        "-Donnxruntime_USE_MIMALLOC_STL_ALLOCATOR=" + (
            "ON" if args.use_mimalloc == "stl" or
            args.use_mimalloc == "all" else "OFF"),
        "-Donnxruntime_USE_MIMALLOC_ARENA_ALLOCATOR=" + (
            "ON" if args.use_mimalloc == "arena" or
            args.use_mimalloc == "all" else "OFF"),
        "-Donnxruntime_ENABLE_PYTHON=" + (
            "ON" if args.enable_pybind else "OFF"),
        "-Donnxruntime_BUILD_CSHARP=" + ("ON" if args.build_csharp else "OFF"),
        "-Donnxruntime_BUILD_JAVA=" + ("ON" if args.build_java else "OFF"),
        "-Donnxruntime_BUILD_NODEJS=" + ("ON" if args.build_nodejs else "OFF"),
        "-Donnxruntime_BUILD_SHARED_LIB=" + (
            "ON" if args.build_shared_lib else "OFF"),
        "-Donnxruntime_USE_EIGEN_FOR_BLAS=" + (
            "OFF" if args.use_openblas else "ON"),
        "-Donnxruntime_USE_OPENBLAS=" + ("ON" if args.use_openblas else "OFF"),
        "-Donnxruntime_USE_DNNL=" + ("ON" if args.use_dnnl else "OFF"),
        "-Donnxruntime_USE_MKLML=" + ("ON" if args.use_mklml else "OFF"),
        "-Donnxruntime_USE_NGRAPH=" + ("ON" if args.use_ngraph else "OFF"),
        "-Donnxruntime_USE_NNAPI_BUILTIN=" + ("ON" if args.use_nnapi else "OFF"),
        "-Donnxruntime_USE_RKNPU=" + ("ON" if args.use_rknpu else "OFF"),
        "-Donnxruntime_USE_OPENMP=" + (
            "ON" if args.use_openmp and not (
                args.use_nnapi or (args.use_mklml and (is_macOS() or is_windows())) or args.use_ngraph or
                args.android or (args.ios and is_macOS())
                or args.use_rknpu)
            else "OFF"),
        "-Donnxruntime_USE_TVM=" + ("ON" if args.use_nuphar else "OFF"),
        "-Donnxruntime_USE_LLVM=" + ("ON" if args.use_nuphar else "OFF"),
        "-Donnxruntime_ENABLE_MICROSOFT_INTERNAL=" + (
            "ON" if args.enable_msinternal else "OFF"),
        "-Donnxruntime_USE_VITISAI=" + ("ON" if args.use_vitisai else "OFF"),
        "-Donnxruntime_USE_NUPHAR=" + ("ON" if args.use_nuphar else "OFF"),
        "-Donnxruntime_USE_TENSORRT=" + ("ON" if args.use_tensorrt else "OFF"),
        "-Donnxruntime_TENSORRT_HOME=" + (
            tensorrt_home if args.use_tensorrt else ""),
        # set vars for migraphx
        "-Donnxruntime_USE_MIGRAPHX=" + ("ON" if args.use_migraphx else "OFF"),
        "-Donnxruntime_MIGRAPHX_HOME=" + (migraphx_home if args.use_migraphx else ""),
        # By default - we currently support only cross compiling for
        # ARM/ARM64 (no native compilation supported through this
        # script).
        "-Donnxruntime_CROSS_COMPILING=" + (
            "ON" if args.arm64 or args.arm else "OFF"),
        "-Donnxruntime_DISABLE_CONTRIB_OPS=" + ("ON" if args.disable_contrib_ops else "OFF"),
        "-Donnxruntime_DISABLE_ML_OPS=" + ("ON" if args.disable_ml_ops else "OFF"),
        "-Donnxruntime_DISABLE_RTTI=" + ("ON" if args.disable_rtti else "OFF"),
        "-Donnxruntime_DISABLE_EXCEPTIONS=" + ("ON" if args.disable_exceptions else "OFF"),
        "-Donnxruntime_DISABLE_ORT_FORMAT_LOAD=" + ("ON" if args.disable_ort_format_load else "OFF"),
        "-Donnxruntime_MINIMAL_BUILD=" + ("ON" if args.minimal_build else "OFF"),
        "-Donnxruntime_REDUCED_OPS_BUILD=" + (
            "ON" if args.include_ops_by_config or args.include_ops_by_model else "OFF"),
        "-Donnxruntime_MSVC_STATIC_RUNTIME=" + (
            "ON" if args.enable_msvc_static_runtime else "OFF"),
        # enable pyop if it is nightly build
        "-Donnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=" + (
            "ON" if args.enable_language_interop_ops else "OFF"),
        "-Donnxruntime_USE_DML=" + ("ON" if args.use_dml else "OFF"),
        "-Donnxruntime_USE_WINML=" + ("ON" if args.use_winml else "OFF"),
        "-Donnxruntime_USE_TELEMETRY=" + (
            "ON" if args.use_telemetry else "OFF"),
        "-Donnxruntime_ENABLE_LTO=" + ("ON" if args.enable_lto else "OFF"),
        "-Donnxruntime_USE_ACL=" + ("ON" if args.use_acl else "OFF"),
        "-Donnxruntime_USE_ACL_1902=" + (
            "ON" if args.use_acl == "ACL_1902" else "OFF"),
        "-Donnxruntime_USE_ACL_1905=" + (
            "ON" if args.use_acl == "ACL_1905" else "OFF"),
        "-Donnxruntime_USE_ACL_1908=" + (
            "ON" if args.use_acl == "ACL_1908" else "OFF"),
        "-Donnxruntime_USE_ACL_2002=" + (
            "ON" if args.use_acl == "ACL_2002" else "OFF"),
        "-Donnxruntime_USE_ARMNN=" + (
            "ON" if args.use_armnn else "OFF"),
        "-Donnxruntime_ARMNN_RELU_USE_CPU=" + (
            "OFF" if args.armnn_relu else "ON"),
        "-Donnxruntime_ARMNN_BN_USE_CPU=" + (
            "OFF" if args.armnn_bn else "ON"),
        # Training related flags
        "-Donnxruntime_ENABLE_NVTX_PROFILE=" + (
            "ON" if args.enable_nvtx_profile else "OFF"),
        "-Donnxruntime_ENABLE_TRAINING=" + (
            "ON" if args.enable_training else "OFF"),
        "-Donnxruntime_USE_HOROVOD=" + (
            "ON" if args.use_horovod else "OFF"),
        "-Donnxruntime_BUILD_BENCHMARKS=" + (
            "ON" if args.build_micro_benchmarks else "OFF"),
        "-Donnxruntime_USE_TORCH=" + (
            "ON" if args.use_torch else "OFF"),
    ]

    if mpi_home and os.path.exists(mpi_home):
        cmake_args += ["-Donnxruntime_MPI_HOME=" + mpi_home]

    if nccl_home and os.path.exists(nccl_home):
        cmake_args += ["-Donnxruntime_NCCL_HOME=" + nccl_home]

    if torch_home and os.path.exists(torch_home):
        cmake_args += ["-Donnxruntime_TORCH_HOME=" + torch_home]

    if args.winml_root_namespace_override:
        cmake_args += ["-Donnxruntime_WINML_NAMESPACE_OVERRIDE=" +
                       args.winml_root_namespace_override]
    if args.use_openvino:
        cmake_args += ["-Donnxruntime_USE_OPENVINO=ON",
                       "-Donnxruntime_USE_OPENVINO_MYRIAD=" + (
                           "ON" if args.use_openvino == "MYRIAD_FP16" else "OFF"),
                       "-Donnxruntime_USE_OPENVINO_GPU_FP32=" + (
                           "ON" if args.use_openvino == "GPU_FP32" else "OFF"),
                       "-Donnxruntime_USE_OPENVINO_GPU_FP16=" + (
                           "ON" if args.use_openvino == "GPU_FP16" else "OFF"),
                       "-Donnxruntime_USE_OPENVINO_CPU_FP32=" + (
                           "ON" if args.use_openvino == "CPU_FP32" else "OFF"),
                       "-Donnxruntime_USE_OPENVINO_VAD_M=" + (
                           "ON" if args.use_openvino == "VAD-M_FP16" else "OFF"),
                       "-Donnxruntime_USE_OPENVINO_VAD_F=" + (
                           "ON" if args.use_openvino == "VAD-F_FP32" else "OFF"),
                       "-Donnxruntime_USE_OPENVINO_BINARY=" + (
                           "ON" if args.use_openvino else "OFF")]
    # temp turn on only for linux gpu build
    if not is_windows():
        if args.use_cuda:
            cmake_args += [
                "-Donnxruntime_USE_FULL_PROTOBUF=ON"]

    # nGraph, TensorRT and OpenVINO providers currently only supports
    # full_protobuf option.
    if (args.use_full_protobuf or args.use_ngraph or args.use_tensorrt or
            args.use_openvino or args.use_vitisai or args.gen_doc):
        cmake_args += [
            "-Donnxruntime_USE_FULL_PROTOBUF=ON",
            "-DProtobuf_USE_STATIC_LIBS=ON"
        ]

    if args.use_nuphar and args.llvm_path is not None:
        cmake_args += ["-DLLVM_DIR=%s" % args.llvm_path]

    if args.use_cuda and not is_windows():
        nvml_stub_path = cuda_home + "/lib64/stubs"
        cmake_args += ["-DCUDA_CUDA_LIBRARY=" + nvml_stub_path]

    if args.use_preinstalled_eigen:
        cmake_args += ["-Donnxruntime_USE_PREINSTALLED_EIGEN=ON",
                       "-Deigen_SOURCE_PATH=" + args.eigen_path]

    if args.android:
        cmake_args += [
            "-DCMAKE_TOOLCHAIN_FILE=" + args.android_ndk_path +
            "/build/cmake/android.toolchain.cmake",
            "-DANDROID_PLATFORM=android-" + str(args.android_api),
            "-DANDROID_ABI=" + str(args.android_abi)
        ]

        if args.android_cpp_shared:
            cmake_args += ["-DANDROID_STL=c++_shared"]

    if args.ios:
        if is_macOS():
            needed_args = [
                args.use_xcode,
                args.ios_sysroot,
                args.osx_arch,
                args.apple_deploy_target,
            ]
            arg_names = [
                "--use_xcode            " +
                "<need use xcode to cross build iOS on MacOS>",
                "--ios_sysroot          " +
                "<the location or name of the macOS platform SDK>",
                "--osx_arch             " +
                "<the Target specific architectures for iOS>",
                "--apple_deploy_target  " +
                "<the minimum version of the target platform>",
            ]
            if not all(needed_args):
                raise BuildError(
                    "iOS build on MacOS canceled due to missing arguments: " +
                    ', '.join(
                        val for val, cond in zip(arg_names, needed_args)
                        if not cond))
            cmake_args += [
                "-DCMAKE_SYSTEM_NAME=iOS",
                "-Donnxruntime_BUILD_SHARED_LIB=ON",
                "-Donnxruntime_BUILD_UNIT_TESTS=OFF",
                "-DCMAKE_OSX_SYSROOT=" + args.ios_sysroot,
                "-DCMAKE_OSX_ARCHITECTURES=" + args.osx_arch,
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + args.apple_deploy_target,
                # we do not need protoc binary for ios cross build
                "-Dprotobuf_BUILD_PROTOC_BINARIES=OFF",
                "-DCMAKE_TOOLCHAIN_FILE=" + (
                    args.ios_toolchain_file if args.ios_toolchain_file
                    else "../cmake/onnxruntime_ios.toolchain.cmake")
            ]
        else:
            # We are cross comppiling on linux
            needed_args = [
                args.ios_sysroot,
                args.arm64 or args.arm,
                args.ios_toolchain_dir
            ]
            arg_names = [
                "--ios_sysroot <path to sysroot>",
                "--arm or --arm64",
                "--ios_toolchain_dir <path to toolchain>"
            ]
            if not all(needed_args):
                raise BuildError(
                    "iOS build canceled due to missing arguments: " +
                    ', '.join(
                        val for val, cond in zip(arg_names, needed_args)
                        if not cond))
            compilers = sorted(
                glob.glob(args.ios_toolchain_dir + "/bin/*-clang*"))
            os.environ["PATH"] = os.path.join(
                args.ios_toolchain_dir, "bin") + os.pathsep + os.environ.get(
                    "PATH", "")
            os.environ["LD_LIBRARY_PATH"] = os.path.join(
                args.ios_toolchain_dir, "/lib") + os.pathsep + os.environ.get(
                    "LD_LIBRARY_PATH", "")
            if len(compilers) != 2:
                raise BuildError(
                    "error identifying compilers in ios_toolchain_dir")
            cmake_args += [
                "-DCMAKE_OSX_ARCHITECTURES=" +
                ("arm64" if args.arm64 else "arm"),
                "-DCMAKE_SYSTEM_NAME=iOSCross",
                "-Donnxruntime_BUILD_UNIT_TESTS=OFF",
                "-DCMAKE_OSX_SYSROOT=" + args.ios_sysroot,
                "-DCMAKE_C_COMPILER=" + compilers[0],
                "-DCMAKE_CXX_COMPILER=" + compilers[1]
            ]

    if path_to_protoc_exe:
        cmake_args += [
            "-DONNX_CUSTOM_PROTOC_EXECUTABLE=%s" % path_to_protoc_exe]

    if args.fuzz_testing:
        if not (args.build_shared_lib and
                is_windows() and
                args.cmake_generator == 'Visual Studio 16 2019' and
                args.use_full_protobuf):
            raise BuildError(
             "Fuzz test has only be tested with build shared libs option using MSVC on windows")
        cmake_args += [
            "-Donnxruntime_BUILD_UNIT_TESTS=ON",
            "-Donnxruntime_FUZZ_TEST=ON",
            "-Donnxruntime_USE_FULL_PROTOBUF=ON"]

    if args.gen_doc:
        cmake_args += ["-Donnxruntime_PYBIND_EXPORT_OPSCHEMA=ON"]
    else:
        cmake_args += ["-Donnxruntime_PYBIND_EXPORT_OPSCHEMA=OFF"]

    cmake_args += ["-D{}".format(define) for define in cmake_extra_defines]

    cmake_args += cmake_extra_args

    # ADO pipelines will store the pipeline build number
    # (e.g. 191101-2300.1.master) and source version in environment
    # variables. If present, use these values to define the
    # WinML/ORT DLL versions.
    build_number = os.getenv('Build_BuildNumber')
    source_version = os.getenv('Build_SourceVersion')
    if build_number and source_version:
        build_matches = re.fullmatch(
            r"(\d\d)(\d\d)(\d\d)(\d\d)\.(\d+)", build_number)
        if build_matches:
            YY = build_matches.group(2)
            MM = build_matches.group(3)
            DD = build_matches.group(4)

            # Get ORT major and minor number
            with open(os.path.join(source_dir, 'VERSION_NUMBER')) as f:
                first_line = f.readline()
                ort_version_matches = re.match(r"(\d+).(\d+)", first_line)
                if not ort_version_matches:
                    raise BuildError("Couldn't read version from VERSION_FILE")
                ort_major = ort_version_matches.group(1)
                ort_minor = ort_version_matches.group(2)
                # Example (BuildNumber: 191101-2300.1.master,
                # SourceVersion: 0bce7ae6755c792eda558e5d27ded701707dc404)
                # MajorPart = 1
                # MinorPart = 0
                # BuildPart = 1911
                # PrivatePart = 123
                # String = 191101-2300.1.master.0bce7ae
                cmake_args += [
                    "-DVERSION_MAJOR_PART={}".format(ort_major),
                    "-DVERSION_MINOR_PART={}".format(ort_minor),
                    "-DVERSION_BUILD_PART={}".format(YY),
                    "-DVERSION_PRIVATE_PART={}{}".format(MM, DD),
                    "-DVERSION_STRING={}.{}.{}.{}".format(
                        ort_major, ort_minor, build_number,
                        source_version[0:7])
                ]

    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)
        os.makedirs(config_build_dir, exist_ok=True)
        if args.use_nuphar:
            os.environ["PATH"] = os.path.join(
                config_build_dir, "external", "tvm",
                config) + os.pathsep + os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

        run_subprocess(
            cmake_args + [
                "-Donnxruntime_ENABLE_MEMLEAK_CHECKER=" +
                ("ON" if config.lower() == 'debug' and not args.use_nuphar and not
                 args.use_ngraph and not args.use_openvino and not
                 args.enable_msvc_static_runtime
                 else "OFF"), "-DCMAKE_BUILD_TYPE={}".format(config)],
            cwd=config_build_dir)


def clean_targets(cmake_path, build_dir, configs):
    for config in configs:
        log.info("Cleaning targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path,
                    "--build", build_dir2,
                    "--config", config,
                    "--target", "clean"]

        run_subprocess(cmd_args)


def build_targets(args, cmake_path, build_dir, configs, parallel, target=None):
    for config in configs:
        log.info("Building targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path,
                    "--build", build_dir2,
                    "--config", config]
        if target:
            cmd_args.extend(['--target', target])

        build_tool_args = []
        if parallel:
            num_cores = str(multiprocessing.cpu_count())
            if is_windows() and args.cmake_generator != 'Ninja':
                build_tool_args += [
                    "/maxcpucount:" + num_cores,
                    # if nodeReuse is true, msbuild processes will stay around for a bit after the build completes
                    "/nodeReuse:False",
                ]
            elif (is_macOS() and args.use_xcode):
                # CMake will generate correct build tool args for Xcode
                cmd_args += ["--parallel", num_cores]
            elif args.cmake_generator != 'Ninja':
                build_tool_args += ["-j" + num_cores]

        if build_tool_args:
            cmd_args += ["--"]
            cmd_args += build_tool_args

        env = {}
        if args.android:
            env['ANDROID_SDK_ROOT'] = args.android_sdk_path

        run_subprocess(cmd_args, env=env)


def add_dir_if_exists(directory, dir_list):
    if os.path.isdir(directory):
        dir_list.append(directory)


def setup_cuda_vars(args):
    cuda_home = ""
    cudnn_home = ""

    if args.use_cuda:
        cuda_home = args.cuda_home if args.cuda_home else os.getenv(
            "CUDA_HOME")
        cudnn_home = args.cudnn_home if args.cudnn_home else os.getenv(
            "CUDNN_HOME")

        cuda_home_valid = (cuda_home is not None and os.path.exists(cuda_home))
        cudnn_home_valid = (cudnn_home is not None and os.path.exists(
            cudnn_home))

        if not cuda_home_valid or not cudnn_home_valid:
            raise BuildError(
                "cuda_home and cudnn_home paths must be specified and valid.",
                "cuda_home='{}' valid={}. cudnn_home='{}' valid={}"
                .format(
                    cuda_home, cuda_home_valid, cudnn_home, cudnn_home_valid))

    return cuda_home, cudnn_home


def setup_tensorrt_vars(args):
    tensorrt_home = ""
    if args.use_tensorrt:
        tensorrt_home = (args.tensorrt_home if args.tensorrt_home
                         else os.getenv("TENSORRT_HOME"))
        tensorrt_home_valid = (tensorrt_home is not None and
                               os.path.exists(tensorrt_home))
        if not tensorrt_home_valid:
            raise BuildError(
                "tensorrt_home paths must be specified and valid.",
                "tensorrt_home='{}' valid={}."
                .format(tensorrt_home, tensorrt_home_valid))

        # Set maximum workspace size in byte for
        # TensorRT (1GB = 1073741824 bytes).
        os.environ["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = "1073741824"

        # Set maximum number of iterations to detect unsupported nodes
        # and partition the models for TensorRT.
        os.environ["ORT_TENSORRT_MAX_PARTITION_ITERATIONS"] = "1000"

        # Set minimum subgraph node size in graph partitioning
        # for TensorRT.
        os.environ["ORT_TENSORRT_MIN_SUBGRAPH_SIZE"] = "1"

        # Set FP16 flag
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"

    return tensorrt_home


def setup_migraphx_vars(args):

    migraphx_home = None

    if (args.use_migraphx):
        print("migraphx_home = {}".format(args.migraphx_home))
        migraphx_home = args.migraphx_home or os.getenv("MIGRAPHX_HOME") or None

        migraphx_home_not_valid = (migraphx_home and not os.path.exists(migraphx_home))

        if (migraphx_home_not_valid):
            raise BuildError("migraphx_home paths must be specified and valid.",
                             "migraphx_home='{}' valid={}."
                             .format(migraphx_home, migraphx_home_not_valid))
    return migraphx_home or ''


def setup_dml_build(args, cmake_path, build_dir, configs):
    if args.use_dml:
        for config in configs:
            # Run the RESTORE_PACKAGES target to perform the initial
            # NuGet setup.
            cmd_args = [cmake_path,
                        "--build", get_config_build_dir(build_dir, config),
                        "--config", config,
                        "--target", "RESTORE_PACKAGES"]
            run_subprocess(cmd_args)


def adb_push(src, dest, **kwargs):
    return run_subprocess(['adb', 'push', src, dest], **kwargs)


def adb_shell(*args, **kwargs):
    return run_subprocess(['adb', 'shell', *args], **kwargs)


def run_training_python_frontend_tests(cwd):
    run_subprocess([sys.executable, 'onnxruntime_test_ort_trainer.py'], cwd=cwd)
    run_subprocess([sys.executable, 'onnxruntime_test_training_unit_tests.py'], cwd=cwd)
    run_subprocess([
        sys.executable, 'orttraining_test_transformers.py',
        'BertModelTest.test_for_pretraining_full_precision_list_input'], cwd=cwd)
    run_subprocess([
        sys.executable, 'orttraining_test_transformers.py',
        'BertModelTest.test_for_pretraining_full_precision_dict_input'], cwd=cwd)
    run_subprocess([
        sys.executable, 'orttraining_test_transformers.py',
        'BertModelTest.test_for_pretraining_full_precision_list_and_dict_input'], cwd=cwd)
    run_subprocess([sys.executable, '-m', 'pytest', '-sv', 'orttraining_test_orttrainer_frontend.py'], cwd=cwd)
    run_subprocess([sys.executable, '-m', 'pytest', '-sv', 'orttraining_test_orttrainer_bert_toy_onnx.py'], cwd=cwd)


def run_training_python_frontend_e2e_tests(cwd):
    # frontend tests are to be added here:
    log.info("Running python frontend e2e tests.")

    import torch
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        log.debug('RUN: mpirun -n {} {} orttraining_run_glue.py'.format(ngpus, sys.executable))
        run_subprocess(['mpirun', '-n', str(ngpus), sys.executable, 'orttraining_run_glue.py'], cwd=cwd)

    # with orttraining_run_glue.py.
    # 1. we like to force to use single GPU (with CUDA_VISIBLE_DEVICES)
    #   for fine-tune tests.
    # 2. need to run test separately (not to mix between fp16
    #   and full precision runs. this need to be investigated).
    run_subprocess(
        [sys.executable, 'orttraining_run_glue.py', 'ORTGlueTest.test_bert_with_mrpc', '-v'],
        cwd=cwd, env={'CUDA_VISIBLE_DEVICES': '0'})

    run_subprocess(
        [sys.executable, 'orttraining_run_glue.py', 'ORTGlueTest.test_bert_fp16_with_mrpc', '-v'],
        cwd=cwd, env={'CUDA_VISIBLE_DEVICES': '0'})

    run_subprocess(
        [sys.executable, 'orttraining_run_glue.py', 'ORTGlueTest.test_roberta_with_mrpc', '-v'],
        cwd=cwd, env={'CUDA_VISIBLE_DEVICES': '0'})

    run_subprocess(
        [sys.executable, 'orttraining_run_glue.py', 'ORTGlueTest.test_roberta_fp16_with_mrpc', '-v'],
        cwd=cwd, env={'CUDA_VISIBLE_DEVICES': '0'})

    run_subprocess(
        [sys.executable, 'orttraining_run_multiple_choice.py', 'ORTMultipleChoiceTest.test_bert_fp16_with_swag', '-v'],
        cwd=cwd, env={'CUDA_VISIBLE_DEVICES': '0'})

    run_subprocess([sys.executable, 'onnxruntime_test_ort_trainer_with_mixed_precision.py'], cwd=cwd)

    run_subprocess([
        sys.executable, 'orttraining_test_transformers.py',
        'BertModelTest.test_for_pretraining_mixed_precision'], cwd=cwd)

    run_subprocess([
        sys.executable, 'orttraining_test_transformers.py',
        'BertModelTest.test_for_pretraining_mixed_precision_with_gradient_accumulation'], cwd=cwd)


def run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs):
    for config in configs:
        log.info("Running tests for %s configuration", config)
        cwd = get_config_build_dir(build_dir, config)

        if args.enable_training and args.use_cuda and args.enable_training_python_frontend_e2e_tests:
            # run frontend tests for orttraining-linux-gpu-frontend_test-ci-pipeline.
            # this is not a PR merge test so skip other non-frontend tests.
            run_training_python_frontend_e2e_tests(cwd=cwd)
            run_training_python_frontend_tests(cwd=cwd)
            continue

        if args.android:
            if args.android_abi == 'x86_64':
                run_subprocess(os.path.join(
                    source_dir, 'tools', 'ci_build', 'github', 'android',
                    'start_android_emulator.sh'))
                adb_push('testdata', '/data/local/tmp/', cwd=cwd)
                adb_push(
                    os.path.join(source_dir, 'cmake', 'external', 'onnx', 'onnx', 'backend', 'test'),
                    '/data/local/tmp/', cwd=cwd)
                adb_push('onnxruntime_test_all', '/data/local/tmp/', cwd=cwd)
                adb_push('onnx_test_runner', '/data/local/tmp/', cwd=cwd)
                adb_shell(
                    'cd /data/local/tmp && /data/local/tmp/onnxruntime_test_all')
                if args.use_nnapi:
                    adb_shell(
                        'cd /data/local/tmp && /data/local/tmp/onnx_test_runner -e nnapi /data/local/tmp/test')  # noqa
                else:
                    adb_shell(
                        'cd /data/local/tmp && /data/local/tmp/onnx_test_runner /data/local/tmp/test')  # noqa
            elif args.android_abi == 'arm64-v8a':
                # For Android arm64 abi we are only verify the size of the binary generated by minimal build config
                # Will fail the build if the shared_lib size is larger than the threshold
                if args.minimal_build and config == 'MinSizeRel' and args.build_shared_lib:
                    # set current size limit to 1100KB
                    bin_size_threshold = 1100000
                    bin_actual_size = os.path.getsize(os.path.join(cwd, 'libonnxruntime.so'))
                    log.info('Android arm64 minsizerel libonnxruntime.so size [' + str(bin_actual_size) + 'B]')
                    if bin_actual_size > bin_size_threshold:
                        raise BuildError('Android arm64 minsizerel libonnxruntime.so size [' + str(bin_actual_size) +
                                         'B] is bigger than threshold [' + str(bin_size_threshold) + 'B]')
            continue
        dll_path_list = []
        if args.use_nuphar:
            dll_path_list.append(os.path.join(
                build_dir, config, "external", "tvm", config))
        if args.use_tensorrt:
            dll_path_list.append(os.path.join(args.tensorrt_home, 'lib'))
        if args.use_mklml:
            dll_path_list.append(os.path.join(build_dir, config, "mklml", "src", "project_mklml", "lib"))
        if not is_windows():
            # A workaround for making libonnxruntime_providers_shared.so loadable.
            dll_path_list.append(os.path.join(build_dir, config))

        dll_path = None
        if len(dll_path_list) > 0:
            dll_path = os.pathsep.join(dll_path_list)

        if ctest_path is None:
            # Get the "Google Test Adapter" for vstest.
            if not os.path.exists(os.path.join(cwd,
                                               'googletestadapter.0.17.1')):
                run_subprocess(
                    ['nuget.exe', 'restore',
                     os.path.join(source_dir, 'packages.config'),
                     '-ConfigFile', os.path.join(source_dir, 'NuGet.config'),
                     '-PackagesDirectory', cwd])
            cwd2 = os.path.join(cwd, config)
            executables = ['onnxruntime_test_all.exe']
            if args.build_shared_lib:
                executables.append('onnxruntime_shared_lib_test.exe')
                executables.append('onnxruntime_global_thread_pools_test.exe')
            run_subprocess(
                ['vstest.console.exe', '--parallel',
                 '--TestAdapterPath:..\\googletestadapter.0.17.1\\build\\_common',  # noqa
                 '/Logger:trx', '/Enablecodecoverage', '/Platform:x64',
                 "/Settings:%s" % os.path.join(
                     source_dir, 'cmake\\codeconv.runsettings')] + executables,
                cwd=cwd2, dll_path=dll_path)
        else:
            ctest_cmd = [ctest_path, "--build-config", config, "--verbose"]
            run_subprocess(ctest_cmd, cwd=cwd, dll_path=dll_path)

        if args.enable_pybind:
            # Disable python tests for TensorRT because many tests are
            # not supported yet.
            if args.use_tensorrt:
                return

            # Disable python tests in a reduced build as we don't know which ops have been included and which
            # models can run
            if args.include_ops_by_model or args.include_ops_by_config or args.minimal_build:
                return

            if is_windows():
                cwd = os.path.join(cwd, config)

            run_subprocess([sys.executable, 'onnxruntime_test_python.py'], cwd=cwd, dll_path=dll_path)

            # For CUDA enabled builds test IOBinding feature
            # Limit testing to Windows non-ARM builds for now
            iobinding_test = False
            if args.use_cuda and not (args.arm or args.arm64):
                # We need to have Torch installed to test the IOBinding feature
                # which currently uses Torch's allocator to allocate GPU memory for testing
                iobinding_test = True

                # Try install Torch on Windows
                if is_windows():
                    log.info("Attempting to install Torch to test ORT's IOBinding feature")
                    install_torch()

                try:
                    import torch  # noqa
                except ImportError as error:
                    iobinding_test = False
                    log.exception(error)
                    log.warning(
                        "Torch is not installed. "
                        "The IOBinding tests will be skipped as it requires Torch.")

            if iobinding_test:
                log.info("Testing IOBinding feature")
                run_subprocess([sys.executable, 'onnxruntime_test_python_iobinding.py'], cwd=cwd, dll_path=dll_path)

            if not args.disable_ml_ops:
                run_subprocess([sys.executable, 'onnxruntime_test_python_mlops.py'], cwd=cwd, dll_path=dll_path)

            if args.enable_training and args.use_cuda:
                # run basic frontend tests
                run_training_python_frontend_tests(cwd=cwd)

            try:
                import onnx  # noqa
                onnx_test = True
            except ImportError as error:
                log.exception(error)
                log.warning("onnx is not installed. The ONNX tests will be skipped.")
                onnx_test = False

            if onnx_test:
                run_subprocess([sys.executable, 'onnxruntime_test_python_backend.py'], cwd=cwd, dll_path=dll_path)

                if not args.disable_ml_ops:
                    run_subprocess([sys.executable, 'onnxruntime_test_python_backend_mlops.py'],
                                   cwd=cwd, dll_path=dll_path)

                run_subprocess([sys.executable,
                                os.path.join(source_dir, 'onnxruntime', 'test', 'onnx', 'gen_test_models.py'),
                                '--output_dir', 'test_models'], cwd=cwd)

                if not args.skip_onnx_tests:
                    run_subprocess([os.path.join(cwd, 'onnx_test_runner'), 'test_models'], cwd=cwd)
                if config != 'Debug':
                    run_subprocess([sys.executable, 'onnx_backend_test_series.py'], cwd=cwd, dll_path=dll_path)

            if not args.skip_keras_test:
                try:
                    import onnxmltools  # noqa
                    import keras  # noqa
                    onnxml_test = True
                except ImportError:
                    log.warning(
                        "onnxmltools and keras are not installed. "
                        "The keras tests will be skipped.")
                    onnxml_test = False
                if onnxml_test:
                    run_subprocess(
                        [sys.executable, 'onnxruntime_test_python_keras.py'],
                        cwd=cwd, dll_path=dll_path)


def nuphar_run_python_tests(build_dir, configs):
    """nuphar temporary function for running python tests separately
    as it requires ONNX 1.5.0
    """
    for config in configs:
        if config == 'Debug':
            continue
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
            cwd = os.path.join(cwd, config)
        dll_path = os.path.join(build_dir, config, "external", "tvm", config)
        # install onnx for shape inference in testing Nuphar scripts
        # this needs to happen after onnx_test_data preparation which
        # uses onnx 1.3.0
        run_subprocess(
            [sys.executable, '-m', 'pip', 'install', '--user', 'onnx==1.5.0'])
        run_subprocess(
            [sys.executable, 'onnxruntime_test_python_nuphar.py'],
            cwd=cwd, dll_path=dll_path)


def run_nodejs_tests(nodejs_binding_dir):
    args = ['npm', 'test', '--', '--timeout=2000']
    if is_windows():
        args = ['cmd', '/c'] + args
    run_subprocess(args, cwd=nodejs_binding_dir)


def build_python_wheel(
        source_dir, build_dir, configs, use_cuda, use_ngraph, use_dnnl,
        use_tensorrt, use_openvino, use_nuphar, use_vitisai, use_acl, use_armnn, use_dml,
        wheel_name_suffix, enable_training, nightly_build=False, featurizers_build=False, use_ninja=False):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows() and not use_ninja:
            cwd = os.path.join(cwd, config)

        args = [sys.executable, os.path.join(source_dir, 'setup.py'),
                'bdist_wheel']

        # We explicitly override the platform tag in the name of the generated build wheel
        # so that we can install the wheel on Mac OS X versions 10.12+.
        # Without this explicit override, we will something like this while building on MacOS 10.14 -
        # [WARNING] MACOSX_DEPLOYMENT_TARGET is set to a lower value (10.12)
        # than the version on which the Python interpreter was compiled (10.14) and will be ignored.
        # Since we need to support 10.12+, we explicitly override the platform tag.
        # See PR #3626 for more details
        if is_macOS():
            args += ['-p', 'macosx_10_12_x86_64']

        # Any combination of the following arguments can be applied
        if nightly_build:
            args.append('--nightly_build')
        if featurizers_build:
            args.append("--use_featurizers")
        if wheel_name_suffix:
            args.append('--wheel_name_suffix={}'.format(wheel_name_suffix))
        if enable_training:
            args.append("--enable_training")

        # The following arguments are mutually exclusive
        if use_tensorrt:
            args.append('--use_tensorrt')
        elif use_cuda:
            args.append('--use_cuda')
        elif use_ngraph:
            args.append('--use_ngraph')
        elif use_openvino:
            args.append('--use_openvino')
        elif use_dnnl:
            args.append('--use_dnnl')
        elif use_nuphar:
            args.append('--use_nuphar')
        elif use_vitisai:
            args.append('--use_vitisai')
        elif use_acl:
            args.append('--use_acl')
        elif use_armnn:
            args.append('--use_armnn')
        elif use_dml:
            args.append('--use_dml')

        run_subprocess(args, cwd=cwd)


def derive_linux_build_property():
    if is_windows():
        return "/p:IsLinuxBuild=\"false\""
    else:
        return "/p:IsLinuxBuild=\"true\""


def build_nuget_package(configs, use_cuda, use_openvino, use_tensorrt, use_dnnl, use_mklml):
    if not (is_windows() or is_linux()):
        raise BuildError(
            'Currently csharp builds and nuget package creation is only supportted '
            'on Windows and Linux platforms.')

    build_dir = os.path.join(os.getcwd(), 'csharp')
    is_linux_build = derive_linux_build_property()

    # derive package name and execution provider based on the build args
    execution_provider = "/p:ExecutionProvider=\"None\""
    package_name = "/p:OrtPackageId=\"Microsoft.ML.OnnxRuntime\""
    if use_openvino:
        execution_provider = "/p:ExecutionProvider=\"openvino\""
        package_name = "/p:OrtPackageId=\"Microsoft.ML.OnnxRuntime.OpenVino\""
    elif use_tensorrt:
        execution_provider = "/p:ExecutionProvider=\"tensorrt\""
        package_name = "/p:OrtPackageId=\"Microsoft.ML.OnnxRuntime.TensorRT\""
    elif use_dnnl:
        execution_provider = "/p:ExecutionProvider=\"dnnl\""
        package_name = "/p:OrtPackageId=\"Microsoft.ML.OnnxRuntime.DNNL\""
    elif use_cuda:
        package_name = "/p:OrtPackageId=\"Microsoft.ML.OnnxRuntime.Gpu\""
    elif use_mklml:
        package_name = "/p:OrtPackageId=\"Microsoft.ML.OnnxRuntime.MKLML\""
    else:
        pass

    # dotnet restore
    cmd_args = ["dotnet", "restore", "OnnxRuntime.CSharp.sln", "--configfile", "Nuget.CSharp.config"]
    run_subprocess(cmd_args, cwd=build_dir)

    # build csharp bindings and create nuget package for each config
    for config in configs:
        if is_linux():
            native_build_dir = os.path.join(os.getcwd(), 'build//Linux//', config)
            cmd_args = ["make", "install", "DESTDIR=.//nuget-staging"]
            run_subprocess(cmd_args, cwd=native_build_dir)

        configuration = "/p:Configuration=\"" + config + "\""

        cmd_args = ["dotnet", "msbuild", "OnnxRuntime.CSharp.sln", configuration, package_name, is_linux_build]
        run_subprocess(cmd_args, cwd=build_dir)

        cmd_args = [
            "dotnet", "msbuild", "OnnxRuntime.CSharp.proj", "/t:CreatePackage",
            package_name, configuration, execution_provider, is_linux_build]
        run_subprocess(cmd_args, cwd=build_dir)


def run_csharp_tests(use_cuda, use_openvino, use_tensorrt, use_dnnl):
    # Currently only running tests on windows.
    if not is_windows():
        return
    build_dir = os.path.join(os.getcwd(), 'csharp')
    is_linux_build = derive_linux_build_property()

    # define macros based on build args
    macros = ""
    if use_openvino:
        macros += "USE_OPENVINO;"
    if use_tensorrt:
        macros += "USE_TENSORRT;"
    if use_dnnl:
        macros += "USE_DNNL;"
    if use_cuda:
        macros += "USE_CUDA;"

    define_constants = ""
    if macros != "":
        define_constants = "/p:DefineConstants=\"" + macros + "\""

    # Skip pretrained models test. Only run unit tests as part of the build
    # "/property:DefineConstants=\"USE_CUDA;USE_OPENVINO\"",
    cmd_args = ["dotnet", "test", "test\\Microsoft.ML.OnnxRuntime.Tests\\Microsoft.ML.OnnxRuntime.Tests.csproj",
                "--filter", "FullyQualifiedName!=Microsoft.ML.OnnxRuntime.Tests.InferenceTest.TestPreTrainedModels",
                is_linux_build, define_constants, "--verbosity", "detailed"]
    run_subprocess(cmd_args, cwd=build_dir)


def build_protoc_for_host(cmake_path, source_dir, build_dir, args):
    if (args.arm or args.arm64 or args.enable_windows_store) and (not is_windows() and not args.ios):
        raise BuildError(
            'Currently only support building protoc for Windows host while '
            'cross-compiling for ARM/ARM64/Store and linux cross-compiling iOS')

    log.info(
        "Building protoc for host to be used in cross-compiled build process")
    protoc_build_dir = os.path.join(os.getcwd(), build_dir, 'host_protoc')
    os.makedirs(protoc_build_dir, exist_ok=True)
    # Generate step
    cmd_args = [
        cmake_path,
        os.path.join(source_dir, 'cmake', 'external', 'protobuf', 'cmake'),
        '-Dprotobuf_BUILD_TESTS=OFF',
        '-Dprotobuf_WITH_ZLIB_DEFAULT=OFF',
        '-Dprotobuf_BUILD_SHARED_LIBS=OFF'
    ]

    is_ninja = args.cmake_generator == 'Ninja'
    if args.cmake_generator is not None and not (is_macOS() and args.use_xcode):
        cmd_args += ['-G', args.cmake_generator]
    if is_windows():
        if not is_ninja:
            cmd_args += ['-T', 'host=x64']
    elif is_macOS():
        if args.use_xcode:
            cmd_args += ['-G', 'Xcode']
            # CMake < 3.18 has a bug setting system arch to arm64 (if not specified) for Xcode 12,
            # protoc for host should be built using host architecture
            # Explicitly specify the CMAKE_OSX_ARCHITECTURES for x86_64 Mac.
            import platform
            if platform.machine() == 'x86_64':
                cmd_args += ['-DCMAKE_OSX_ARCHITECTURES=x86_64']

    run_subprocess(cmd_args, cwd=protoc_build_dir)
    # Build step
    cmd_args = [cmake_path,
                "--build", protoc_build_dir,
                "--config", "Release",
                "--target", "protoc"]
    run_subprocess(cmd_args)

    # Absolute protoc path is needed for cmake
    config_dir = ''
    suffix = ''

    if (is_windows() and not is_ninja) or (is_macOS() and args.use_xcode):
        config_dir = 'Release'

    if is_windows():
        suffix = '.exe'

    expected_protoc_path = os.path.join(protoc_build_dir, config_dir, 'protoc' + suffix)

    if not os.path.exists(expected_protoc_path):
        raise BuildError("Couldn't find {}. Host build of protoc failed.".format(expected_protoc_path))

    return expected_protoc_path


def generate_documentation(source_dir, build_dir, configs):
    operator_doc_path = os.path.join(source_dir, 'docs', 'ContribOperators.md')
    opkernel_doc_path = os.path.join(source_dir, 'docs', 'OperatorKernels.md')
    for config in configs:
        # Copy the gen_contrib_doc.py.
        shutil.copy(
            os.path.join(source_dir, 'tools', 'python', 'gen_contrib_doc.py'),
            os.path.join(build_dir, config))
        shutil.copy(
            os.path.join(source_dir, 'tools', 'python', 'gen_opkernel_doc.py'),
            os.path.join(build_dir, config))
        run_subprocess(
            [sys.executable,
             'gen_contrib_doc.py',
             '--output_path', operator_doc_path],
            cwd=os.path.join(build_dir, config))
        run_subprocess(
            [sys.executable,
             'gen_opkernel_doc.py',
             '--output_path', opkernel_doc_path],
            cwd=os.path.join(build_dir, config))
    docdiff = ''
    try:
        docdiff = subprocess.check_output(['git', 'diff', opkernel_doc_path])
    except subprocess.CalledProcessError:
        print('git diff returned non-zero error code')
    if len(docdiff) > 0:
        # Show warning instead of throwing exception, because it is
        # dependent on build configuration for including
        # execution propviders
        log.warning(
            'The updated opkernel document file ' + str(opkernel_doc_path) +
            ' is different from the checked in version. Consider '
            'regenerating the file with CPU, DNNL and CUDA providers enabled.')
        log.debug('diff:\n' + str(docdiff))

    docdiff = ''
    try:
        docdiff = subprocess.check_output(['git', 'diff', operator_doc_path])
    except subprocess.CalledProcessError:
        print('git diff returned non-zero error code')
    if len(docdiff) > 0:
        raise BuildError(
            'The updated operator document file ' +
            str(operator_doc_path) + ' must be checked in.\n diff:\n' +
            str(docdiff))


def main():
    args = parse_arguments()
    cmake_extra_defines = (args.cmake_extra_defines
                           if args.cmake_extra_defines else [])
    cross_compiling = args.arm or args.arm64 or args.android

    # If there was no explicit argument saying what to do, default
    # to update, build and test (for native builds).
    if not (args.update or args.clean or args.build or args.test):
        log.debug(
            "Defaulting to running update, build "
            "[and test for native builds].")
        args.update = True
        args.build = True
        if cross_compiling:
            args.test = args.android_abi == 'x86_64' or args.android_abi == 'arm64-v8a'
        else:
            args.test = True

    if args.skip_tests:
        args.test = False

    if args.include_ops_by_model or args.include_ops_by_config:
        from exclude_unused_ops import exclude_unused_ops
        models_path = args.include_ops_by_model if args.include_ops_by_model else ''
        config_path = args.include_ops_by_config if args.include_ops_by_config else ''
        exclude_unused_ops(models_path, config_path, use_cuda=args.use_cuda)

    if args.use_tensorrt:
        args.use_cuda = True

    if args.build_wheel or args.gen_doc:
        args.enable_pybind = True

    if args.build_csharp or args.build_nuget or args.build_java or args.build_nodejs:
        args.build_shared_lib = True

    if args.build_nuget and cross_compiling:
        raise BuildError('Currently nuget package creation is not supported while cross-compiling')

    if args.enable_pybind and args.disable_exceptions:
        raise BuildError('Python bindings require exceptions to be enabled.')

    if args.minimal_build and args.disable_ort_format_load:
        raise BuildError('Minimal build requires loading ORT format models.')

    # Disabling unit tests for VAD-F as FPGA only supports
    # models with NCHW layout
    if args.use_openvino == "VAD-F_FP32":
        args.test = False

    configs = set(args.config)

    # setup paths and directories
    cmake_path = resolve_executable_path(args.cmake_path)
    ctest_path = None if args.use_vstest else resolve_executable_path(
        args.ctest_path)
    build_dir = args.build_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))

    # if using cuda, setup cuda paths and env vars
    cuda_home, cudnn_home = setup_cuda_vars(args)

    mpi_home = args.mpi_home
    nccl_home = args.nccl_home
    torch_home = args.torch_home

    # if using tensorrt, setup tensorrt paths
    tensorrt_home = setup_tensorrt_vars(args)

    # if using migraphx, setup migraphx paths
    migraphx_home = setup_migraphx_vars(args)

    os.makedirs(build_dir, exist_ok=True)

    log.info("Build started")
    if args.update:
        cmake_extra_args = []
        path_to_protoc_exe = args.path_to_protoc_exe
        if not args.skip_submodule_sync:
            update_submodules(source_dir)
        if is_windows():
            if args.cmake_generator == 'Ninja':
                if args.x86 or args.arm or args.arm64:
                    raise BuildError(
                        "To cross-compile with Ninja, load the toolset "
                        "environment for the target processor (e.g. Cross "
                        "Tools Command Prompt for VS)")
                cmake_extra_args = ['-G', args.cmake_generator]
            elif args.x86:
                cmake_extra_args = [
                    '-A', 'Win32', '-T', 'host=x64', '-G', args.cmake_generator
                ]
            elif args.arm or args.arm64:
                # Cross-compiling for ARM(64) architecture
                # First build protoc for host to use during cross-compilation
                if path_to_protoc_exe is None:
                    path_to_protoc_exe = build_protoc_for_host(
                        cmake_path, source_dir, build_dir, args)
                if args.arm:
                    cmake_extra_args = ['-A', 'ARM']
                else:
                    cmake_extra_args = ['-A', 'ARM64']
                cmake_extra_args += ['-G', args.cmake_generator]
                # Cannot test on host build machine for cross-compiled
                # builds (Override any user-defined behaviour for test if any)
                if args.test:
                    log.info(
                        "Cannot test on host build machine for cross-compiled "
                        "ARM(64) builds. Will skip test running after build.")
                    args.test = False
            else:
                if (args.msvc_toolset == '14.16' and
                        args.cmake_generator == 'Visual Studio 16 2019'):
                    # CUDA 10.0 requires _MSC_VER >= 1700 and
                    # _MSC_VER < 1920, aka Visual Studio version
                    # in [2012, 2019). In VS2019, we have to use
                    # Side-by-side minor version MSVC toolsets from
                    # Visual Studio 2017 14.16 is MSVC version
                    # 141 is MSVC Toolset Version
                    # Cuda VS extension should be installed to
                    # C:\Program Files (x86)\Microsoft Visual
                    # Studio\2019\Enterprise\MSBuild\Microsoft\VC\v160\BuildCustomizations  # noqa
                    toolset = 'v141,host=x64,version=' + args.msvc_toolset
                elif args.msvc_toolset:
                    toolset = 'host=x64,version=' + args.msvc_toolset
                else:
                    toolset = 'host=x64'
                if args.cuda_version:
                    toolset += ',cuda=' + args.cuda_version
                cmake_extra_args = [
                    '-A', 'x64', '-T', toolset, '-G', args.cmake_generator
                ]
            if args.enable_windows_store:
                cmake_extra_args.append(
                    '-DCMAKE_TOOLCHAIN_FILE=' + os.path.join(
                        source_dir, 'cmake', 'store_toolchain.cmake'))
            if args.enable_wcos:
                cmake_extra_args.append('-DCMAKE_USER_MAKE_RULES_OVERRIDE=wcos_rules_override.cmake')
        elif args.cmake_generator is not None and not (is_macOS() and args.use_xcode):
            cmake_extra_args += ['-G', args.cmake_generator]
        elif is_macOS() and args.use_xcode:
            cmake_extra_args += ['-G', 'Xcode']

        if (args.android or args.ios or args.enable_windows_store) and args.path_to_protoc_exe is None:
            # Cross-compiling for Android and iOS
            path_to_protoc_exe = build_protoc_for_host(
                cmake_path, source_dir, build_dir, args)

        if is_ubuntu_1604():
            if (args.arm or args.arm64):
                raise BuildError(
                    "Only Windows ARM(64) cross-compiled builds supported "
                    "currently through this script")
            install_ubuntu_deps(args)
            if not is_docker() and not args.use_acl and not args.use_armnn:
                install_python_deps()
        if args.enable_pybind and is_windows():
            install_python_deps(args.numpy_version)
        if args.enable_onnx_tests:
            setup_test_data(build_dir, configs)
        generate_build_tree(
            cmake_path, source_dir, build_dir, cuda_home, cudnn_home, mpi_home, nccl_home,
            tensorrt_home, migraphx_home, torch_home, path_to_protoc_exe, configs, cmake_extra_defines,
            args, cmake_extra_args)

    if args.clean:
        clean_targets(cmake_path, build_dir, configs)

    # if using DML, perform initial nuget package restore
    setup_dml_build(args, cmake_path, build_dir, configs)

    if args.build:
        build_targets(args, cmake_path, build_dir, configs, args.parallel, args.target)

    if args.test:
        run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs)

        # run nuphar python tests last, as it installs ONNX 1.5.0
        if args.enable_pybind and not args.skip_onnx_tests and args.use_nuphar:
            nuphar_run_python_tests(build_dir, configs)

        # run node.js binding tests
        if args.build_nodejs and not args.skip_nodejs_tests:
            nodejs_binding_dir = os.path.normpath(os.path.join(source_dir, "nodejs"))
            run_nodejs_tests(nodejs_binding_dir)

    if args.build:
        if args.build_wheel:
            nightly_build = bool(os.getenv('NIGHTLY_BUILD') == '1')
            build_python_wheel(
                source_dir,
                build_dir,
                configs,
                args.use_cuda,
                args.use_ngraph,
                args.use_dnnl,
                args.use_tensorrt,
                args.use_openvino,
                args.use_nuphar,
                args.use_vitisai,
                args.use_acl,
                args.use_armnn,
                args.use_dml,
                args.wheel_name_suffix,
                args.enable_training,
                nightly_build=nightly_build,
                featurizers_build=args.use_featurizers,
                use_ninja=(args.cmake_generator == 'Ninja')
            )
        if args.build_nuget:
            build_nuget_package(
                configs,
                args.use_cuda,
                args.use_openvino,
                args.use_tensorrt,
                args.use_dnnl,
                args.use_mklml
            )

    if args.test and args.build_nuget:
        run_csharp_tests(
            args.use_cuda,
            args.use_openvino,
            args.use_tensorrt,
            args.use_dnnl)

    if args.gen_doc and (args.build or args.test):
        generate_documentation(source_dir, build_dir, configs)

    log.info("Build complete")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseError as e:
        log.error(str(e))
        sys.exit(1)
