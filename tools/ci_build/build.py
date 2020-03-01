#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import fileinput
import getpass
import glob
import logging
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import hashlib
import itertools
from os.path import expanduser

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("Build")

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

def checkPythonVersion():
    # According to the BUILD.md, python 3.5+ is required:
    # Python 2 is definitely not supported and it should be safer to consider it wont run with python 4:
    if sys.version_info[0] != 3 :
        raise BuildError("Bad python major version: expecting python 3, found version '{}'".format(sys.version))
    if sys.version_info[1] < 5 :
        raise BuildError("Bad python minor version: expecting python 3.5+, found version '{}'".format(sys.version))

checkPythonVersion()

def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNXRuntime CI build driver.",
                                     usage='''
Default behavior is --update --build --test for native architecture builds.
Default behavior is --update --build for cross-compiled builds.

The Update phase will update git submodules, and run cmake to generate makefiles.
The Build phase will build all projects.
The Test phase will run all unit tests, and optionally the ONNX tests.

Use the individual flags to only run the specified stages.
                                     ''')
    # Main arguments
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument("--config", nargs="+", default=["Debug"],
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration(s) to build.")
    parser.add_argument("--update", action='store_true', help="Update makefiles.")
    parser.add_argument("--build", action='store_true', help="Build.")
    parser.add_argument("--clean", action='store_true', help="Run 'cmake --build --target clean' for the selected config/s.")
    parser.add_argument("--parallel", action='store_true', help='''Use parallel build.
    The build setup doesn't get all dependencies right, so --parallel only works if you're just rebuilding ONNXRuntime code.
    If you've done an update that fetched external dependencies you have to build without --parallel the first time.
    Once that's done, run with "--build --parallel --test" to just build in parallel and run tests.''')
    parser.add_argument("--test", action='store_true', help="Run unit tests.")
    parser.add_argument("--skip_tests", action='store_true', help="Skip all tests.")

    # Test options
    parser.add_argument("--ctest_label_regex",
                        help="Only run CTest tests with a label matching the pattern (passed to ctest --label-regex).")
    parser.add_argument("--enable_training_e2e_tests", action="store_true",
                        help="Enable the training end-to-end tests.")
    parser.add_argument("--training_e2e_test_data_path",
                        help="Path to training end-to-end test data directory.")

    # enable ONNX tests
    parser.add_argument("--enable_onnx_tests", action='store_true',
                        help='''When running the Test phase, run onnx_test_running against available test data directories.''')
    parser.add_argument("--path_to_protoc_exe", help="Path to protoc exe. ")

    # generate documentaiton
    parser.add_argument("--gen_doc", action='store_true', help="Generate documentation on contrib ops")

    # CUDA related
    parser.add_argument("--use_cuda", action='store_true', help="Enable CUDA.")
    parser.add_argument("--cuda_version", help="The version of CUDA toolkit to use. Auto-detect if not specified. e.g. 9.0")
    parser.add_argument("--cuda_home", help="Path to CUDA home."
                                            "Read from CUDA_HOME environment variable if --use_cuda is true and --cuda_home is not specified.")
    parser.add_argument("--cudnn_home", help="Path to CUDNN home. "
                                             "Read from CUDNN_HOME environment variable if --use_cuda is true and --cudnn_home is not specified.")

    # Python bindings
    parser.add_argument("--enable_pybind", action='store_true', help="Enable Python Bindings.")
    parser.add_argument("--build_wheel", action='store_true', help="Build Python Wheel. ")
    parser.add_argument("--numpy_version", help="Installs a specific version of numpy "
                        "before building the python binding.")
    parser.add_argument("--skip-keras-test", action='store_true', help="Skip tests with Keras if keras is installed")

    # C-Sharp bindings
    parser.add_argument("--build_csharp", action='store_true', help="Build C#.Net DLL and NuGet package")

    # Java bindings
    parser.add_argument("--build_java", action='store_true', help="Build Java bindings.")

    # Build a shared lib
    parser.add_argument("--build_shared_lib", action='store_true', help="Build a shared library for the ONNXRuntime.")

    # Build options
    parser.add_argument("--cmake_extra_defines", nargs="+",
                        help="Extra definitions to pass to CMake during build system generation. " +
                             "These are just CMake -D options without the leading -D.")
    parser.add_argument("--x86", action='store_true',
                        help="Create x86 makefiles. Requires --update and no existing cache CMake setup. Delete CMakeCache.txt if needed")
    parser.add_argument("--arm", action='store_true',
                        help="Create ARM makefiles. Requires --update and no existing cache CMake setup. Delete CMakeCache.txt if needed")
    parser.add_argument("--arm64", action='store_true',
                        help="Create ARM64 makefiles. Requires --update and no existing cache CMake setup. Delete CMakeCache.txt if needed")
    parser.add_argument("--msvc_toolset", help="MSVC toolset to use. e.g. 14.11")
    parser.add_argument("--android", action='store_true', help='Build for Android')
    parser.add_argument("--android_abi", type=str, default='arm64-v8a',
            help='')
    parser.add_argument("--android_api", type=int, default=27,
            help='Android API Level, e.g. 21')
    parser.add_argument("--android_sdk_path", type=str, help='Path to the Android SDK')
    parser.add_argument("--android_ndk_path", default="", help="Path to the Android NDK")

    # Arguments needed by CI
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake program.")
    parser.add_argument("--ctest_path", default="ctest", help="Path to the CTest program.")
    parser.add_argument("--skip_submodule_sync", action='store_true', help="Don't do a 'git submodule update'. Makes the Update phase faster.")
    parser.add_argument("--use_vstest", action='store_true', help="Use use_vstest for running unitests.")
    parser.add_argument("--use_jemalloc", action='store_true', help="Use jemalloc.")
    parser.add_argument("--use_mimalloc", default=['none'], choices=['none', 'stl', 'arena', 'all'], help="Use mimalloc.")
    parser.add_argument("--use_openblas", action='store_true', help="Build with OpenBLAS.")
    parser.add_argument("--use_dnnl", action='store_true', help="Build with DNNL.")
    parser.add_argument("--use_mklml", action='store_true', help="Build with MKLML.")
    parser.add_argument("--use_gemmlowp", action='store_true', help="Build with gemmlowp for quantized gemm.")
    parser.add_argument("--use_featurizers", action='store_true', help="Build with ML Featurizer support.")
    parser.add_argument("--use_ngraph", action='store_true', help="Build with nGraph.")
    parser.add_argument("--use_openvino", nargs="?", const="CPU_FP32",
                        choices=["CPU_FP32","GPU_FP32","GPU_FP16","VAD-M_FP16","MYRIAD_FP16","VAD-F_FP32"], help="Build with OpenVINO for specific hardware.")
    parser.add_argument("--use_dnnlibrary", action='store_true', help="Build with DNNLibrary.")
    parser.add_argument("--use_preinstalled_eigen", action='store_true', help="Use pre-installed eigen.")
    parser.add_argument("--eigen_path", help="Path to pre-installed eigen.")
    parser.add_argument("--use_tvm", action="store_true", help="Build with tvm")
    parser.add_argument("--use_openmp", action='store_true', help="Build with OpenMP.")
    parser.add_argument("--use_llvm", action="store_true", help="Build tvm with llvm")
    parser.add_argument("--enable_msinternal", action="store_true", help="Enable for Microsoft internal builds only.")
    parser.add_argument("--llvm_path", help="Path to llvm dir")
    parser.add_argument("--use_nuphar", action='store_true', help="Build with nuphar")
    parser.add_argument("--use_tensorrt", action='store_true', help="Build with TensorRT")
    parser.add_argument("--tensorrt_home", help="Path to TensorRT installation dir")
    parser.add_argument("--use_full_protobuf", action='store_true', help="Use the full protobuf library")
    parser.add_argument("--disable_contrib_ops", action='store_true', help="Disable contrib ops (reduces binary size)")
    parser.add_argument("--enable_training", action='store_true', help="Enable training in ORT.")
    parser.add_argument("--use_horovod", action='store_true', help="Enable Horovod.")
    parser.add_argument("--skip_onnx_tests", action='store_true', help="Explicitly disable all onnx related tests. Note: Use --skip_tests to skip all tests.")
    parser.add_argument("--skip_winml_tests", action='store_true', help="Explicitly disable all WinML related tests")
    parser.add_argument("--enable_msvc_static_runtime", action='store_true', help="Enable static linking of MSVC runtimes.")
    parser.add_argument("--enable_language_interop_ops", action='store_true', help="Enable operator implemented in language other than cpp")
    parser.add_argument("--cmake_generator", choices=['Visual Studio 15 2017', 'Visual Studio 16 2019'],
                        default='Visual Studio 15 2017', help="Specify the generator that CMake invokes. This is only supported on Windows")
    parser.add_argument("--enable_multi_device_test", action='store_true', help="Test with multi-device. Mostly used for multi-device GPU")
    parser.add_argument("--use_dml", action='store_true', help="Build with DirectML.")
    parser.add_argument("--use_winml", action='store_true', help="Build with WinML.")
    parser.add_argument("--use_telemetry", action='store_true', help="Only official builds can set this flag to enable telemetry.")
    parser.add_argument("--enable_wcos", action='store_true', help="Build for Windows Core OS.")
    parser.add_argument("--enable_lto", action='store_true', help="Enable Link Time Optimization")
    return parser.parse_args()

def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    executable_path = shutil.which(command_or_path)
    if executable_path is None:
        raise BuildError("Failed to resolve executable path for '{}'.".format(command_or_path))
    return os.path.realpath(executable_path)

def is_windows():
    return sys.platform.startswith("win")

def get_linux_distro():
    try:
        with open('/etc/os-release', 'r') as f:
            dist_info = dict(line.strip().split('=', 1) for line in f.readlines())
        return dist_info.get('NAME', '').strip('"'), dist_info.get('VERSION', '').strip('"')
    except:
        return '', ''

def is_ubuntu_1604():
    dist, ver = get_linux_distro()
    return dist == 'Ubuntu' and ver.startswith('16.04')

def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)

def run_subprocess(args, cwd=None, capture=False, dll_path=None, shell=False, env={}):
    log.debug("Running subprocess in '{0}'\n{1}".format(cwd or os.getcwd(), args))
    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path

    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (None, None)
    my_env.update(env)
    completed_process = subprocess.run(args, cwd=cwd, check=True, stdout=stdout, stderr=stderr, env=my_env, shell=shell)
    log.debug("Subprocess completed. Return code=" + str(completed_process.returncode))
    return completed_process

def update_submodules(source_dir):
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=source_dir)
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir)

def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )

def is_sudo():
    return 'SUDO_UID' in os.environ.keys()

def install_apt_package(package):
    have = package in str(run_subprocess(["apt", "list", "--installed", package], capture=True).stdout)
    if not have:
        if is_sudo():
            run_subprocess(['apt-get', 'install', '-y', package])
        else:
            raise BuildError(package + " APT package missing. Please re-run this script using sudo to install.")

def install_ubuntu_deps(args):
    'Check if the necessary Ubuntu dependencies are installed. Not required on docker. Provide help output if missing.'

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
            raise BuildError("Error setting up required APT packages. {}".format(str(e)))

def install_python_deps(numpy_version=""):
    dep_packages = ['setuptools', 'wheel', 'pytest']
    dep_packages.append('numpy=={}'.format(numpy_version) if numpy_version else 'numpy>=1.18.0')
    dep_packages.append('sympy>=1.1')
    dep_packages.append('packaging')
    run_subprocess([sys.executable, '-m', 'pip', 'install', '--trusted-host', 'files.pythonhosted.org'] + dep_packages)

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
    # create a shortcut for test models if there is a 'models' folder in build_dir
    if is_windows():
        src_model_dir = os.path.join(build_dir, 'models')
        for config in configs:
            config_build_dir = get_config_build_dir(build_dir, config)
            os.makedirs(config_build_dir, exist_ok=True)
            dest_model_dir = os.path.join(config_build_dir, 'models')
            if os.path.exists(src_model_dir) and not os.path.exists(dest_model_dir):
                log.debug("creating shortcut %s -> %s"  % (src_model_dir, dest_model_dir))
                run_subprocess(['mklink', '/D', '/J', dest_model_dir, src_model_dir], shell=True)

def generate_build_tree(cmake_path, source_dir, build_dir, cuda_home, cudnn_home, tensorrt_home, path_to_protoc_exe, configs, cmake_extra_defines, args, cmake_extra_args):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    # TODO: fix jemalloc build so it does not conflict with onnxruntime shared lib builds. (e.g. onnxuntime_pybind)
    # for now, disable jemalloc if pybind is also enabled.
    cmake_args = [cmake_path, cmake_dir,
                 "-Donnxruntime_RUN_ONNX_TESTS=" + ("ON" if args.enable_onnx_tests else "OFF"),
                 "-Donnxruntime_BUILD_WINML_TESTS=" + ("OFF" if args.skip_winml_tests else "ON"),
                 "-Donnxruntime_GENERATE_TEST_REPORTS=ON",
                 "-Donnxruntime_DEV_MODE=" + ("OFF" if args.android else "ON"),
                 "-DPYTHON_EXECUTABLE=" + sys.executable,
                 "-Donnxruntime_USE_CUDA=" + ("ON" if args.use_cuda else "OFF"),
                 "-Donnxruntime_CUDNN_HOME=" + (cudnn_home if args.use_cuda else ""),
                 "-Donnxruntime_USE_FEATURIZERS=" + ("ON" if args.use_featurizers else "OFF"),
                 "-Donnxruntime_CUDA_HOME=" + (cuda_home if args.use_cuda else ""),
                 "-Donnxruntime_USE_JEMALLOC=" + ("ON" if args.use_jemalloc else "OFF"),
                 "-Donnxruntime_USE_MIMALLOC_STL_ALLOCATOR=" + ("ON" if args.use_mimalloc == "stl" or args.use_mimalloc == "all" else "OFF"),
                 "-Donnxruntime_USE_MIMALLOC_ARENA_ALLOCATOR=" + ("ON" if args.use_mimalloc == "arena" or args.use_mimalloc == "all" else "OFF"),
                 "-Donnxruntime_ENABLE_PYTHON=" + ("ON" if args.enable_pybind else "OFF"),
                 "-Donnxruntime_BUILD_CSHARP=" + ("ON" if args.build_csharp else "OFF"),
                 "-Donnxruntime_BUILD_JAVA=" + ("ON" if args.build_java else "OFF"),
                 "-Donnxruntime_BUILD_SHARED_LIB=" + ("ON" if args.build_shared_lib else "OFF"),
                 "-Donnxruntime_USE_EIGEN_FOR_BLAS=" + ("OFF" if args.use_openblas else "ON"),
                 "-Donnxruntime_USE_OPENBLAS=" + ("ON" if args.use_openblas else "OFF"),
                 "-Donnxruntime_USE_DNNL=" + ("ON" if args.use_dnnl else "OFF"),
                 "-Donnxruntime_USE_MKLML=" + ("ON" if args.use_mklml else "OFF"),
                 "-Donnxruntime_USE_GEMMLOWP=" + ("ON" if args.use_gemmlowp else "OFF"),
                 "-Donnxruntime_USE_NGRAPH=" + ("ON" if args.use_ngraph else "OFF"),
                 "-Donnxruntime_USE_OPENVINO=" + ("ON" if args.use_openvino else "OFF"),
                 "-Donnxruntime_USE_OPENVINO_MYRIAD=" + ("ON" if args.use_openvino == "MYRIAD_FP16" else "OFF"),
                 "-Donnxruntime_USE_OPENVINO_GPU_FP32=" + ("ON" if args.use_openvino == "GPU_FP32" else "OFF"),
                 "-Donnxruntime_USE_OPENVINO_GPU_FP16=" + ("ON" if args.use_openvino == "GPU_FP16" else "OFF"),
                 "-Donnxruntime_USE_OPENVINO_CPU_FP32=" + ("ON" if args.use_openvino == "CPU_FP32" else "OFF"),
                 "-Donnxruntime_USE_OPENVINO_VAD_M=" + ("ON" if args.use_openvino == "VAD-M_FP16" else "OFF"),
                 "-Donnxruntime_USE_OPENVINO_VAD_F=" + ("ON" if args.use_openvino == "VAD-F_FP32" else "OFF"),
                 "-Donnxruntime_USE_NNAPI=" + ("ON" if args.use_dnnlibrary else "OFF"),
                 "-Donnxruntime_USE_OPENMP=" + ("ON" if args.use_openmp and not args.use_dnnlibrary and not args.use_mklml and not args.use_ngraph else "OFF"),
                 "-Donnxruntime_USE_TVM=" + ("ON" if args.use_tvm else "OFF"),
                 "-Donnxruntime_USE_LLVM=" + ("ON" if args.use_llvm else "OFF"),
                 "-Donnxruntime_ENABLE_MICROSOFT_INTERNAL=" + ("ON" if args.enable_msinternal else "OFF"),
                 "-Donnxruntime_USE_NUPHAR=" + ("ON" if args.use_nuphar else "OFF"),
                 "-Donnxruntime_USE_TENSORRT=" + ("ON" if args.use_tensorrt else "OFF"),
                 "-Donnxruntime_TENSORRT_HOME=" + (tensorrt_home if args.use_tensorrt else ""),
                  # By default - we currently support only cross compiling for ARM/ARM64 (no native compilation supported through this script)
                 "-Donnxruntime_CROSS_COMPILING=" + ("ON" if args.arm64 or args.arm else "OFF"),
                 "-Donnxruntime_DISABLE_CONTRIB_OPS=" + ("ON" if args.disable_contrib_ops else "OFF"),
                 "-Donnxruntime_MSVC_STATIC_RUNTIME=" + ("ON" if args.enable_msvc_static_runtime else "OFF"),
                 # enable pyop if it is nightly build
                 "-Donnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=" + ("ON" if args.enable_language_interop_ops or (args.config != 'Debug' and bool(os.getenv('NIGHTLY_BUILD') == '1')) else "OFF"),
                 "-Donnxruntime_USE_DML=" + ("ON" if args.use_dml else "OFF"),
                 "-Donnxruntime_USE_WINML=" + ("ON" if args.use_winml else "OFF"),
                 "-Donnxruntime_USE_TELEMETRY=" + ("ON" if args.use_telemetry else "OFF"),
                 "-Donnxruntime_ENABLE_WCOS=" + ("ON" if args.enable_wcos else "OFF"),
                 "-Donnxruntime_ENABLE_LTO=" + ("ON" if args.enable_lto else "OFF"),
                 # Training related flags
                 "-Donnxruntime_ENABLE_TRAINING=" + ("ON" if args.enable_training else "OFF"),
                 "-Donnxruntime_ENABLE_TRAINING_E2E_TESTS=" + ("ON" if args.enable_training_e2e_tests else "OFF"),
                 "-Donnxruntime_USE_HOROVOD=" + ("ON" if args.use_horovod else "OFF"),
                 ]

    # temp turn on only for linux gpu build
    if not is_windows():
        if args.use_cuda:
            if "-Donnxruntime_USE_HOROVOD=OFF" in cmake_args:
               cmake_args.remove("-Donnxruntime_USE_HOROVOD=OFF")
            cmake_args += [
                "-Donnxruntime_USE_HOROVOD=ON",
                "-Donnxruntime_USE_FULL_PROTOBUF=ON"]

    # nGraph and TensorRT providers currently only supports full_protobuf option.
    if args.use_full_protobuf or args.use_ngraph or args.use_tensorrt or args.gen_doc:
       cmake_args += ["-Donnxruntime_USE_FULL_PROTOBUF=ON", "-DProtobuf_USE_STATIC_LIBS=ON"]

    if args.use_llvm:
        cmake_args += ["-DLLVM_DIR=%s" % args.llvm_path]

    if args.use_cuda and not is_windows():
        nvml_stub_path = cuda_home + "/lib64/stubs"
        cmake_args += ["-DCUDA_CUDA_LIBRARY=" + nvml_stub_path]

    if args.use_preinstalled_eigen:
        cmake_args += ["-Donnxruntime_USE_PREINSTALLED_EIGEN=ON",
                       "-Deigen_SOURCE_PATH=" + args.eigen_path]

    if args.android:
        cmake_args += ["-DCMAKE_TOOLCHAIN_FILE=" + args.android_ndk_path + "/build/cmake/android.toolchain.cmake",
                "-DANDROID_PLATFORM=android-" + str(args.android_api),
                "-DANDROID_ABI=" + str(args.android_abi)]

    if path_to_protoc_exe:
        cmake_args += ["-DONNX_CUSTOM_PROTOC_EXECUTABLE=%s" % path_to_protoc_exe]

    if args.gen_doc:
        cmake_args += ["-Donnxruntime_PYBIND_EXPORT_OPSCHEMA=ON"]
    else:
        cmake_args += ["-Donnxruntime_PYBIND_EXPORT_OPSCHEMA=OFF"]

    if args.training_e2e_test_data_path is not None:
        cmake_args += ["-Donnxruntime_TRAINING_E2E_TEST_DATA_ROOT={}".format(
            os.path.abspath(args.training_e2e_test_data_path))]

    cmake_args += ["-D{}".format(define) for define in cmake_extra_defines]

    if is_windows():
        cmake_args += cmake_extra_args

    # ADO pipelines will store the pipeline build number (e.g. 191101-2300.1.master) and
    # source version in environment variables. If present, use these values to define the
    # WinML/ORT DLL versions.
    build_number = os.getenv('Build_BuildNumber')
    source_version = os.getenv('Build_SourceVersion')
    if build_number and source_version:
        build_matches = re.match(r"^(\d\d)(\d\d)(\d\d)-(\d\d)(\d\d)\.(\d)\.(\S+)$", build_number)
        if build_matches:
            YY = build_matches.group(1)
            MM = build_matches.group(2)
            DD = build_matches.group(3)
            HH = build_matches.group(4)

            # Get ORT major and minor number
            with open(os.path.join(source_dir, 'VERSION_NUMBER')) as f:
                first_line = f.readline()
                ort_version_matches = re.match(r"(\d+).(\d+)", first_line)
                if not ort_version_matches:
                    raise BuildError("Couldn't read version from VERSION_FILE")
                ort_major = ort_version_matches.group(1)
                ort_minor = ort_version_matches.group(2)
                # Example (BuildNumber: 191101-2300.1.master, SourceVersion: 0bce7ae6755c792eda558e5d27ded701707dc404)
                # MajorPart = 1
                # MinorPart = 0
                # BuildPart = 1911
                # PrivatePart = 123
                # String = 191101-2300.1.master.0bce7ae
                cmake_args += ["-DVERSION_MAJOR_PART={}".format(ort_major),
                            "-DVERSION_MINOR_PART={}".format(ort_minor),
                            "-DVERSION_BUILD_PART={}{}".format(YY, MM),
                            "-DVERSION_PRIVATE_PART={}{}".format(DD, HH),
                            "-DVERSION_STRING={}.{}.{}.{}".format(ort_major, ort_minor, build_number, source_version[0:7])]

    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)
        os.makedirs(config_build_dir, exist_ok=True)

        if args.use_tvm:
            os.environ["PATH"] = os.path.join(config_build_dir, "external", "tvm", config) + os.pathsep + os.environ["PATH"]

        run_subprocess(cmake_args  + ["-Donnxruntime_ENABLE_MEMLEAK_CHECKER=" + ("ON" if config.lower() == 'debug' and not args.use_tvm and not args.use_ngraph and not args.enable_msvc_static_runtime else "OFF"), "-DCMAKE_BUILD_TYPE={}".format(config)], cwd=config_build_dir)


def clean_targets(cmake_path, build_dir, configs):
    for config in configs:
        log.info("Cleaning targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path,
                    "--build", build_dir2,
                    "--config", config,
                    "--target", "clean"]

        run_subprocess(cmd_args)

def build_targets(args, cmake_path, build_dir, configs, parallel):
    for config in configs:
        log.info("Building targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path,
                    "--build", build_dir2,
                    "--config", config]

        build_tool_args = []
        if parallel:
            num_cores = str(multiprocessing.cpu_count())
            if is_windows():
                build_tool_args += [
                    "/maxcpucount:" + num_cores,
                    # if nodeReuse is true, msbuild processes will stay around for a bit after the build completes
                    "/nodeReuse:False",
                    ]
            else:
                build_tool_args += ["-j" + num_cores]

        if (build_tool_args):
            cmd_args += [ "--" ]
            cmd_args += build_tool_args

        env = {}
        if args.android:
            env['ANDROID_SDK_ROOT']=args.android_sdk_path

        run_subprocess(cmd_args, env=env)

def add_dir_if_exists(dir, dir_list):
    if (os.path.isdir(dir)):
        dir_list.append(dir)

def setup_cuda_vars(args):

    cuda_home = ""
    cudnn_home = ""

    if (args.use_cuda):
        cuda_home = args.cuda_home if args.cuda_home else os.getenv("CUDA_HOME")
        cudnn_home = args.cudnn_home if args.cudnn_home else os.getenv("CUDNN_HOME")

        cuda_home_valid = (cuda_home != None and os.path.exists(cuda_home))
        cudnn_home_valid = (cudnn_home != None and os.path.exists(cudnn_home))

        if (not cuda_home_valid or not cudnn_home_valid):
            raise BuildError("cuda_home and cudnn_home paths must be specified and valid.",
                             "cuda_home='{}' valid={}. cudnn_home='{}' valid={}"
                             .format(cuda_home, cuda_home_valid, cudnn_home, cudnn_home_valid))

        if (is_windows()):
            # Validate that the cudnn_home is pointing at the right level
            if (not os.path.exists(os.path.join(cudnn_home, "bin"))):
                raise BuildError("cudnn_home path should include the 'cuda' folder, and must contain the CUDNN 'bin' directory.",
                                 "cudnn_home='{}'".format(cudnn_home))

            os.environ["CUDA_PATH"] = cuda_home
            os.environ["CUDA_TOOLKIT_ROOT_DIR"] = cuda_home

            cuda_bin_path = os.path.join(cuda_home, 'bin')
            os.environ["CUDA_BIN_PATH"] = cuda_bin_path
            os.environ["PATH"] += os.pathsep + cuda_bin_path + os.pathsep + os.path.join(cudnn_home, 'bin')
            # Add version specific CUDA_PATH_Vx_y value as the Visual Studio build files require that
            version_file = os.path.join(cuda_home, 'version.txt')
            if not os.path.exists(version_file):
                raise BuildError("No version file found in CUDA install directory. Looked for " + version_file)

            cuda_major_version = "unknown"

            with open(version_file) as f:
                # First line of version file should have something like 'CUDA Version 9.2.148'
                first_line = f.readline()
                m = re.match(r"CUDA Version (\d+).(\d+)", first_line)
                if not m:
                    raise BuildError("Couldn't read version from first line of " + version_file)

                cuda_major_version = m.group(1)
                minor = m.group(2)
                os.environ["CUDA_PATH_V{}_{}".format(cuda_major_version, minor)] = cuda_home

            vc_ver_str = os.getenv("VCToolsVersion") or ""
            vc_ver = vc_ver_str.split(".")
            if len(vc_ver) != 3:
                log.warning("Unable to automatically verify VS 2017 toolset is compatible with CUDA. Will attempt to use.")
                log.warning("Failed to get valid Visual C++ Tools version from VCToolsVersion environment variable value of '" + vc_ver_str + "'")
                log.warning("VCToolsVersion is set in a VS 2017 Developer Command shell, or by running \"%VS2017INSTALLDIR%\\VC\\Auxiliary\\Build\\vcvars64.bat\"")
                log.warning("See build.md in the root ONNXRuntime directory for instructions on installing the Visual C++ 2017 14.11 toolset if needed.")

            elif cuda_major_version == "9" and vc_ver[0] == "14" and int(vc_ver[1]) > 11:
                raise BuildError("Visual C++ Tools version not supported by CUDA v9. You must setup the environment to use the 14.11 toolset.",
                                 "Current version is {}. CUDA 9.2 requires version 14.11.*".format(vc_ver_str),
                                 "If necessary manually install the 14.11 toolset using the Visual Studio 2017 updater.",
                                 "See 'Windows CUDA Build' in build.md in the root directory of this repository.")

            # TODO: check if cuda_version >=10.1, when cuda is enabled and VS version >=2019

    return cuda_home, cudnn_home

def setup_tensorrt_vars(args):

    tensorrt_home = ""

    if (args.use_tensorrt):
        tensorrt_home = args.tensorrt_home if args.tensorrt_home else os.getenv("TENSORRT_HOME")

        tensorrt_home_valid = (tensorrt_home != None and os.path.exists(tensorrt_home))

        if (not tensorrt_home_valid):
            raise BuildError("tensorrt_home paths must be specified and valid.",
                             "tensorrt_home='{}' valid={}."
                             .format(tensorrt_home, tensorrt_home_valid))

        # Set maximum workspace size in byte for TensorRT (1GB = 1073741824 bytes)
        os.environ["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = "1073741824"

        # Set maximum number of iterations to detect unsupported nodes and partition the models for TensorRT
        os.environ["ORT_TENSORRT_MAX_PARTITION_ITERATIONS"] = "1000"

        # Set minimum subgraph node size in graph partitioning for TensorRT
        os.environ["ORT_TENSORRT_MIN_SUBGRAPH_SIZE"] = "1"

        # Set FP16 flag
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"

    return tensorrt_home

def setup_dml_build(args, cmake_path, build_dir, configs):
    if (args.use_dml):
        for config in configs:
            # Run the RESTORE_PACKAGES target to perform the initial NuGet setup
            cmd_args = [cmake_path,
                        "--build", get_config_build_dir(build_dir, config),
                        "--config", config,
                        "--target", "RESTORE_PACKAGES"]
            run_subprocess(cmd_args)


def adb_push(source_dir, src, dest, **kwargs):
    return run_subprocess([os.path.join(source_dir, 'tools', 'ci_build', 'github', 'android', 'adb-push.sh'), src, dest], **kwargs)

def adb_shell(*args, **kwargs):
    return run_subprocess(['adb', 'shell', *args], **kwargs)

def run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs, enable_tvm = False, enable_tensorrt = False):
    for config in configs:
        log.info("Running tests for %s configuration", config)
        cwd = get_config_build_dir(build_dir, config)
        android_x86_64 = args.android_abi == 'x86_64'
        if android_x86_64:
            run_subprocess(os.path.join(source_dir, 'tools', 'ci_build', 'github', 'android', 'start_android_emulator.sh'))
            adb_push(source_dir, 'testdata', '/data/local/tmp/', cwd=cwd)
            adb_push(source_dir, os.path.join(source_dir, 'cmake', 'external', 'onnx', 'onnx', 'backend', 'test'), '/data/local/tmp/', cwd=cwd)
            adb_push(source_dir, 'onnxruntime_test_all', '/data/local/tmp/', cwd=cwd)
            adb_push(source_dir, 'onnx_test_runner', '/data/local/tmp/', cwd=cwd)
            adb_shell('cd /data/local/tmp && /data/local/tmp/onnxruntime_test_all')
            if args.use_dnnlibrary:
                adb_shell('cd /data/local/tmp && /data/local/tmp/onnx_test_runner -e nnapi /data/local/tmp/test')
            else:
                adb_shell('cd /data/local/tmp && /data/local/tmp/onnx_test_runner /data/local/tmp/test')
            continue
        if enable_tvm:
          dll_path = os.path.join(build_dir, config, "external", "tvm", config)
        elif enable_tensorrt:
          dll_path = os.path.join(args.tensorrt_home, 'lib')
        else:
          dll_path = None
        if ctest_path is None:
            #Get the "Google Test Adapter" for vstest
            if not os.path.exists(os.path.join(cwd,'googletestadapter.0.17.1')):
                run_subprocess(['nuget.exe', 'restore', os.path.join(source_dir, 'packages.config'), '-ConfigFile',os.path.join(source_dir, 'NuGet.config'),'-PackagesDirectory',cwd])
            cwd2=os.path.join(cwd,config)
            executables = ['onnxruntime_test_all.exe']
            if args.build_shared_lib:
                executables.append('onnxruntime_shared_lib_test.exe')
            run_subprocess(['vstest.console.exe', '--parallel', '--TestAdapterPath:..\\googletestadapter.0.17.1\\build\\_common', '/Logger:trx','/Enablecodecoverage','/Platform:x64',"/Settings:%s" % os.path.join(source_dir, 'cmake\\codeconv.runsettings')] + executables,
                       cwd=cwd2, dll_path=dll_path)
        else:
            ctest_cmd = [ctest_path, "--build-config", config, "--verbose"]
            if args.ctest_label_regex is not None:
                ctest_cmd += ["--label-regex", args.ctest_label_regex]

            run_subprocess(ctest_cmd, cwd=cwd, dll_path=dll_path)

        if args.enable_pybind:
            # Disable python tests for TensorRT because many tests are not supported yet
            if enable_tensorrt :
                return
            if is_windows():
                cwd = os.path.join(cwd, config)

            run_subprocess([sys.executable, 'onnxruntime_test_python.py'], cwd=cwd, dll_path=dll_path)

            try:
                import onnx
                import scipy  # gen_test_models.py used by onnx_test has a dependency on scipy
                onnx_test = True
            except ImportError as error:
                log.exception(error)
                log.warning("onnx or scipy is not installed. The ONNX tests will be skipped.")
                onnx_test = False

            if onnx_test:
                run_subprocess([sys.executable, 'onnxruntime_test_python_backend.py'], cwd=cwd, dll_path=dll_path)
                run_subprocess([sys.executable, os.path.join(source_dir,'onnxruntime','test','onnx','gen_test_models.py'),
                                '--output_dir','test_models'], cwd=cwd)
                if not args.skip_onnx_tests:
                    run_subprocess([os.path.join(cwd,'onnx_test_runner'), 'test_models'], cwd=cwd)
                if config != 'Debug':
                    run_subprocess([sys.executable, 'onnx_backend_test_series.py'], cwd=cwd, dll_path=dll_path)

            if not args.skip_keras_test:
                try:
                    import onnxmltools
                    import keras
                    onnxml_test = True
                except ImportError:
                    log.warning("onnxmltools and keras are not installed. The keras tests will be skipped.")
                    onnxml_test = False
                if onnxml_test:
                    run_subprocess([sys.executable, 'onnxruntime_test_python_keras.py'], cwd=cwd, dll_path=dll_path)

def run_onnx_tests(build_dir, configs, onnx_test_data_dir, provider, enable_multi_device_test, enable_parallel_executor_test, num_parallel_models, num_parallel_tests=0):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
           exe = os.path.join(cwd, config, 'onnx_test_runner')
           model_dir = os.path.join(cwd, "models")
        else:
           exe = os.path.join(cwd, 'onnx_test_runner')
           model_dir = os.path.join(build_dir, "models")
        cmd = []
        if provider:
          cmd += ["-e", provider]

        if num_parallel_tests != 0:
          cmd += ['-c', str(num_parallel_tests)]

        if num_parallel_models > 0:
          cmd += ["-j", str(num_parallel_models)]

        if enable_multi_device_test:
          cmd += ['-d', '1']

        if config != 'Debug' and os.path.exists(model_dir):
          cmd.append(model_dir)
        if os.path.exists(onnx_test_data_dir):
          cmd.append(onnx_test_data_dir)

        if config == 'Debug' and provider == 'nuphar':
          return

        run_subprocess([exe] + cmd, cwd=cwd)
        if enable_parallel_executor_test:
          run_subprocess([exe,'-x'] + cmd, cwd=cwd)

# tensorrt function to run onnx test and model test.
def tensorrt_run_onnx_tests(args, build_dir, configs, onnx_test_data_dir, provider, num_parallel_models, num_parallel_tests=0):
    dll_path = os.path.join(args.tensorrt_home, 'lib')
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
           exe = os.path.join(cwd, config, 'onnx_test_runner')
           model_dir = os.path.join(cwd, "models")
        else:
           exe = os.path.join(cwd, 'onnx_test_runner')
           model_dir = os.path.join(build_dir, "models")

        cmd_base = []
        if provider:
          cmd_base += ["-e", provider]

        if num_parallel_tests != 0:
          cmd_base += ['-c', str(num_parallel_tests)]

        if num_parallel_models > 0:
          cmd_base += ["-j", str(num_parallel_models)]

        #onnx test
        if os.path.exists(onnx_test_data_dir):
          onnx_test_cmd = cmd_base + [onnx_test_data_dir]
          run_subprocess([exe] + onnx_test_cmd, cwd=cwd, dll_path=dll_path)

        # model test
        # TensorRT can run most of the model tests, but only part of them is enabled here to save CI build time.
        if config != 'Debug' and os.path.exists(model_dir):
          model_dir_opset8 = os.path.join(model_dir, "opset8")
          model_dir_opset8 = glob.glob(os.path.join(model_dir_opset8, "test_*"))
          model_dir_opset10 = os.path.join(model_dir, "opset10")
          model_dir_opset10 = glob.glob(os.path.join(model_dir_opset10, "tf_inception_v1"))
          for dir_path in itertools.chain(model_dir_opset8, model_dir_opset10):
            model_test_cmd = cmd_base + [dir_path]
            run_subprocess([exe] + model_test_cmd, cwd=cwd, dll_path=dll_path)

# dnnl temporary function for running onnx tests and model tests separately.
def dnnl_run_onnx_tests(build_dir, configs, onnx_test_data_dir):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
           exe = os.path.join(cwd, config, 'onnx_test_runner')
           model_dir = os.path.join(cwd, "models")
        else:
           exe = os.path.join(cwd, 'onnx_test_runner')
           model_dir = os.path.join(build_dir, "models")
        cmd_base = ['-e', 'dnnl', '-c', '1', '-j', '1']
        if os.path.exists(onnx_test_data_dir):
          onnxdata_cmd = cmd_base + [onnx_test_data_dir]
          # /data/onnx
          run_subprocess([exe] + onnxdata_cmd, cwd=cwd)
          run_subprocess([exe,'-x'] + onnxdata_cmd, cwd=cwd)

        if config != 'Debug' and os.path.exists(model_dir):
          opset7_model_dir = os.path.join(model_dir, 'opset7')
          opset7_cmd = cmd_base + [opset7_model_dir]
          opset8_model_dir = os.path.join(model_dir, 'opset8')
          opset8_cmd = cmd_base + [opset8_model_dir]
          opset9_model_dir = os.path.join(model_dir, 'opset9')
          opset9_cmd = cmd_base + [opset9_model_dir]
          opset10_model_dir = os.path.join(model_dir, 'opset10')
          opset10_cmd = cmd_base + [opset10_model_dir]
          run_subprocess([exe] + opset7_cmd, cwd=cwd)
          run_subprocess([exe] + opset8_cmd, cwd=cwd)
          run_subprocess([exe] + opset9_cmd, cwd=cwd)
          run_subprocess([exe] + opset10_cmd, cwd=cwd)

          # temporarily disable -x invocations on Windows as they
          # are causing instability in CI
          if not is_windows():
            run_subprocess([exe, '-x'] + opset7_cmd, cwd=cwd)
            run_subprocess([exe, '-x'] + opset8_cmd, cwd=cwd)
            run_subprocess([exe, '-x'] + opset9_cmd, cwd=cwd)
            run_subprocess([exe, '-x'] + opset10_cmd, cwd=cwd)


# nuphar temporary function for running python tests separately as it requires ONNX 1.5.0
def nuphar_run_python_tests(build_dir, configs):
    for config in configs:
        if config == 'Debug':
            continue
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
            cwd = os.path.join(cwd, config)
        dll_path = os.path.join(build_dir, config, "external", "tvm", config)
        # install onnx for shape inference in testing Nuphar scripts
        # this needs to happen after onnx_test_data preparation which uses onnx 1.3.0
        run_subprocess([sys.executable, '-m', 'pip', 'install', '--user', 'onnx==1.5.0'])
        run_subprocess([sys.executable, 'onnxruntime_test_python_nuphar.py'], cwd=cwd, dll_path=dll_path)


def build_python_wheel(source_dir, build_dir, configs, use_cuda, use_ngraph, use_dnnl, use_tensorrt, use_openvino, use_nuphar, nightly_build = False):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
            cwd = os.path.join(cwd, config)
        args = [sys.executable, os.path.join(source_dir, 'setup.py'), 'bdist_wheel']
        if nightly_build:
            args.append('--nightly_build')
        if use_tensorrt:
            args.append('--use_tensorrt')
        elif use_cuda:
            args.append('--use_cuda')
        elif use_ngraph:
            args.append('--use_ngraph')
        elif use_dnnl:
            args.append('--use_dnnl')
        elif use_openvino:
            args.append('--use_openvino')
        elif use_nuphar:
            args.append('--use_nuphar')
        run_subprocess(args, cwd=cwd)

def build_protoc_for_host(cmake_path, source_dir, build_dir, args):
    if (args.arm or args.arm64) and not is_windows():
        raise BuildError('Currently only support building protoc for Windows host while cross-compiling for ARM/ARM64 arch')

    log.info("Building protoc for host to be used in cross-compiled build process")
    protoc_build_dir = os.path.join(os.getcwd(), build_dir, 'host_protoc')
    os.makedirs(protoc_build_dir, exist_ok=True)
    # Generate step
    cmd_args = [cmake_path,
                os.path.join(source_dir, 'cmake', 'external', 'protobuf', 'cmake'),
                '-Dprotobuf_BUILD_TESTS=OFF',
                '-Dprotobuf_WITH_ZLIB_DEFAULT=OFF',
                '-Dprotobuf_BUILD_SHARED_LIBS=OFF']
    if is_windows():
        cmd_args += ['-T',
                'host=x64',
                '-G',
                args.cmake_generator]
    run_subprocess(cmd_args, cwd= protoc_build_dir)
    # Build step
    cmd_args = [cmake_path,
                "--build", protoc_build_dir,
                "--config", "Release",
                "--target", "protoc"]
    run_subprocess(cmd_args)

    # Absolute protoc path is needed for cmake
    expected_protoc_path = os.path.join(protoc_build_dir, 'Release', 'protoc.exe') if is_windows() else os.path.join(protoc_build_dir, 'protoc')
    if not os.path.exists(expected_protoc_path):
        raise BuildError("Couldn't build protoc for host. Failing build.")

    return expected_protoc_path

def generate_documentation(source_dir, build_dir, configs):
    operator_doc_path = os.path.join(source_dir, 'docs', 'ContribOperators.md')
    opkernel_doc_path = os.path.join(source_dir, 'docs', 'OperatorKernels.md')
    for config in configs:
        #copy the gen_doc.py
        shutil.copy(os.path.join(source_dir,'tools','python','gen_doc.py'),
                    os.path.join(build_dir,config, config))
        shutil.copy(os.path.join(source_dir,'tools','python','gen_opkernel_doc.py'),
                    os.path.join(build_dir,config, config))

        run_subprocess([
                        sys.executable,
                        'gen_doc.py',
                        '--output_path', operator_doc_path
                    ],
                    cwd = os.path.join(build_dir,config, config))

        run_subprocess([
                        sys.executable,
                        'gen_opkernel_doc.py',
                        '--output_path', opkernel_doc_path
                    ],
                    cwd = os.path.join(build_dir,config, config))

    docdiff = ''
    try:
        docdiff = subprocess.check_output(['git', 'diff', opkernel_doc_path])
    except subprocess.CalledProcessError:
        print('git diff returned non-zero error code')
    if len(docdiff) > 0:
        # Show warning instead of throwing exception, because it is dependent on build configuration for including execution propviders
        log.warning('The updated opkernel document file '+str(opkernel_doc_path)+' is different from the checked in version. Consider regenrating the file with CPU, DNNL and CUDA providers enabled.')
        log.debug('diff:\n'+str(docdiff))

    docdiff = ''
    try:
        docdiff = subprocess.check_output(['git', 'diff', operator_doc_path])
    except subprocess.CalledProcessError:
        print('git diff returned non-zero error code')
    if len(docdiff) > 0:
        raise BuildError('The updated operator document file '+str(operator_doc_path)+' must be checked in.\n diff:\n'+str(docdiff))


def main():
    args = parse_arguments()

    cmake_extra_defines = args.cmake_extra_defines if args.cmake_extra_defines else []

    cross_compiling = args.arm or args.arm64 or args.android

    # if there was no explicit argument saying what to do, default to update, build and test (for native builds).
    if (args.update == False and args.clean == False and args.build == False and args.test == False):
        log.debug("Defaulting to running update, build [and test for native builds].")
        args.update = True
        args.build = True
        if cross_compiling:
            args.test = args.android_abi == 'x86_64'
        else:
            args.test = True

    if args.skip_tests:
        args.test = False

    if args.use_tensorrt:
        args.use_cuda = True

    if args.build_wheel:
        args.enable_pybind = True

    if args.build_csharp or args.build_java:
        args.build_shared_lib = True

    # Disabling unit tests for VAD-F as FPGA only supports models with NCHW layout
    if args.use_openvino == "VAD-F_FP32":
        args.test = False

    configs = set(args.config)

    # setup paths and directories
    cmake_path = resolve_executable_path(args.cmake_path)
    ctest_path = None if args.use_vstest else resolve_executable_path(args.ctest_path)
    build_dir = args.build_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))

    # if using cuda, setup cuda paths and env vars
    cuda_home, cudnn_home = setup_cuda_vars(args)

    # if using tensorrt, setup tensorrt paths
    tensorrt_home = setup_tensorrt_vars(args)

    os.makedirs(build_dir, exist_ok=True)

    log.info("Build started")
    if (args.update):
        cmake_extra_args = []
        path_to_protoc_exe = None
        if(is_windows()):
          if (args.x86):
            cmake_extra_args = ['-A','Win32','-T','host=x64','-G', args.cmake_generator]
          elif (args.arm or args.arm64):
            # Cross-compiling for ARM(64) architecture
            # First build protoc for host to use during cross-compilation
            path_to_protoc_exe = build_protoc_for_host(cmake_path, source_dir, build_dir, args)
            if args.arm:
                cmake_extra_args = ['-A', 'ARM']
            else:
                cmake_extra_args = ['-A', 'ARM64']
            cmake_extra_args += ['-G', args.cmake_generator]
            # Cannot test on host build machine for cross-compiled builds (Override any user-defined behaviour for test if any)
            if args.test:
                log.info("Cannot test on host build machine for cross-compiled ARM(64) builds. Will skip test running after build.")
                args.test = False
          else:
            if args.msvc_toolset == '14.16' and args.cmake_generator == 'Visual Studio 16 2019':
                #CUDA 10.0 requires _MSC_VER >= 1700 and _MSC_VER < 1920, aka Visual Studio version in [2012, 2019)
                #In VS2019, we have to use Side-by-side minor version MSVC toolsets from Visual Studio 2017
                #14.16 is MSVC version
                #141 is MSVC Toolset Version
                #Cuda VS extension should be installed to C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Microsoft\VC\v160\BuildCustomizations
                toolset = 'v141,host=x64,version=' + args.msvc_toolset
            elif args.msvc_toolset:
                toolset = 'host=x64,version=' + args.msvc_toolset
            else:
                toolset = 'host=x64'
            if (args.cuda_version):
                toolset += ',cuda=' + args.cuda_version

            cmake_extra_args = ['-A','x64','-T', toolset, '-G', args.cmake_generator]
        if args.android:
            # Cross-compiling for Android
            path_to_protoc_exe = build_protoc_for_host(cmake_path, source_dir, build_dir, args)
        if is_ubuntu_1604():
            if (args.arm or args.arm64):
                raise BuildError("Only Windows ARM(64) cross-compiled builds supported currently through this script")
            install_ubuntu_deps(args)
            if not is_docker():
                install_python_deps()
        if (args.enable_pybind and is_windows()):
            install_python_deps(args.numpy_version)
        if (not args.skip_submodule_sync):
            update_submodules(source_dir)

        if args.enable_onnx_tests:
            setup_test_data(build_dir, configs)

        if args.path_to_protoc_exe:
            path_to_protoc_exe = args.path_to_protoc_exe

        generate_build_tree(cmake_path, source_dir, build_dir, cuda_home, cudnn_home, tensorrt_home, path_to_protoc_exe, configs, cmake_extra_defines,
                            args, cmake_extra_args)

    if (args.clean):
        clean_targets(cmake_path, build_dir, configs)

    # if using DML, perform initial nuget package restore
    setup_dml_build(args, cmake_path, build_dir, configs)

    if (args.build):
        build_targets(args, cmake_path, build_dir, configs, args.parallel)

    if args.test :
        run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs,
                              args.use_tvm, args.use_tensorrt)
        # run the onnx model tests if requested explicitly.
        if args.enable_onnx_tests and not args.skip_onnx_tests:
            # directory from ONNX submodule with ONNX test data
            onnx_test_data_dir = '/data/onnx'
            if is_windows() or not os.path.exists(onnx_test_data_dir):
                onnx_test_data_dir = os.path.join(source_dir, "cmake", "external", "onnx", "onnx", "backend", "test", "data")

            if args.use_tensorrt:
              # Disable some onnx unit tests that TensorRT doesn't supported yet
              if not is_windows():
                trt_onnx_test_data_dir = os.path.join(source_dir, "cmake", "external", "onnx", "onnx", "backend", "test", "data", "simple")
              else:
                trt_onnx_test_data_dir = ""
              tensorrt_run_onnx_tests(args, build_dir, configs, trt_onnx_test_data_dir, "tensorrt",1)

            if args.use_cuda and not args.use_tensorrt:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'cuda', args.enable_multi_device_test, False, 2)

            if args.use_ngraph:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'ngraph', args.enable_multi_device_test, True, 1)

            if args.use_openvino:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'openvino', args.enable_multi_device_test, False, 1, 1)
              # TODO: parallel executor test fails on MacOS
            if args.use_nuphar:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'nuphar', args.enable_multi_device_test, False, 1, 1)

            if args.use_dml:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'dml', args.enable_multi_device_test, False, 1)

            # Run some models are disabled to keep memory utilization under control
            if args.use_dnnl:
              dnnl_run_onnx_tests(build_dir, configs, onnx_test_data_dir)

            if args.use_tensorrt:
              tensorrt_run_onnx_tests(args, build_dir, configs, onnx_test_data_dir, None,1)
            else:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, None, args.enable_multi_device_test, False,
                1 if args.x86 or platform.system() == 'Darwin' else 0,
                1 if args.x86 or platform.system() == 'Darwin' else 0)

        # run nuphar python tests last, as it installs ONNX 1.5.0
        if args.enable_pybind and not args.skip_onnx_tests and args.use_nuphar:
            nuphar_run_python_tests(build_dir, configs)

    if args.build:
        if args.build_wheel:
            nightly_build = bool(os.getenv('NIGHTLY_BUILD') == '1')
            build_python_wheel(source_dir, build_dir, configs, args.use_cuda, args.use_ngraph, args.use_dnnl, args.use_tensorrt, args.use_openvino, args.use_nuphar, nightly_build)

    if args.gen_doc and (args.build or args.test):
        generate_documentation(source_dir, build_dir, configs)

    log.info("Build complete")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseError as e:
        log.error(str(e))
        sys.exit(1)
