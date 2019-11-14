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
import warnings
import hashlib
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

    # enable ONNX tests
    parser.add_argument("--enable_onnx_tests", action='store_true',
                        help='''When running the Test phase, run onnx_test_running against available test data directories.''')
    parser.add_argument("--path_to_protoc_exe", help="Path to protoc exe. ")
    parser.add_argument("--download_test_data", action="store_true",
                        help='''Downloads test data without running the tests''')
    parser.add_argument("--test_data_url", help="Test data URL.")
    parser.add_argument("--test_data_checksum", help="Test data checksum (MD5 digest).")

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


    # Build a shared lib
    parser.add_argument("--build_shared_lib", action='store_true', help="Build a shared library for the ONNXRuntime.")

    # Build ONNX Runtime server
    parser.add_argument("--build_server", action='store_true', help="Build server application for the ONNXRuntime.")
    parser.add_argument("--enable_server_tests", action='store_true', help="Run server application tests.")
    parser.add_argument("--enable_server_model_tests", action='store_true', help="Run server model tests.")

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
    parser.add_argument("--android_ndk_path", default="", help="Path to the Android NDK")

    # Arguments needed by CI
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake program.")
    parser.add_argument("--ctest_path", default="ctest", help="Path to the CTest program.")
    parser.add_argument("--skip_submodule_sync", action='store_true', help="Don't do a 'git submodule update'. Makes the Update phase faster.")

    parser.add_argument("--use_jemalloc", action='store_true', help="Use jemalloc.")
    parser.add_argument("--use_mimalloc", action='store_true', help="Use mimalloc.")
    parser.add_argument("--use_openblas", action='store_true', help="Build with OpenBLAS.")
    parser.add_argument("--use_mkldnn", action='store_true', help="Build with MKLDNN.")
    parser.add_argument("--use_mklml", action='store_true', help="Build with MKLML.")
    parser.add_argument("--use_gemmlowp", action='store_true', help="Build with gemmlowp for quantized gemm.")
    parser.add_argument("--use_automl", action='store_true', help="Build with AutoML support.")
    parser.add_argument("--use_ngraph", action='store_true', help="Build with nGraph.")
    parser.add_argument("--use_openvino", nargs="?", const="CPU_FP32",
                        choices=["CPU_FP32","GPU_FP32","GPU_FP16","VAD-M_FP16","MYRIAD_FP16","VAD-F_FP32"], help="Build with OpenVINO for specific hardware.")
    parser.add_argument("--use_dnnlibrary", action='store_true', help="Build with DNNLibrary.")
    parser.add_argument("--use_nsync", action='store_true', help="Build with NSYNC.")
    parser.add_argument("--use_preinstalled_eigen", action='store_true', help="Use pre-installed eigen.")
    parser.add_argument("--eigen_path", help="Path to pre-installed eigen.")
    parser.add_argument("--use_tvm", action="store_true", help="Build with tvm")
    parser.add_argument("--use_openmp", action='store_true', help="Build with OpenMP.")
    parser.add_argument("--use_llvm", action="store_true", help="Build tvm with llvm")
    parser.add_argument("--use_eigenthreadpool", action="store_true", help="Build with eigenthreadpool")
    parser.add_argument("--enable_msinternal", action="store_true", help="Enable for Microsoft internal builds only.")
    parser.add_argument("--llvm_path", help="Path to llvm dir")
    parser.add_argument("--azure_sas_key", help="Azure storage sas key, starts with '?'")
    parser.add_argument("--use_brainslice", action="store_true", help="Build with brain slice")
    parser.add_argument("--brain_slice_package_path", help="Path to brain slice packages")
    parser.add_argument("--brain_slice_package_name", help="Name of brain slice packages")
    parser.add_argument("--brain_slice_client_package_name", help="Name of brainslice client package")
    parser.add_argument("--use_nuphar", action='store_true', help="Build with nuphar")
    parser.add_argument("--use_tensorrt", action='store_true', help="Build with TensorRT")
    parser.add_argument("--tensorrt_home", help="Path to TensorRT installation dir")
    parser.add_argument("--use_full_protobuf", action='store_true', help="Use the full protobuf library")
    parser.add_argument("--disable_contrib_ops", action='store_true', help="Disable contrib ops (reduces binary size)")
    parser.add_argument("--skip_onnx_tests", action='store_true', help="Explicitly disable all onnx related tests")
    parser.add_argument("--enable_msvc_static_runtime", action='store_true', help="Enable static linking of MSVC runtimes.")
    parser.add_argument("--enable_language_interop_ops", action='store_true', help="Enable operator implemented in language other than cpp")
    parser.add_argument("--cmake_generator", choices=['Visual Studio 15 2017', 'Visual Studio 16 2019'],
                        default='Visual Studio 15 2017', help="Specify the generator that CMake invokes. This is only supported on Windows")
    parser.add_argument("--enable_multi_device_test", action='store_true', help="Test with multi-device. Mostly used for multi-device GPU")
    parser.add_argument("--use_dml", action='store_true', help="Build with DirectML.")
    return parser.parse_args()

def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    executable_path = shutil.which(command_or_path)
    if executable_path is None:
        raise BuildError("Failed to resolve executable path for '{}'.".format(command_or_path))
    return os.path.realpath(executable_path)

def is_windows():
    return sys.platform.startswith("win")

def is_ubuntu_1604():
    return platform.linux_distribution()[0] == 'Ubuntu' and platform.linux_distribution()[1] == '16.04'

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
    return subprocess.run(args, cwd=cwd, check=True, stdout=stdout, stderr=stderr, env=my_env, shell=shell)

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
    dep_packages.append('numpy=={}'.format(numpy_version) if numpy_version else 'numpy>=1.15.0')
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

#the last part of src_url should be unique, across all the builds
def download_test_data(build_dir, src_url, expected_md5, azure_sas_key):
    cache_dir = os.path.join(expanduser("~"), '.cache','onnxruntime')
    os.makedirs(cache_dir, exist_ok=True)
    local_zip_file = os.path.join(cache_dir, os.path.basename(src_url))
    if not check_md5(local_zip_file, expected_md5):
        log.info("Downloading test data")
        if azure_sas_key:
            src_url += azure_sas_key
        # try to avoid logging azure_sas_key
        if shutil.which('aria2c'):
            result = subprocess.run(['aria2c','-x', '5', '-j',' 5',  '-q', src_url, '-d', cache_dir])
            if result.returncode != 0:
                raise BuildError("aria2c exited with code {}.".format(result.returncode))
        elif shutil.which('curl'):
            result = subprocess.run(['curl', '-s', src_url, '-o', local_zip_file])
            if result.returncode != 0:
                raise BuildError("curl exited with code {}.".format(result.returncode))
        else:
            import urllib.request
            import urllib.error
            try:
                urllib.request.urlretrieve(src_url, local_zip_file)
            except urllib.error.URLError:
                raise BuildError("urllib.request.urlretrieve() failed.")
    models_dir = os.path.join(build_dir,'models')
    if os.path.exists(models_dir):
        log.info('deleting %s' % models_dir)
        shutil.rmtree(models_dir)
    if shutil.which('unzip'):
        run_subprocess(['unzip','-qd', models_dir, local_zip_file])
    elif shutil.which('7z'):  # 7-Zip
        run_subprocess(['7z','x', local_zip_file, '-y', '-o' + models_dir])
    elif shutil.which('7za'):  # 7-Zip standalone
        run_subprocess(['7za', 'x', local_zip_file, '-y', '-o' + models_dir])
    else:
        #TODO: use python for unzip
        log.error("No unzip tool for use")
        return False
    return True

def setup_test_data(build_dir, configs, test_data_url, test_data_checksum, azure_sas_key):
    if test_data_url is not None:
        """Sets up the test data, downloading it if needed."""
        if not download_test_data(build_dir, test_data_url, test_data_checksum, azure_sas_key):
            raise BuildError("Failed to set up test data.")

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
                 "-Donnxruntime_GENERATE_TEST_REPORTS=ON",
                 "-Donnxruntime_DEV_MODE=" + ("OFF" if args.android else "ON"),
                 "-DPYTHON_EXECUTABLE=" + sys.executable,
                 "-Donnxruntime_USE_CUDA=" + ("ON" if args.use_cuda else "OFF"),
                 "-Donnxruntime_USE_NSYNC=" + ("OFF" if is_windows() or not args.use_nsync else "ON"),
                 "-Donnxruntime_CUDNN_HOME=" + (cudnn_home if args.use_cuda else ""),
                 "-Donnxruntime_USE_AUTOML=" + ("ON" if args.use_automl else "OFF"),
                 "-Donnxruntime_CUDA_HOME=" + (cuda_home if args.use_cuda else ""),
                 "-Donnxruntime_USE_JEMALLOC=" + ("ON" if args.use_jemalloc else "OFF"),
                 "-Donnxruntime_USE_MIMALLOC=" + ("ON" if args.use_mimalloc else "OFF"),
                 "-Donnxruntime_ENABLE_PYTHON=" + ("ON" if args.enable_pybind else "OFF"),
                 "-Donnxruntime_BUILD_CSHARP=" + ("ON" if args.build_csharp else "OFF"),
                 "-Donnxruntime_BUILD_SHARED_LIB=" + ("ON" if args.build_shared_lib or args.build_server else "OFF"),
                 "-Donnxruntime_USE_EIGEN_FOR_BLAS=" + ("OFF" if args.use_openblas else "ON"),
                 "-Donnxruntime_USE_OPENBLAS=" + ("ON" if args.use_openblas else "OFF"),
                 "-Donnxruntime_USE_MKLDNN=" + ("ON" if args.use_mkldnn else "OFF"),
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
                 "-Donnxruntime_USE_BRAINSLICE=" + ("ON" if args.use_brainslice else "OFF"),
                 "-Donnxruntime_USE_NUPHAR=" + ("ON" if args.use_nuphar else "OFF"),
                 "-Donnxruntime_USE_EIGEN_THREADPOOL=" + ("ON" if args.use_eigenthreadpool else "OFF"),
                 "-Donnxruntime_USE_TENSORRT=" + ("ON" if args.use_tensorrt else "OFF"),
                 "-Donnxruntime_TENSORRT_HOME=" + (tensorrt_home if args.use_tensorrt else ""),
                  # By default - we currently support only cross compiling for ARM/ARM64 (no native compilation supported through this script)
                 "-Donnxruntime_CROSS_COMPILING=" + ("ON" if args.arm64 or args.arm else "OFF"),
                 "-Donnxruntime_BUILD_SERVER=" + ("ON" if args.build_server else "OFF"),
                 "-Donnxruntime_BUILD_x86=" + ("ON" if args.x86 else "OFF"),
                  # nGraph and TensorRT providers currently only supports full_protobuf option.
                 "-Donnxruntime_USE_FULL_PROTOBUF=" + ("ON" if args.use_full_protobuf or args.use_ngraph or args.use_tensorrt or args.build_server or args.gen_doc else "OFF"),
                 "-Donnxruntime_DISABLE_CONTRIB_OPS=" + ("ON" if args.disable_contrib_ops else "OFF"),
                 "-Donnxruntime_MSVC_STATIC_RUNTIME=" + ("ON" if args.enable_msvc_static_runtime else "OFF"),
                 # enable pyop if it is nightly build
                 "-Donnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=" + ("ON" if args.enable_language_interop_ops or (args.config != 'Debug' and bool(os.getenv('NIGHTLY_BUILD') == '1')) else "OFF"),
                 "-Donnxruntime_USE_DML=" + ("ON" if args.use_dml else "OFF"),
                 ]
    if args.use_brainslice:
        bs_pkg_name = args.brain_slice_package_name.split('.', 1)
        bs_shared_lib_name = '.'.join((bs_pkg_name[0], 'redist', bs_pkg_name[1]))
        cmake_args += [
            "-Donnxruntime_BRAINSLICE_LIB_PATH=%s/%s" % (args.brain_slice_package_path, args.brain_slice_package_name),
            "-Donnxruntime_BS_CLIENT_PACKAGE=%s/%s" % (args.brain_slice_package_path, args.brain_slice_client_package_name),
            "-Donnxruntime_BRAINSLICE_dynamic_lib_PATH=%s/%s" % (args.brain_slice_package_path, bs_shared_lib_name)]

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

    cmake_args += ["-D{}".format(define) for define in cmake_extra_defines]

    if is_windows():
        cmake_args += cmake_extra_args

    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)
        os.makedirs(config_build_dir, exist_ok=True)

        if args.use_tvm:
            os.environ["PATH"] = os.path.join(config_build_dir, "external", "tvm", config) + os.pathsep + os.environ["PATH"]

        run_subprocess(cmake_args  + ["-DCMAKE_BUILD_TYPE={}".format(config)], cwd=config_build_dir)


def clean_targets(cmake_path, build_dir, configs):
    for config in configs:
        log.info("Cleaning targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path,
                    "--build", build_dir2,
                    "--config", config,
                    "--target", "clean"]

        run_subprocess(cmd_args)

def build_targets(cmake_path, build_dir, configs, parallel):
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
                build_tool_args += ["/maxcpucount:" + num_cores]
            else:
                build_tool_args += ["-j" + num_cores]

        if (build_tool_args):
            cmd_args += [ "--" ]
            cmd_args += build_tool_args

        run_subprocess(cmd_args)

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
                m = re.match("CUDA Version (\d+).(\d+)", first_line)
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

        # Set maximum batch size for TensorRT. The number needs to be no less than maximum batch size in all unit tests
        os.environ["ORT_TENSORRT_MAX_BATCH_SIZE"] = "13"

        # Set maximum workspace size in byte for TensorRT (1GB = 1073741824 bytes)
        os.environ["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = "1073741824"

        # Set maximum number of iterations to detect unsupported nodes and partition the models for TensorRT
        os.environ["ORT_TENSORRT_MAX_PARSER_ITERATIONS"] = "6"

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

def run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs, enable_python_tests, enable_tvm = False, enable_tensorrt = False, enable_ngraph = False, enable_nnapi=False):
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
        run_subprocess([ctest_path, "--build-config", config, "--verbose"],
                       cwd=cwd, dll_path=dll_path)

        if enable_python_tests:
            # Disable python tests for TensorRT because many tests are not supported yet
            if enable_tensorrt :
                return
            if is_windows():
                cwd = os.path.join(cwd, config)
            run_subprocess([sys.executable, 'onnxruntime_test_python.py'], cwd=cwd, dll_path=dll_path)
            try:
                import onnx
                onnx_test = True
            except ImportError:
                warnings.warn("onnx is not installed. Following test cannot be run.")
                onnx_test = False
            if onnx_test:
                run_subprocess([sys.executable, 'onnxruntime_test_python_backend.py'], cwd=cwd, dll_path=dll_path)
                run_subprocess([sys.executable, os.path.join(source_dir,'onnxruntime','test','onnx','gen_test_models.py'),'--output_dir','test_models'], cwd=cwd)
                run_subprocess([os.path.join(cwd,'onnx_test_runner'), 'test_models'], cwd=cwd)
                if config != 'Debug':
                    run_subprocess([sys.executable, 'onnx_backend_test_series.py'], cwd=cwd, dll_path=dll_path)
            if not args.skip_keras_test:
                try:
                    import onnxmltools
                    import keras
                    onnxml_test = True
                except ImportError:
                    warnings.warn("onnxmltools and keras are not installed. Following test cannot be run.")
                    onnxml_test = False
                if onnxml_test:
                    run_subprocess([sys.executable, 'onnxruntime_test_python_keras.py'], cwd=cwd, dll_path=dll_path)

def run_onnx_tests(build_dir, configs, onnx_test_data_dir, provider, enable_multi_device_test, enable_parallel_executor_test, num_parallel_models):
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
          if provider == 'mkldnn':
             cmd += ['-c', '1']
          if provider == 'openvino':
             cmd += ['-c', '1']
          if provider == 'nuphar':
             cmd += ['-c', '1']

        if num_parallel_models > 0:
          cmd += ["-j", str(num_parallel_models)]

        if enable_multi_device_test:
          cmd += ['-d', '1']

        if config != 'Debug' and os.path.exists(model_dir):
          # some models in opset9 and above are not supported by TensorRT yet
          if provider == 'tensorrt':
            model_dir = os.path.join(model_dir, "opset8")
          cmd.append(model_dir)
        if os.path.exists(onnx_test_data_dir):
          cmd.append(onnx_test_data_dir)

        if config == 'Debug' and provider == 'nuphar':
          return

        run_subprocess([exe] + cmd, cwd=cwd)
        if enable_parallel_executor_test:
          run_subprocess([exe,'-x'] + cmd, cwd=cwd)


# mkldnn temporary function for running onnx tests and model tests separately.
def mkldnn_run_onnx_tests(build_dir, configs, onnx_test_data_dir):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
           exe = os.path.join(cwd, config, 'onnx_test_runner')
           model_dir = os.path.join(cwd, "models")
        else:
           exe = os.path.join(cwd, 'onnx_test_runner')
           model_dir = os.path.join(build_dir, "models")
        cmd_base = ['-e', 'mkldnn', '-c', '1', '-j', '1']
        if os.path.exists(onnx_test_data_dir):
          onnxdata_cmd = cmd_base + [onnx_test_data_dir]
          # /data/onnx
          run_subprocess([exe] + onnxdata_cmd, cwd=cwd)
          run_subprocess([exe,'-x'] + onnxdata_cmd, cwd=cwd)

        # models/opset7, models/opset8, models/opset9
        if config != 'Debug' and os.path.exists(model_dir):
          opset7_model_dir = os.path.join(model_dir, 'opset7')
          opset7_cmd = cmd_base + [opset7_model_dir]
          opset8_model_dir = os.path.join(model_dir, 'opset8')
          opset8_cmd = cmd_base + [opset8_model_dir]
          opset9_model_dir = os.path.join(model_dir, 'opset9')
          opset9_cmd = cmd_base + [opset9_model_dir]
          run_subprocess([exe] + opset7_cmd, cwd=cwd)
          run_subprocess([exe, '-x'] + opset7_cmd, cwd=cwd)
          run_subprocess([exe] + opset8_cmd, cwd=cwd)
          run_subprocess([exe, '-x'] + opset8_cmd, cwd=cwd)
          run_subprocess([exe] + opset9_cmd, cwd=cwd)
          run_subprocess([exe, '-x'] + opset9_cmd, cwd=cwd)


# nuphar temporary function for running python tests separately as it requires ONNX 1.5.0
def nuphar_run_python_tests(build_dir, configs, azure_sas_key):
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


def split_server_binary_and_symbol(build_dir, configs):
    if is_windows():
        # TODO: Windows support
        pass
    else:
        for config in configs:
            if config == 'RelWithDebInfo':
                config_build_dir = get_config_build_dir(build_dir, config)
                run_subprocess(['objcopy', '--only-keep-debug', 'onnxruntime_server', 'onnxruntime_server.symbol'], cwd=config_build_dir)
                run_subprocess(['strip', '--strip-debug', '--strip-unneeded', 'onnxruntime_server'], cwd=config_build_dir)
                run_subprocess(['objcopy', '--add-gnu-debuglink=onnxruntime_server.symbol', 'onnxruntime_server'], cwd=config_build_dir)
                libonnx = glob.glob(os.path.join(config_build_dir, "libonnxruntime.so.*"))
                if len(libonnx) != 1 :
                    raise ValueError("Too many libonxruntime.so.*")
                libonnx = libonnx[0]
                run_subprocess(['objcopy', '--only-keep-debug', libonnx, libonnx+'.symbol'], cwd=config_build_dir)
                run_subprocess(['strip', '--strip-debug', libonnx], cwd=config_build_dir)
                run_subprocess(['objcopy', '--add-gnu-debuglink={}.symbol'.format(libonnx), libonnx], cwd=config_build_dir)


def run_server_tests(build_dir, configs):
    pip_freeze_result = run_subprocess([sys.executable, '-m', 'pip', 'freeze'], capture=True).stdout
    installed_packages = [r.decode().split('==')[0] for r in pip_freeze_result.split()]
    if not (('requests' in installed_packages) and ('protobuf' in installed_packages) and ('numpy' in installed_packages) and ('grpcio' in installed_packages)):
        if hasattr(sys, 'real_prefix'):
            # In virtualenv
            run_subprocess([sys.executable, '-m', 'pip', 'install', '--trusted-host', 'files.pythonhosted.org', 'requests', 'protobuf', 'numpy', 'grpcio'])
        else:
            # Outside virtualenv
            run_subprocess([sys.executable, '-m', 'pip', 'install', '--user', '--trusted-host', 'files.pythonhosted.org', 'requests', 'protobuf', 'numpy', 'grpcio'])
    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)
        if is_windows():
            server_app_path = os.path.join(config_build_dir, config, 'onnxruntime_server.exe')
            python_package_path = os.path.join(config_build_dir, config)
        else:
            server_app_path = os.path.join(config_build_dir, 'onnxruntime_server')
            python_package_path = config_build_dir
        server_test_folder = os.path.join(config_build_dir, 'server_test')
        server_test_model_folder = os.path.join(build_dir, 'models', 'opset8', 'test_mnist')
        server_test_data_folder = os.path.join(config_build_dir, 'testdata', 'server')
        run_subprocess([sys.executable, 'test_main.py', server_app_path, server_test_model_folder, server_test_data_folder, python_package_path, server_test_folder], cwd=server_test_folder, dll_path=None)


def run_server_model_tests(build_dir, configs):
    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)
        server_test_folder = os.path.join(config_build_dir, 'server_test')
        server_test_data_folder = os.path.join(config_build_dir, 'server_test_data')

        if is_windows():
            server_app_path = os.path.join(config_build_dir, config, 'onnxruntime_server.exe')
            test_raw_data_folder = os.path.join(config_build_dir, 'models')
            python_package_path = os.path.join(config_build_dir, config)
        else:
            server_app_path = os.path.join(config_build_dir, 'onnxruntime_server')
            test_raw_data_folder = os.path.join(build_dir, 'models')
            python_package_path = config_build_dir

        run_subprocess([sys.executable, 'model_zoo_data_prep.py', test_raw_data_folder, server_test_data_folder, python_package_path, server_test_folder], cwd=server_test_folder, dll_path=None)
        run_subprocess([sys.executable, 'model_zoo_tests.py', server_app_path, test_raw_data_folder, server_test_data_folder, python_package_path, server_test_folder], cwd=server_test_folder, dll_path=None)


def build_python_wheel(source_dir, build_dir, configs, use_cuda, use_ngraph, use_tensorrt, use_openvino, use_nuphar, nightly_build = False):
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
        log.warning('The updated opkernel document file '+str(opkernel_doc_path)+' is different from the checked in version. Consider regenrating the file with CPU, MKLDNN and CUDA providers enabled.')
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

    if args.use_tensorrt:
        args.use_cuda = True

    if args.build_wheel or args.enable_server_model_tests:
        args.enable_pybind = True

    if args.build_csharp:
        args.build_shared_lib = True

    # Disabling unit tests for VAD-F as FPGA only supports models with NCHW layout
    if args.use_openvino == "VAD-F_FP32":
        args.test = False

    configs = set(args.config)

    # setup paths and directories
    cmake_path = resolve_executable_path(args.cmake_path)
    ctest_path = resolve_executable_path(args.ctest_path)
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
            toolset = 'host=x64'
            if (args.msvc_toolset):
                toolset += ',version=' + args.msvc_toolset
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

        if args.enable_onnx_tests or args.download_test_data:
            if args.download_test_data:
                if not args.test_data_url or not args.test_data_checksum:
                   raise UsageError("The test_data_url and test_data_checksum arguments are required.")
            setup_test_data(build_dir, configs, args.test_data_url, args.test_data_checksum, args.azure_sas_key)

        if args.path_to_protoc_exe:
            path_to_protoc_exe = args.path_to_protoc_exe

        generate_build_tree(cmake_path, source_dir, build_dir, cuda_home, cudnn_home, tensorrt_home, path_to_protoc_exe, configs, cmake_extra_defines,
                            args, cmake_extra_args)

    if (args.clean):
        clean_targets(cmake_path, build_dir, configs)

    # if using DML, perform initial nuget package restore
    setup_dml_build(args, cmake_path, build_dir, configs)

    if (args.build):
        build_targets(cmake_path, build_dir, configs, args.parallel)

    if args.test :
        run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs,
                              args.enable_pybind and not args.skip_onnx_tests,
                              args.use_tvm, args.use_tensorrt, args.use_ngraph,
                              args.use_dnnlibrary)
        # run the onnx model tests if requested explicitly.
        if args.enable_onnx_tests and not args.skip_onnx_tests:
            # directory from ONNX submodule with ONNX test data
            onnx_test_data_dir = '/data/onnx'
            if is_windows() or not os.path.exists(onnx_test_data_dir):
                onnx_test_data_dir = os.path.join(source_dir, "cmake", "external", "onnx", "onnx", "backend", "test", "data")

            if args.use_tensorrt:
              # Disable some onnx unit tests that TensorRT doesn't supported yet
              if not is_windows():
                onnx_test_data_dir = os.path.join(source_dir, "cmake", "external", "onnx", "onnx", "backend", "test", "data", "simple")
                run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'tensorrt', args.enable_multi_device_test, False, 1)
            elif args.use_cuda:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'cuda', args.enable_multi_device_test, False, 2)
            elif args.x86 or platform.system() == 'Darwin':
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, None, args.enable_multi_device_test, False, 1)
            elif args.use_ngraph:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'ngraph', args.enable_multi_device_test, True, 1)
            elif args.use_openvino:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'openvino', args.enable_multi_device_test, False, 1)
              # TODO: parallel executor test fails on MacOS
            elif args.use_nuphar:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'nuphar', args.enable_multi_device_test, False, 1)
            else:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, None, args.enable_multi_device_test, True, 0)

            if args.use_dml:
              run_onnx_tests(build_dir, configs, onnx_test_data_dir, 'dml', args.enable_multi_device_test, False, 1)

              if args.use_mkldnn:
                mkldnn_run_onnx_tests(build_dir, configs, onnx_test_data_dir)

        # run nuphar python tests last, as it installs ONNX 1.5.0
        if args.enable_pybind and not args.skip_onnx_tests and args.use_nuphar:
            nuphar_run_python_tests(build_dir, configs, args.azure_sas_key)

    if args.build_server:
        split_server_binary_and_symbol(build_dir, configs)
        if args.enable_server_tests:
            run_server_tests(build_dir, configs)
        if args.enable_server_model_tests:
            run_server_model_tests(build_dir, configs)

    if args.build:
        if args.build_wheel:
            nightly_build = bool(os.getenv('NIGHTLY_BUILD') == '1')
            build_python_wheel(source_dir, build_dir, configs, args.use_cuda, args.use_ngraph, args.use_tensorrt, args.use_openvino, args.use_nuphar, nightly_build)

    if args.gen_doc and (args.build or args.test):
        generate_documentation(source_dir, build_dir, configs)

    log.info("Build complete")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseError as e:
        log.error(str(e))
        sys.exit(1)
