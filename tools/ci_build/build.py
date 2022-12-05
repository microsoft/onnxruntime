#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import contextlib
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from distutils.version import LooseVersion
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, os.path.join(REPO_DIR, "tools", "python"))


import util.android as android  # noqa: E402
from util import get_logger, is_linux, is_macOS, is_windows, run  # noqa: E402

log = get_logger("build")


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
        raise BuildError("Bad python major version: expecting python 3, found version " "'{}'".format(sys.version))
    if sys.version_info[1] < 6:
        raise BuildError("Bad python minor version: expecting python 3.6+, found version " "'{}'".format(sys.version))


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ["true", "false"]:
        raise ValueError("Need bool; got %r" % s)
    return {"true": True, "false": False}[s.lower()]


_check_python_version()


def _openvino_verify_device_type(device_read):
    choices = ["CPU_FP32", "CPU_FP16", "GPU_FP32", "GPU_FP16", "VAD-M_FP16", "MYRIAD_FP16", "VAD-F_FP32"]

    choices1 = [
        "CPU_FP32_NO_PARTITION",
        "CPU_FP16_NO_PARTITION",
        "GPU_FP32_NO_PARTITION",
        "GPU_FP16_NO_PARTITION",
        "VAD-M_FP16_NO_PARTITION",
        "MYRIAD_FP16_NO_PARTITION",
        "VAD-F_FP32_NO_PARTITION",
    ]
    status_hetero = True
    res = False
    if device_read in choices:
        res = True
    elif device_read in choices1:
        res = True
    elif device_read.startswith("HETERO:") or device_read.startswith("MULTI:") or device_read.startswith("AUTO:"):
        res = True
        comma_separated_devices = device_read.split(":")
        comma_separated_devices = comma_separated_devices[1].split(",")
        if len(comma_separated_devices) < 2:
            print("At least two devices required in Hetero/Multi/Auto Mode")
            status_hetero = False
        dev_options = ["CPU", "GPU", "MYRIAD", "FPGA", "HDDL"]
        for dev in comma_separated_devices:
            if dev not in dev_options:
                status_hetero = False
                break

    def invalid_hetero_build():
        print("\n" + "If trying to build Hetero/Multi/Auto, specifiy the supported devices along with it." + +"\n")
        print("specify the keyword HETERO or MULTI or AUTO followed by the devices ")
        print("in the order of priority you want to build" + "\n")
        print("The different hardware devices that can be added in HETERO or MULTI or AUTO")
        print("are ['CPU','GPU','MYRIAD','FPGA','HDDL']" + "\n")
        print("An example of how to specify the hetero build type. Ex: HETERO:GPU,CPU" + "\n")
        print("An example of how to specify the MULTI build type. Ex: MULTI:MYRIAD,CPU" + "\n")
        print("An example of how to specify the AUTO build type. Ex: AUTO:GPU,CPU" + "\n")
        sys.exit("Wrong Build Type selected")

    if res is False:
        print("\n" + "You have selcted wrong configuration for the build.")
        print("pick the build type for specific Hardware Device from following options: ", choices)
        print("(or) from the following options with graph partitioning disabled: ", choices1)
        print("\n")
        if not (device_read.startswith("HETERO") or device_read.startswith("MULTI") or device_read.startswith("AUTO")):
            invalid_hetero_build()
        sys.exit("Wrong Build Type selected")

    if status_hetero is False:
        invalid_hetero_build()

    return device_read


def parse_arguments():
    class Parser(argparse.ArgumentParser):
        # override argument file line parsing behavior - allow multiple arguments per line and handle quotes
        def convert_arg_line_to_args(self, arg_line):
            return shlex.split(arg_line)

    parser = Parser(
        description="ONNXRuntime CI build driver.",
        usage="""
        Default behavior is --update --build --test for native architecture builds.
        Default behavior is --update --build for cross-compiled builds.

        The Update phase will update git submodules, and run cmake to generate makefiles.
        The Build phase will build all projects.
        The Test phase will run all unit tests, and optionally the ONNX tests.

        Use the individual flags to only run the specified stages.
        """,
        # files containing arguments can be specified on the command line with "@<filename>" and the arguments within
        # will be included at that point
        fromfile_prefix_chars="@",
    )
    # Main arguments
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["Debug"],
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build.",
    )
    parser.add_argument("--update", action="store_true", help="Update makefiles.")
    parser.add_argument("--build", action="store_true", help="Build.")
    parser.add_argument(
        "--clean", action="store_true", help="Run 'cmake --build --target clean' for the selected config/s."
    )
    parser.add_argument(
        "--parallel",
        nargs="?",
        const="0",
        default="1",
        type=int,
        help="Use parallel build. The optional value specifies the maximum number of parallel jobs. "
        "If the optional value is 0 or unspecified, it is interpreted as the number of CPUs.",
    )
    parser.add_argument("--test", action="store_true", help="Run unit tests.")
    parser.add_argument("--skip_tests", action="store_true", help="Skip all tests.")
    parser.add_argument(
        "--compile_no_warning_as_error",
        action="store_true",
        help="Preventing warnings from being treated as errors on compile.",
    )
    # Training options
    parser.add_argument("--enable_nvtx_profile", action="store_true", help="Enable NVTX profile in ORT.")
    parser.add_argument("--enable_memory_profile", action="store_true", help="Enable memory profile in ORT.")
    parser.add_argument("--enable_training", action="store_true", help="Enable training in ORT.")
    parser.add_argument("--enable_training_ops", action="store_true", help="Enable training ops in inference graph.")
    parser.add_argument(
        "--enable_training_torch_interop", action="store_true", help="Enable training kernels interop with torch."
    )
    parser.add_argument("--enable_training_on_device", action="store_true", help="Enable on device training in ORT.")
    parser.add_argument("--disable_nccl", action="store_true", help="Disable Nccl.")
    parser.add_argument("--mpi_home", help="Path to MPI installation dir")
    parser.add_argument("--nccl_home", help="Path to NCCL installation dir")
    parser.add_argument("--use_mpi", nargs="?", default=True, const=True, type=_str_to_bool)

    # enable ONNX tests
    parser.add_argument(
        "--enable_onnx_tests",
        action="store_true",
        help="""When running the Test phase, run onnx_test_running against
        available test data directories.""",
    )
    parser.add_argument("--path_to_protoc_exe", help="Path to protoc exe.")
    parser.add_argument("--fuzz_testing", action="store_true", help="Enable Fuzz testing of the onnxruntime.")
    parser.add_argument(
        "--enable_symbolic_shape_infer_tests",
        action="store_true",
        help="""When running the Test phase, run symbolic shape inference against
        available test data directories.""",
    )

    # generate documentation
    parser.add_argument(
        "--gen_doc",
        nargs="?",
        const="yes",
        type=str,
        help="Generate documentation listing standard ONNX operators and types implemented by "
        "various execution providers and contrib operator schemas. Must be used for inference builds, only!"
        "Use `--gen_doc validate` to validate these match the current contents in /docs.",
    )

    parser.add_argument("--gen-api-doc", action="store_true", help="Generate API documentation for PyTorch frontend")

    # CUDA related
    parser.add_argument("--use_cuda", action="store_true", help="Enable CUDA.")
    parser.add_argument(
        "--cuda_version", help="The version of CUDA toolkit to use. " "Auto-detect if not specified. e.g. 9.0"
    )
    parser.add_argument(
        "--cuda_home",
        help="Path to CUDA home."
        "Read from CUDA_HOME environment variable if --use_cuda is true and "
        "--cuda_home is not specified.",
    )
    parser.add_argument(
        "--cudnn_home",
        help="Path to CUDNN home. "
        "Read from CUDNN_HOME environment variable if --use_cuda is true and "
        "--cudnn_home is not specified.",
    )
    parser.add_argument("--enable_cuda_line_info", action="store_true", help="Enable CUDA line info.")

    # Python bindings
    parser.add_argument("--enable_pybind", action="store_true", help="Enable Python Bindings.")
    parser.add_argument("--build_wheel", action="store_true", help="Build Python Wheel.")
    parser.add_argument(
        "--wheel_name_suffix",
        help="Suffix to append to created wheel names. " "This value is currently only used for nightly builds.",
    )
    parser.add_argument(
        "--numpy_version", help="Installs a specific version of numpy " "before building the python binding."
    )
    parser.add_argument("--skip-keras-test", action="store_true", help="Skip tests with Keras if keras is installed")

    # C-Sharp bindings
    parser.add_argument(
        "--build_csharp",
        action="store_true",
        help="Build C#.Net DLL and NuGet package. This should be only used in CI pipelines. "
        "For building C# bindings and packaging them into nuget package use --build_nuget arg.",
    )

    parser.add_argument(
        "--build_nuget",
        action="store_true",
        help="Build C#.Net DLL and NuGet package on the local machine. "
        "Currently only Windows and Linux platforms are supported.",
    )

    # Java bindings
    parser.add_argument("--build_java", action="store_true", help="Build Java bindings.")

    # Node.js binding
    parser.add_argument("--build_nodejs", action="store_true", help="Build Node.js binding and NPM package.")

    # Objective-C binding
    parser.add_argument("--build_objc", action="store_true", help="Build Objective-C binding.")

    # Build a shared lib
    parser.add_argument("--build_shared_lib", action="store_true", help="Build a shared library for the ONNXRuntime.")

    # Build a shared lib
    parser.add_argument(
        "--build_apple_framework", action="store_true", help="Build a macOS/iOS framework for the ONNXRuntime."
    )

    # Build options
    parser.add_argument(
        "--cmake_extra_defines",
        nargs="+",
        action="append",
        help="Extra definitions to pass to CMake during build system "
        "generation. These are just CMake -D options without the leading -D.",
    )
    parser.add_argument("--target", help="Build a specific target, e.g. winml_dll")
    # This flag is needed when :
    # 1. The OS is 64 bits Windows
    # 2. And the target binary is for 32 bits Windows
    # 3. And the python used for running this script is 64 bits.
    # But if you can get a 32 bits python, the build will run better and you won't need this flag.
    parser.add_argument(
        "--x86",
        action="store_true",
        help="[cross-compiling] Create Windows x86 makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed",
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help="[cross-compiling] Create ARM makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed",
    )
    parser.add_argument(
        "--arm64",
        action="store_true",
        help="[cross-compiling] Create ARM64 makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed",
    )
    parser.add_argument(
        "--arm64ec",
        action="store_true",
        help="[cross-compiling] Create ARM64EC makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed",
    )
    parser.add_argument("--msvc_toolset", help="MSVC toolset to use. e.g. 14.11")
    parser.add_argument("--android", action="store_true", help="Build for Android")
    parser.add_argument(
        "--android_abi",
        default="arm64-v8a",
        choices=["armeabi-v7a", "arm64-v8a", "x86", "x86_64"],
        help="Specify the target Android Application Binary Interface (ABI)",
    )
    parser.add_argument("--android_api", type=int, default=27, help="Android API Level, e.g. 21")
    parser.add_argument(
        "--android_sdk_path", type=str, default=os.environ.get("ANDROID_HOME", ""), help="Path to the Android SDK"
    )
    parser.add_argument(
        "--android_ndk_path", type=str, default=os.environ.get("ANDROID_NDK_HOME", ""), help="Path to the Android NDK"
    )
    parser.add_argument(
        "--android_cpp_shared",
        action="store_true",
        help="Build with shared libc++ instead of the default static libc++.",
    )
    parser.add_argument("--android_run_emulator", action="store_true", help="Start up an Android emulator if needed.")

    parser.add_argument("--use_gdk", action="store_true", help="Build with the GDK toolchain.")
    parser.add_argument(
        "--gdk_edition",
        default=os.path.normpath(os.environ.get("GameDKLatest", "")).split(os.sep)[-1],
        help="Build with a specific GDK edition. Defaults to the latest installed.",
    )
    parser.add_argument("--gdk_platform", default="Scarlett", help="Sets the GDK target platform.")

    parser.add_argument("--ios", action="store_true", help="build for ios")
    parser.add_argument(
        "--ios_sysroot", default="", help="Specify the location name of the macOS platform SDK to be used"
    )
    parser.add_argument(
        "--ios_toolchain_file",
        default="",
        help="Path to ios toolchain file, " "or cmake/onnxruntime_ios.toolchain.cmake will be used",
    )
    parser.add_argument(
        "--xcode_code_signing_team_id", default="", help="The development team ID used for code signing in Xcode"
    )
    parser.add_argument(
        "--xcode_code_signing_identity", default="", help="The development identity used for code signing in Xcode"
    )
    parser.add_argument(
        "--use_xcode", action="store_true", help="Use Xcode as cmake generator, this is only supported on MacOS."
    )
    parser.add_argument(
        "--osx_arch",
        default="arm64" if platform.machine() == "arm64" else "x86_64",
        choices=["arm64", "arm64e", "x86_64"],
        help="Specify the Target specific architectures for macOS and iOS, This is only supported on MacOS",
    )
    parser.add_argument(
        "--apple_deploy_target",
        type=str,
        help="Specify the minimum version of the target platform "
        "(e.g. macOS or iOS)"
        "This is only supported on MacOS",
    )
    parser.add_argument(
        "--disable_memleak_checker", action="store_true", help="Disable memory leak checker from Windows build"
    )

    # WebAssembly build
    parser.add_argument("--build_wasm", action="store_true", help="Build for WebAssembly")
    parser.add_argument("--build_wasm_static_lib", action="store_true", help="Build for WebAssembly static library")
    parser.add_argument("--emsdk_version", default="3.1.19", help="Specify version of emsdk")

    parser.add_argument("--enable_wasm_simd", action="store_true", help="Enable WebAssembly SIMD")
    parser.add_argument("--enable_wasm_threads", action="store_true", help="Enable WebAssembly multi-threads support")

    parser.add_argument(
        "--disable_wasm_exception_catching", action="store_true", help="Disable exception catching in WebAssembly."
    )
    parser.add_argument(
        "--enable_wasm_api_exception_catching", action="store_true", help="Catch exceptions at top level api."
    )
    parser.add_argument(
        "--enable_wasm_exception_throwing_override",
        action="store_true",
        help="Enable exception throwing in WebAssembly, this will override default disabling exception throwing "
        "behavior when disable exceptions.",
    )

    parser.add_argument(
        "--enable_wasm_profiling", action="store_true", help="Enable WebAsselby profiling and preserve function names"
    )
    parser.add_argument(
        "--enable_wasm_debug_info", action="store_true", help="Build WebAssembly with DWARF format debug info"
    )

    parser.add_argument("--wasm_malloc", help="Specify memory allocator for WebAssembly")

    parser.add_argument(
        "--emscripten_settings",
        nargs="+",
        action="append",
        help="Extra emscripten settings to pass to emcc using '-s <key>=<value>' during build.",
    )

    # Enable onnxruntime-extensions
    parser.add_argument(
        "--use_extensions",
        action="store_true",
        help="Enable custom operators in onnxruntime-extensions, use git submodule onnxruntime-extensions "
        "in path cmake/external/onnxruntime-extensions by default.",
    )
    parser.add_argument(
        "--extensions_overridden_path",
        type=str,
        help="Path to pre-pulled onnxruntime-extensions, will override default onnxruntime-extensions path.",
    )

    # Arguments needed by CI
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake program.")
    parser.add_argument(
        "--ctest_path",
        default="ctest",
        help="Path to the CTest program. It can be an empty string. If it is empty, "
        "we will use this script driving the test programs directly.",
    )
    parser.add_argument(
        "--skip_submodule_sync",
        action="store_true",
        help="Don't do a " "'git submodule update'. Makes the Update phase faster.",
    )
    parser.add_argument("--use_vstest", action="store_true", help="Use use_vstest for running unitests.")
    parser.add_argument("--use_mimalloc", action="store_true", help="Use mimalloc allocator")
    parser.add_argument("--use_dnnl", action="store_true", help="Build with DNNL.")
    parser.add_argument(
        "--dnnl_gpu_runtime", action="store", default="", type=str.lower, help="e.g. --dnnl_gpu_runtime ocl"
    )
    parser.add_argument(
        "--dnnl_opencl_root",
        action="store",
        default="",
        help="Path to OpenCL SDK. "
        'e.g. --dnnl_opencl_root "C:/Program Files (x86)/IntelSWTools/sw_dev_tools/OpenCL/sdk"',
    )
    parser.add_argument(
        "--use_openvino",
        nargs="?",
        const="CPU_FP32",
        type=_openvino_verify_device_type,
        help="Build with OpenVINO for specific hardware.",
    )
    parser.add_argument("--use_coreml", action="store_true", help="Build with CoreML support.")
    parser.add_argument("--use_snpe", action="store_true", help="Build with SNPE support.")
    parser.add_argument("--snpe_root", help="Path to SNPE SDK root.")
    parser.add_argument("--use_nnapi", action="store_true", help="Build with NNAPI support.")
    parser.add_argument(
        "--nnapi_min_api", type=int, help="Minimum Android API level to enable NNAPI, should be no less than 27"
    )
    parser.add_argument("--use_rknpu", action="store_true", help="Build with RKNPU.")
    parser.add_argument("--use_preinstalled_eigen", action="store_true", help="Use pre-installed Eigen.")
    parser.add_argument("--eigen_path", help="Path to pre-installed Eigen.")
    parser.add_argument("--enable_msinternal", action="store_true", help="Enable for Microsoft internal builds only.")
    parser.add_argument("--llvm_path", help="Path to llvm dir")
    parser.add_argument("--use_vitisai", action="store_true", help="Build with Vitis-AI")
    parser.add_argument("--use_tvm", action="store_true", help="Build with TVM")
    parser.add_argument("--tvm_cuda_runtime", action="store_true", default=False, help="Build TVM with CUDA support")
    parser.add_argument(
        "--use_tvm_hash", action="store_true", help="Build ipp-crypto for hash generation. It is used by TVM EP only"
    )
    parser.add_argument("--use_tensorrt", action="store_true", help="Build with TensorRT")
    parser.add_argument("--use_tensorrt_builtin_parser", action="store_true", help="Use TensorRT builtin parser")
    parser.add_argument(
        "--tensorrt_placeholder_builder", action="store_true", help="Instantiate Placeholder TensorRT Builder"
    )
    parser.add_argument("--tensorrt_home", help="Path to TensorRT installation dir")
    parser.add_argument("--use_migraphx", action="store_true", help="Build with MIGraphX")
    parser.add_argument("--migraphx_home", help="Path to MIGraphX installation dir")
    parser.add_argument("--use_full_protobuf", action="store_true", help="Use the full protobuf library")

    parser.add_argument(
        "--llvm_config",
        type=str,
        default="",
        help="Path to llvm-config.exe for LLVM buit from sources. It is strongly needed for build on Windows",
    )
    parser.add_argument(
        "--skip_onnx_tests",
        action="store_true",
        help="Explicitly disable all onnx related tests. Note: Use --skip_tests to skip all tests.",
    )
    parser.add_argument("--skip_winml_tests", action="store_true", help="Explicitly disable all WinML related tests")
    parser.add_argument("--skip_nodejs_tests", action="store_true", help="Explicitly disable all Node.js binding tests")

    parser.add_argument(
        "--enable_msvc_static_runtime", action="store_true", help="Enable static linking of MSVC runtimes."
    )
    parser.add_argument(
        "--enable_language_interop_ops",
        action="store_true",
        help="Enable operator implemented in language other than cpp",
    )
    parser.add_argument(
        "--cmake_generator",
        choices=["Visual Studio 15 2017", "Visual Studio 16 2019", "Visual Studio 17 2022", "Ninja"],
        default="Visual Studio 16 2019" if is_windows() else None,
        help="Specify the generator that CMake invokes. " "This is only supported on Windows",
    )
    parser.add_argument(
        "--enable_multi_device_test",
        action="store_true",
        help="Test with multi-device. Mostly used for multi-device GPU",
    )
    parser.add_argument("--use_dml", action="store_true", help="Build with DirectML.")
    parser.add_argument(
        "--dml_path",
        type=str,
        default="",
        help="Path to a custom DirectML installation (must have bin/, lib/, and include/ subdirectories).",
    )
    parser.add_argument("--use_winml", action="store_true", help="Build with WinML.")
    parser.add_argument(
        "--winml_root_namespace_override", type=str, help="Specify the namespace that WinML builds into."
    )
    parser.add_argument(
        "--dml_external_project", action="store_true", help="Build with DirectML as an external project."
    )
    parser.add_argument(
        "--use_telemetry", action="store_true", help="Only official builds can set this flag to enable telemetry."
    )
    parser.add_argument("--enable_wcos", action="store_true", help="Build for Windows Core OS.")
    parser.add_argument("--enable_lto", action="store_true", help="Enable Link Time Optimization")
    parser.add_argument("--enable_transformers_tool_test", action="store_true", help="Enable transformers tool test")
    parser.add_argument(
        "--use_acl",
        nargs="?",
        const="ACL_1905",
        choices=["ACL_1902", "ACL_1905", "ACL_1908", "ACL_2002"],
        help="Build with ACL for ARM architectures.",
    )
    parser.add_argument("--acl_home", help="Path to ACL home dir")
    parser.add_argument("--acl_libs", help="Path to ACL libraries")
    parser.add_argument("--use_armnn", action="store_true", help="Enable ArmNN Execution Provider.")
    parser.add_argument(
        "--armnn_relu", action="store_true", help="Use the Relu operator implementation from the ArmNN EP."
    )
    parser.add_argument(
        "--armnn_bn", action="store_true", help="Use the Batch Normalization operator implementation from the ArmNN EP."
    )
    parser.add_argument("--armnn_home", help="Path to ArmNN home dir")
    parser.add_argument("--armnn_libs", help="Path to ArmNN libraries")
    parser.add_argument("--build_micro_benchmarks", action="store_true", help="Build ONNXRuntime micro-benchmarks.")

    # options to reduce binary size
    parser.add_argument(
        "--minimal_build",
        default=None,
        nargs="*",
        type=str.lower,
        help="Create a build that only supports ORT format models. "
        "See https://onnxruntime.ai/docs/tutorials/mobile/ for more information. "
        "RTTI is automatically disabled in a minimal build. "
        "To enable execution providers that compile kernels at runtime (e.g. NNAPI) pass 'extended' "
        "as a parameter. e.g. '--minimal_build extended'. "
        "To enable support for custom operators pass 'custom_ops' as a parameter. "
        "e.g. '--minimal_build custom_ops'. This can be combined with an 'extended' build by passing "
        "'--minimal_build extended custom_ops'",
    )

    parser.add_argument(
        "--include_ops_by_config",
        type=str,
        help="Include ops from config file. " "See /docs/Reduced_Operator_Kernel_build.md for more information.",
    )
    parser.add_argument(
        "--enable_reduced_operator_type_support",
        action="store_true",
        help="If --include_ops_by_config is specified, and the configuration file has type reduction "
        "information, limit the types individual operators support where possible to further "
        "reduce the build size. "
        "See /docs/Reduced_Operator_Kernel_build.md for more information.",
    )

    parser.add_argument("--disable_contrib_ops", action="store_true", help="Disable contrib ops (reduces binary size)")
    parser.add_argument(
        "--disable_ml_ops", action="store_true", help="Disable traditional ML ops (reduces binary size)"
    )
    # Please note in our CMakeLists.txt this is already default on. But in this file we reverse it to default OFF.
    parser.add_argument("--disable_rtti", action="store_true", help="Disable RTTI (reduces binary size)")
    parser.add_argument(
        "--disable_exceptions",
        action="store_true",
        help="Disable exceptions to reduce binary size. Requires --minimal_build.",
    )

    parser.add_argument("--rocm_version", help="The version of ROCM stack to use. ")
    parser.add_argument("--use_rocm", action="store_true", help="Build with ROCm")
    parser.add_argument("--rocm_home", help="Path to ROCm installation dir")

    # Code coverage
    parser.add_argument(
        "--code_coverage", action="store_true", help="Generate code coverage when targetting Android (only)."
    )

    # lazy tensor support.
    parser.add_argument(
        "--enable_lazy_tensor", action="store_true", help="Enable use ORT as backend in Pytorch LazyTensor."
    )

    parser.add_argument("--ms_experimental", action="store_true", help="Build microsoft experimental operators.")

    # eager mode
    parser.add_argument("--build_eager_mode", action="store_true", help="Build ONNXRuntime micro-benchmarks.")
    parser.add_argument(
        "--eager_customop_module", default=None, help="Module containing custom op mappings for eager mode."
    )
    parser.add_argument(
        "--eager_customop_header", default=None, help="Header containing custom op definitions for eager mode."
    )

    parser.add_argument(
        "--enable_external_custom_op_schemas",
        action="store_true",
        help="Enable registering user defined custom operation schemas at shared library load time.\
            This feature is only supported/available on Ubuntu.",
    )

    parser.add_argument(
        "--external_graph_transformer_path", type=str, help="path to the external graph transformer dir."
    )

    parser.add_argument(
        "--test_external_transformer_example",
        action="store_true",
        help="run the example external transformer test, mainly used in CI pipeline.",
    )

    parser.add_argument(
        "--enable_cuda_profiling",
        action="store_true",
        help="enable cuda kernel profiling, \
        cupti library must be added to PATH beforehand.",
    )
    parser.add_argument("--use_cann", action="store_true", help="Build with CANN")
    parser.add_argument("--cann_home", help="Path to CANN installation dir")

    parser.add_argument(
        "--enable_rocm_profiling",
        action="store_true",
        help="enable rocm kernel profiling.",
    )

    parser.add_argument("--use_xnnpack", action="store_true", help="Enable xnnpack EP.")

    args = parser.parse_args()
    if args.android_sdk_path:
        args.android_sdk_path = os.path.normpath(args.android_sdk_path)
    if args.android_ndk_path:
        args.android_ndk_path = os.path.normpath(args.android_ndk_path)

    if args.enable_wasm_api_exception_catching:
        # if we catch on api level, we don't want to catch all
        args.disable_wasm_exception_catching = True
    if not args.disable_wasm_exception_catching or args.enable_wasm_api_exception_catching:
        # doesn't make sense to catch if no one throws
        args.enable_wasm_exception_throwing_override = True

    return args


def is_reduced_ops_build(args):
    return args.include_ops_by_config is not None


def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    if command_or_path and command_or_path.strip():
        executable_path = shutil.which(command_or_path)
        if executable_path is None:
            raise BuildError("Failed to resolve executable path for " "'{}'.".format(command_or_path))
        return os.path.abspath(executable_path)
    else:
        return None


def get_linux_distro():
    try:
        with open("/etc/os-release", "r") as f:
            dist_info = dict(line.strip().split("=", 1) for line in f.readlines())
        return dist_info.get("NAME", "").strip('"'), dist_info.get("VERSION", "").strip('"')
    except (IOError, ValueError):
        return "", ""


def is_ubuntu_1604():
    dist, ver = get_linux_distro()
    return dist == "Ubuntu" and ver.startswith("16.04")


def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)


def run_subprocess(
    args, cwd=None, capture_stdout=False, dll_path=None, shell=False, env={}, python_path=None, cuda_home=None
):
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            if "PATH" in my_env:
                my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
            else:
                my_env["PATH"] = dll_path
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path
    # Add nvcc's folder to PATH env so that our cmake file can find nvcc
    if cuda_home:
        my_env["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + my_env["PATH"]

    if python_path:
        if "PYTHONPATH" in my_env:
            my_env["PYTHONPATH"] += os.pathsep + python_path
        else:
            my_env["PYTHONPATH"] = python_path

    my_env.update(env)

    return run(*args, cwd=cwd, capture_stdout=capture_stdout, shell=shell, env=my_env)


def update_submodules(source_dir):
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=source_dir)
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir)


def is_docker():
    path = "/proc/self/cgroup"
    return os.path.exists("/.dockerenv") or os.path.isfile(path) and any("docker" in line for line in open(path))


def install_python_deps(numpy_version=""):
    dep_packages = ["setuptools", "wheel", "pytest"]
    dep_packages.append("numpy=={}".format(numpy_version) if numpy_version else "numpy>=1.16.6")
    dep_packages.append("sympy>=1.10")
    dep_packages.append("packaging")
    dep_packages.append("cerberus")
    run_subprocess([sys.executable, "-m", "pip", "install"] + dep_packages)


def setup_test_data(source_onnx_model_dir, dest_model_dir_name, build_dir, configs):
    # create the symlink/shortcut of onnx models dir under build_dir
    # currently, there're 2 sources of onnx models, one is build in OS image, another is
    # from {source_dir}/js/test, which is downloaded from onnx web.
    if is_windows():
        src_model_dir = os.path.join(build_dir, dest_model_dir_name)
        if os.path.exists(source_onnx_model_dir) and not os.path.exists(src_model_dir):
            log.debug("creating shortcut %s -> %s" % (source_onnx_model_dir, src_model_dir))
            run_subprocess(["mklink", "/D", "/J", src_model_dir, source_onnx_model_dir], shell=True)
        for config in configs:
            config_build_dir = get_config_build_dir(build_dir, config)
            os.makedirs(config_build_dir, exist_ok=True)
            dest_model_dir = os.path.join(config_build_dir, dest_model_dir_name)
            if os.path.exists(source_onnx_model_dir) and not os.path.exists(dest_model_dir):
                log.debug("creating shortcut %s -> %s" % (source_onnx_model_dir, dest_model_dir))
                run_subprocess(["mklink", "/D", "/J", dest_model_dir, source_onnx_model_dir], shell=True)
            elif os.path.exists(src_model_dir) and not os.path.exists(dest_model_dir):
                log.debug("creating shortcut %s -> %s" % (src_model_dir, dest_model_dir))
                run_subprocess(["mklink", "/D", "/J", dest_model_dir, src_model_dir], shell=True)
    else:
        src_model_dir = os.path.join(build_dir, dest_model_dir_name)
        if os.path.exists(source_onnx_model_dir) and not os.path.exists(src_model_dir):
            log.debug(f"create symlink {source_onnx_model_dir} -> {src_model_dir}")
            os.symlink(source_onnx_model_dir, src_model_dir, target_is_directory=True)


def use_dev_mode(args):
    if args.compile_no_warning_as_error:
        return False
    if args.use_acl:
        return False
    if args.use_armnn:
        return False
    if args.ios and is_macOS():
        return False
    SYSTEM_COLLECTIONURI = os.getenv("SYSTEM_COLLECTIONURI")
    if SYSTEM_COLLECTIONURI and not SYSTEM_COLLECTIONURI == "https://dev.azure.com/onnxruntime/":
        return False
    return True


def add_default_definition(definition_list, key, default_value):
    for x in definition_list:
        if x.startswith(key + "="):
            return definition_list
    definition_list.append(key + "=" + default_value)


def normalize_arg_list(nested_list):
    return [i for j in nested_list for i in j] if nested_list else []


def generate_build_tree(
    cmake_path,
    source_dir,
    build_dir,
    cuda_home,
    cudnn_home,
    rocm_home,
    mpi_home,
    nccl_home,
    tensorrt_home,
    migraphx_home,
    acl_home,
    acl_libs,
    armnn_home,
    armnn_libs,
    snpe_root,
    cann_home,
    path_to_protoc_exe,
    configs,
    cmake_extra_defines,
    args,
    cmake_extra_args,
):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    cmake_args = [cmake_path, cmake_dir]
    if not use_dev_mode(args):
        cmake_args += ["--compile-no-warning-as-error"]

    cmake_args += [
        "-Donnxruntime_RUN_ONNX_TESTS=" + ("ON" if args.enable_onnx_tests else "OFF"),
        "-Donnxruntime_GENERATE_TEST_REPORTS=ON",
        # There are two ways of locating python C API header file. "find_package(PythonLibs 3.5 REQUIRED)"
        # and "find_package(Python 3.5 COMPONENTS Development.Module)". The first one is deprecated and it
        # depends on the "PYTHON_EXECUTABLE" variable. The second needs "Python_EXECUTABLE". Here we set both
        # of them to get the best compatibility.
        "-DPython_EXECUTABLE=" + sys.executable,
        "-DPYTHON_EXECUTABLE=" + sys.executable,
        "-Donnxruntime_USE_MIMALLOC=" + ("ON" if args.use_mimalloc else "OFF"),
        "-Donnxruntime_ENABLE_PYTHON=" + ("ON" if args.enable_pybind else "OFF"),
        "-Donnxruntime_BUILD_CSHARP=" + ("ON" if args.build_csharp else "OFF"),
        "-Donnxruntime_BUILD_JAVA=" + ("ON" if args.build_java else "OFF"),
        "-Donnxruntime_BUILD_NODEJS=" + ("ON" if args.build_nodejs else "OFF"),
        "-Donnxruntime_BUILD_OBJC=" + ("ON" if args.build_objc else "OFF"),
        "-Donnxruntime_BUILD_SHARED_LIB=" + ("ON" if args.build_shared_lib else "OFF"),
        "-Donnxruntime_BUILD_APPLE_FRAMEWORK=" + ("ON" if args.build_apple_framework else "OFF"),
        "-Donnxruntime_USE_DNNL=" + ("ON" if args.use_dnnl else "OFF"),
        "-Donnxruntime_USE_NNAPI_BUILTIN=" + ("ON" if args.use_nnapi else "OFF"),
        "-Donnxruntime_USE_RKNPU=" + ("ON" if args.use_rknpu else "OFF"),
        "-Donnxruntime_USE_LLVM=" + ("ON" if args.use_tvm else "OFF"),
        "-Donnxruntime_ENABLE_MICROSOFT_INTERNAL=" + ("ON" if args.enable_msinternal else "OFF"),
        "-Donnxruntime_USE_VITISAI=" + ("ON" if args.use_vitisai else "OFF"),
        "-Donnxruntime_USE_TENSORRT=" + ("ON" if args.use_tensorrt else "OFF"),
        "-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=" + ("ON" if args.use_tensorrt_builtin_parser else "OFF"),
        "-Donnxruntime_TENSORRT_PLACEHOLDER_BUILDER=" + ("ON" if args.tensorrt_placeholder_builder else "OFF"),
        # set vars for TVM
        "-Donnxruntime_USE_TVM=" + ("ON" if args.use_tvm else "OFF"),
        "-Donnxruntime_TVM_CUDA_RUNTIME=" + ("ON" if args.use_tvm and args.tvm_cuda_runtime else "OFF"),
        "-Donnxruntime_TVM_USE_HASH=" + ("ON" if args.use_tvm_hash else "OFF"),
        # set vars for migraphx
        "-Donnxruntime_USE_MIGRAPHX=" + ("ON" if args.use_migraphx else "OFF"),
        # By default - we currently support only cross compiling for ARM/ARM64
        # (no native compilation supported through this script).
        "-Donnxruntime_CROSS_COMPILING=" + ("ON" if args.arm64 or args.arm64ec or args.arm else "OFF"),
        "-Donnxruntime_DISABLE_CONTRIB_OPS=" + ("ON" if args.disable_contrib_ops else "OFF"),
        "-Donnxruntime_DISABLE_ML_OPS=" + ("ON" if args.disable_ml_ops else "OFF"),
        "-Donnxruntime_DISABLE_RTTI="
        + ("ON" if args.disable_rtti or (args.minimal_build is not None and not args.enable_pybind) else "OFF"),
        "-Donnxruntime_DISABLE_EXCEPTIONS=" + ("ON" if args.disable_exceptions else "OFF"),
        # Need to use 'is not None' with minimal_build check as it could be an empty list.
        "-Donnxruntime_MINIMAL_BUILD=" + ("ON" if args.minimal_build is not None else "OFF"),
        "-Donnxruntime_EXTENDED_MINIMAL_BUILD="
        + ("ON" if args.minimal_build and "extended" in args.minimal_build else "OFF"),
        "-Donnxruntime_MINIMAL_BUILD_CUSTOM_OPS="
        + (
            "ON"
            if (args.minimal_build is not None and ("custom_ops" in args.minimal_build or args.use_extensions))
            else "OFF"
        ),
        "-Donnxruntime_REDUCED_OPS_BUILD=" + ("ON" if is_reduced_ops_build(args) else "OFF"),
        "-Donnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=" + ("ON" if args.enable_language_interop_ops else "OFF"),
        "-Donnxruntime_USE_DML=" + ("ON" if args.use_dml else "OFF"),
        "-Donnxruntime_USE_WINML=" + ("ON" if args.use_winml else "OFF"),
        "-Donnxruntime_BUILD_MS_EXPERIMENTAL_OPS=" + ("ON" if args.ms_experimental else "OFF"),
        "-Donnxruntime_USE_TELEMETRY=" + ("ON" if args.use_telemetry else "OFF"),
        "-Donnxruntime_ENABLE_LTO=" + ("ON" if args.enable_lto else "OFF"),
        "-Donnxruntime_USE_ACL=" + ("ON" if args.use_acl else "OFF"),
        "-Donnxruntime_USE_ACL_1902=" + ("ON" if args.use_acl == "ACL_1902" else "OFF"),
        "-Donnxruntime_USE_ACL_1905=" + ("ON" if args.use_acl == "ACL_1905" else "OFF"),
        "-Donnxruntime_USE_ACL_1908=" + ("ON" if args.use_acl == "ACL_1908" else "OFF"),
        "-Donnxruntime_USE_ACL_2002=" + ("ON" if args.use_acl == "ACL_2002" else "OFF"),
        "-Donnxruntime_USE_ARMNN=" + ("ON" if args.use_armnn else "OFF"),
        "-Donnxruntime_ARMNN_RELU_USE_CPU=" + ("OFF" if args.armnn_relu else "ON"),
        "-Donnxruntime_ARMNN_BN_USE_CPU=" + ("OFF" if args.armnn_bn else "ON"),
        # Training related flags
        "-Donnxruntime_ENABLE_NVTX_PROFILE=" + ("ON" if args.enable_nvtx_profile else "OFF"),
        "-Donnxruntime_ENABLE_TRAINING=" + ("ON" if args.enable_training else "OFF"),
        "-Donnxruntime_ENABLE_TRAINING_OPS=" + ("ON" if args.enable_training_ops else "OFF"),
        "-Donnxruntime_ENABLE_TRAINING_TORCH_INTEROP=" + ("ON" if args.enable_training_torch_interop else "OFF"),
        "-Donnxruntime_ENABLE_TRAINING_ON_DEVICE=" + ("ON" if args.enable_training_on_device else "OFF"),
        # Enable advanced computations such as AVX for some traininig related ops.
        "-Donnxruntime_ENABLE_CPU_FP16_OPS=" + ("ON" if args.enable_training else "OFF"),
        "-Donnxruntime_USE_NCCL=" + ("ON" if args.enable_training and not args.disable_nccl else "OFF"),
        "-Donnxruntime_BUILD_BENCHMARKS=" + ("ON" if args.build_micro_benchmarks else "OFF"),
        "-Donnxruntime_USE_ROCM=" + ("ON" if args.use_rocm else "OFF"),
        "-DOnnxruntime_GCOV_COVERAGE=" + ("ON" if args.code_coverage else "OFF"),
        "-Donnxruntime_USE_MPI=" + ("ON" if args.use_mpi else "OFF"),
        "-Donnxruntime_ENABLE_MEMORY_PROFILE=" + ("ON" if args.enable_memory_profile else "OFF"),
        "-Donnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO=" + ("ON" if args.enable_cuda_line_info else "OFF"),
        "-Donnxruntime_BUILD_WEBASSEMBLY=" + ("ON" if args.build_wasm else "OFF"),
        "-Donnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB=" + ("ON" if args.build_wasm_static_lib else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING="
        + ("OFF" if args.disable_wasm_exception_catching else "ON"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_API_EXCEPTION_CATCHING="
        + ("ON" if args.enable_wasm_api_exception_catching else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING="
        + ("ON" if args.enable_wasm_exception_throwing_override else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_THREADS=" + ("ON" if args.enable_wasm_threads else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO=" + ("ON" if args.enable_wasm_debug_info else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_PROFILING=" + ("ON" if args.enable_wasm_profiling else "OFF"),
        "-Donnxruntime_ENABLE_EAGER_MODE=" + ("ON" if args.build_eager_mode else "OFF"),
        "-Donnxruntime_ENABLE_LAZY_TENSOR=" + ("ON" if args.enable_lazy_tensor else "OFF"),
        "-Donnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS="
        + ("ON" if args.enable_external_custom_op_schemas else "OFF"),
        "-Donnxruntime_ENABLE_CUDA_PROFILING=" + ("ON" if args.enable_cuda_profiling else "OFF"),
        "-Donnxruntime_ENABLE_ROCM_PROFILING=" + ("ON" if args.enable_rocm_profiling else "OFF"),
        "-Donnxruntime_USE_XNNPACK=" + ("ON" if args.use_xnnpack else "OFF"),
        "-Donnxruntime_USE_CANN=" + ("ON" if args.use_cann else "OFF"),
    ]
    # By default cmake does not check TLS/SSL certificates. Here we turn it on.
    # But, in some cases you may also need to supply a CA file.
    add_default_definition(cmake_extra_defines, "CMAKE_TLS_VERIFY", "ON")
    add_default_definition(cmake_extra_defines, "FETCHCONTENT_QUIET", "OFF")
    if args.external_graph_transformer_path:
        cmake_args.append("-Donnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH=" + args.external_graph_transformer_path)
    if args.use_winml:
        cmake_args.append("-Donnxruntime_BUILD_WINML_TESTS=" + ("OFF" if args.skip_winml_tests else "ON"))
    if args.use_dnnl:
        cmake_args.append("-Donnxruntime_DNNL_GPU_RUNTIME=" + args.dnnl_gpu_runtime)
        cmake_args.append("-Donnxruntime_DNNL_OPENCL_ROOT=" + args.dnnl_opencl_root)
    if args.build_wasm:
        cmake_args.append("-Donnxruntime_ENABLE_WEBASSEMBLY_SIMD=" + ("ON" if args.enable_wasm_simd else "OFF"))
    if args.use_migraphx:
        cmake_args.append("-Donnxruntime_MIGRAPHX_HOME=" + migraphx_home)
    if args.use_cuda:
        cmake_args.append("-Donnxruntime_NVCC_THREADS=" + str(args.parallel))
    if args.use_rocm:
        cmake_args.append("-Donnxruntime_ROCM_HOME=" + rocm_home)
        cmake_args.append("-Donnxruntime_ROCM_VERSION=" + args.rocm_version)
    if args.use_tensorrt:
        cmake_args.append("-Donnxruntime_TENSORRT_HOME=" + tensorrt_home)
    if args.llvm_config:
        cmake_args.append("-Donnxruntime_TVM_USE_LLVM=" + args.llvm_config)

    if args.use_cuda:
        add_default_definition(cmake_extra_defines, "onnxruntime_USE_CUDA", "ON")
        if args.cuda_version:
            add_default_definition(cmake_extra_defines, "onnxruntime_CUDA_VERSION", args.cuda_version)
        # TODO: this variable is not really needed
        add_default_definition(cmake_extra_defines, "onnxruntime_CUDA_HOME", cuda_home)
        if cudnn_home:
            add_default_definition(cmake_extra_defines, "onnxruntime_CUDNN_HOME", cudnn_home)

    if is_windows():
        if args.enable_msvc_static_runtime:
            add_default_definition(
                cmake_extra_defines, "CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            )
            add_default_definition(cmake_extra_defines, "ONNX_USE_MSVC_STATIC_RUNTIME", "ON")
            add_default_definition(cmake_extra_defines, "protobuf_MSVC_STATIC_RUNTIME", "ON")
            add_default_definition(cmake_extra_defines, "gtest_force_shared_crt", "OFF")
        else:
            # CMAKE_MSVC_RUNTIME_LIBRARY is default to MultiThreaded$<$<CONFIG:Debug>:Debug>DLL
            add_default_definition(cmake_extra_defines, "ONNX_USE_MSVC_STATIC_RUNTIME", "OFF")
            add_default_definition(cmake_extra_defines, "protobuf_MSVC_STATIC_RUNTIME", "OFF")
            add_default_definition(cmake_extra_defines, "gtest_force_shared_crt", "ON")

    if acl_home and os.path.exists(acl_home):
        cmake_args += ["-Donnxruntime_ACL_HOME=" + acl_home]

    if acl_libs and os.path.exists(acl_libs):
        cmake_args += ["-Donnxruntime_ACL_LIBS=" + acl_libs]

    if armnn_home and os.path.exists(armnn_home):
        cmake_args += ["-Donnxruntime_ARMNN_HOME=" + armnn_home]

    if armnn_libs and os.path.exists(armnn_libs):
        cmake_args += ["-Donnxruntime_ARMNN_LIBS=" + armnn_libs]

    if mpi_home and os.path.exists(mpi_home):
        if args.use_mpi:
            cmake_args += ["-Donnxruntime_MPI_HOME=" + mpi_home]
        else:
            log.warning(
                "mpi_home is supplied but use_mpi is set to false."
                " Build will continue without linking MPI libraries."
            )

    if nccl_home and os.path.exists(nccl_home):
        cmake_args += ["-Donnxruntime_NCCL_HOME=" + nccl_home]

    if snpe_root and os.path.exists(snpe_root):
        cmake_args += ["-DSNPE_ROOT=" + snpe_root]

    if cann_home and os.path.exists(cann_home):
        cmake_args += ["-Donnxruntime_CANN_HOME=" + cann_home]

    if args.winml_root_namespace_override:
        cmake_args += ["-Donnxruntime_WINML_NAMESPACE_OVERRIDE=" + args.winml_root_namespace_override]
    if args.use_openvino:
        cmake_args += [
            "-Donnxruntime_USE_OPENVINO=ON",
            "-Donnxruntime_USE_OPENVINO_MYRIAD=" + ("ON" if args.use_openvino == "MYRIAD_FP16" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_GPU_FP32=" + ("ON" if args.use_openvino == "GPU_FP32" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_GPU_FP16=" + ("ON" if args.use_openvino == "GPU_FP16" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_CPU_FP32=" + ("ON" if args.use_openvino == "CPU_FP32" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_CPU_FP16=" + ("ON" if args.use_openvino == "CPU_FP16" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_VAD_M=" + ("ON" if args.use_openvino == "VAD-M_FP16" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_VAD_F=" + ("ON" if args.use_openvino == "VAD-F_FP32" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_MYRIAD_NP="
            + ("ON" if args.use_openvino == "MYRIAD_FP16_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_GPU_FP32_NP="
            + ("ON" if args.use_openvino == "GPU_FP32_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_GPU_FP16_NP="
            + ("ON" if args.use_openvino == "GPU_FP16_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_CPU_FP32_NP="
            + ("ON" if args.use_openvino == "CPU_FP32_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_CPU_FP16_NP="
            + ("ON" if args.use_openvino == "CPU_FP16_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_VAD_M_NP="
            + ("ON" if args.use_openvino == "VAD-M_FP16_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_VAD_F_NP="
            + ("ON" if args.use_openvino == "VAD-F_FP32_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_HETERO=" + ("ON" if args.use_openvino.startswith("HETERO") else "OFF"),
            "-Donnxruntime_USE_OPENVINO_DEVICE=" + (args.use_openvino),
            "-Donnxruntime_USE_OPENVINO_MULTI=" + ("ON" if args.use_openvino.startswith("MULTI") else "OFF"),
            "-Donnxruntime_USE_OPENVINO_AUTO=" + ("ON" if args.use_openvino.startswith("AUTO") else "OFF"),
        ]

    # TensorRT and OpenVINO providers currently only support
    # full_protobuf option.
    if args.use_full_protobuf or args.use_tensorrt or args.use_openvino or args.use_vitisai or args.gen_doc:
        cmake_args += ["-Donnxruntime_USE_FULL_PROTOBUF=ON", "-DProtobuf_USE_STATIC_LIBS=ON"]

    if args.use_tvm and args.llvm_path is not None:
        cmake_args += ["-DLLVM_DIR=%s" % args.llvm_path]

    if args.use_cuda and not is_windows():
        nvml_stub_path = cuda_home + "/lib64/stubs"
        cmake_args += ["-DCUDA_CUDA_LIBRARY=" + nvml_stub_path]

    if args.use_preinstalled_eigen:
        cmake_args += ["-Donnxruntime_USE_PREINSTALLED_EIGEN=ON", "-Deigen_SOURCE_PATH=" + args.eigen_path]

    if args.nnapi_min_api:
        cmake_args += ["-Donnxruntime_NNAPI_MIN_API=" + str(args.nnapi_min_api)]

    if args.android:
        if not args.android_ndk_path:
            raise BuildError("android_ndk_path required to build for Android")
        if not args.android_sdk_path:
            raise BuildError("android_sdk_path required to build for Android")
        cmake_args += [
            "-DCMAKE_TOOLCHAIN_FILE="
            + os.path.join(args.android_ndk_path, "build", "cmake", "android.toolchain.cmake"),
            "-DANDROID_PLATFORM=android-" + str(args.android_api),
            "-DANDROID_ABI=" + str(args.android_abi),
            "-DANDROID_MIN_SDK=" + str(args.android_api),
        ]

        if args.android_cpp_shared:
            cmake_args += ["-DANDROID_STL=c++_shared"]

    if args.dml_path:
        cmake_args += [
            "-Donnxruntime_USE_CUSTOM_DIRECTML=ON",
            "-Ddml_INCLUDE_DIR=" + os.path.join(args.dml_path, "include"),
            "-Ddml_LIB_DIR=" + os.path.join(args.dml_path, "lib"),
        ]

    if args.dml_external_project:
        cmake_args += [
            "-Donnxruntime_USE_CUSTOM_DIRECTML=ON",
            "-Ddml_EXTERNAL_PROJECT=ON",
        ]

    if args.use_gdk:
        cmake_args += [
            "-DCMAKE_TOOLCHAIN_FILE=" + os.path.join(source_dir, "cmake", "gdk_toolchain.cmake"),
            "-DGDK_EDITION=" + args.gdk_edition,
            "-DGDK_PLATFORM=" + args.gdk_platform,
            "-Donnxruntime_BUILD_UNIT_TESTS=OFF",  # gtest doesn't build for GDK
        ]
        if args.use_dml and not (args.dml_path or args.dml_external_project):
            raise BuildError("You must set dml_path or dml_external_project when building with the GDK.")

    if is_macOS() and not args.android:
        cmake_args += ["-DCMAKE_OSX_ARCHITECTURES=" + args.osx_arch]
        if args.use_xcode:
            cmake_ver = LooseVersion(subprocess.check_output(["cmake", "--version"]).decode("utf-8").split()[2])
            xcode_ver = LooseVersion(
                subprocess.check_output(["xcrun", "xcodebuild", "-version"]).decode("utf-8").split()[1]
            )
            # Requires Cmake 3.21.1+ for XCode 13+
            # The legacy build system is not longer supported on XCode 13+
            if xcode_ver >= LooseVersion("13") and cmake_ver < LooseVersion("3.21.1"):
                raise BuildError("CMake 3.21.1+ required to use XCode 13+")
            # Use legacy build system for old CMake [3.19, 3.21.1) which uses new build system by default
            # CMake 3.18- use the legacy build system by default
            if cmake_ver >= LooseVersion("3.19.0") and cmake_ver < LooseVersion("3.21.1"):
                cmake_args += ["-T", "buildsystem=1"]
        if args.apple_deploy_target:
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=" + args.apple_deploy_target]
        # Code sign the binaries, if the code signing development identity and/or team id are provided
        if args.xcode_code_signing_identity:
            cmake_args += ["-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=" + args.xcode_code_signing_identity]
        if args.xcode_code_signing_team_id:
            cmake_args += ["-DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=" + args.xcode_code_signing_team_id]

    if args.use_coreml:
        cmake_args += ["-Donnxruntime_USE_COREML=ON"]

    if args.use_snpe:
        cmake_args += ["-Donnxruntime_USE_SNPE=ON"]

    if args.ios:
        needed_args = [
            args.use_xcode,
            args.ios_sysroot,
            args.apple_deploy_target,
        ]
        arg_names = [
            "--use_xcode            " + "<need use xcode to cross build iOS on MacOS>",
            "--ios_sysroot          " + "<the location or name of the macOS platform SDK>",
            "--apple_deploy_target  " + "<the minimum version of the target platform>",
        ]
        if not all(needed_args):
            raise BuildError(
                "iOS build on MacOS canceled due to missing arguments: "
                + ", ".join(val for val, cond in zip(arg_names, needed_args) if not cond)
            )
        cmake_args += [
            "-DCMAKE_SYSTEM_NAME=iOS",
            "-Donnxruntime_BUILD_SHARED_LIB=ON",
            "-DCMAKE_OSX_SYSROOT=" + args.ios_sysroot,
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + args.apple_deploy_target,
            # we do not need protoc binary for ios cross build
            "-Dprotobuf_BUILD_PROTOC_BINARIES=OFF",
            "-DCMAKE_TOOLCHAIN_FILE="
            + (args.ios_toolchain_file if args.ios_toolchain_file else "../cmake/onnxruntime_ios.toolchain.cmake"),
        ]

    if args.build_wasm:
        emsdk_dir = os.path.join(cmake_dir, "external", "emsdk")
        emscripten_cmake_toolchain_file = os.path.join(
            emsdk_dir, "upstream", "emscripten", "cmake", "Modules", "Platform", "Emscripten.cmake"
        )
        cmake_args += ["-DCMAKE_TOOLCHAIN_FILE=" + emscripten_cmake_toolchain_file]
        if args.disable_wasm_exception_catching:
            # WebAssembly unittest requires exception catching to work. If this feature is disabled, we do not build
            # unit test.
            cmake_args += [
                "-Donnxruntime_BUILD_UNIT_TESTS=OFF",
            ]

        # add default emscripten settings
        emscripten_settings = normalize_arg_list(args.emscripten_settings)

        # set -s MALLOC
        if args.wasm_malloc is not None:
            add_default_definition(emscripten_settings, "MALLOC", args.wasm_malloc)
        add_default_definition(emscripten_settings, "MALLOC", "dlmalloc")

        if emscripten_settings:
            cmake_args += [f"-Donnxruntime_EMSCRIPTEN_SETTINGS={';'.join(emscripten_settings)}"]

    # Append onnxruntime-extensions cmake options
    if args.use_extensions:
        cmake_args += ["-Donnxruntime_USE_EXTENSIONS=ON"]

        # default path of onnxruntime-extensions, using git submodule
        onnxruntime_extensions_path = os.path.join(cmake_dir, "external", "onnxruntime-extensions")

        if args.extensions_overridden_path and os.path.exists(args.extensions_overridden_path):
            # use absolute path here because onnxruntime-extensions is outside onnxruntime
            onnxruntime_extensions_path = os.path.abspath(args.extensions_overridden_path)

        cmake_args += ["-Donnxruntime_EXTENSIONS_PATH=" + onnxruntime_extensions_path]
        print("[onnxruntime-extensions] onnxruntime_extensions_path: ", onnxruntime_extensions_path)

        if is_reduced_ops_build(args):
            operators_config_file = os.path.abspath(args.include_ops_by_config)
            cmake_tool_dir = os.path.join(onnxruntime_extensions_path, "tools")

            # generate _selectedoplist.cmake by operators config file
            run_subprocess([sys.executable, "gen_selectedops.py", operators_config_file], cwd=cmake_tool_dir)

    if path_to_protoc_exe:
        cmake_args += ["-DONNX_CUSTOM_PROTOC_EXECUTABLE=%s" % path_to_protoc_exe]

    if args.fuzz_testing:
        if not (
            args.build_shared_lib
            and is_windows()
            and args.cmake_generator == "Visual Studio 16 2019"
            and args.use_full_protobuf
        ):
            raise BuildError("Fuzz test has only be tested with build shared libs option using MSVC on windows")
        cmake_args += [
            "-Donnxruntime_BUILD_UNIT_TESTS=ON",
            "-Donnxruntime_FUZZ_TEST=ON",
            "-Donnxruntime_USE_FULL_PROTOBUF=ON",
        ]

    if args.gen_doc:
        if args.enable_training:
            raise BuildError("--gen_doc is not supported along with --enable_training")
        add_default_definition(cmake_extra_defines, "onnxruntime_PYBIND_EXPORT_OPSCHEMA", "ON")
    else:
        add_default_definition(cmake_extra_defines, "onnxruntime_PYBIND_EXPORT_OPSCHEMA", "OFF")

    if args.build_eager_mode or args.enable_lazy_tensor:
        import torch

        cmake_args += ["-Donnxruntime_PREBUILT_PYTORCH_PATH=%s" % os.path.dirname(torch.__file__)]
        cmake_args += ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]

    cmake_args += ["-D{}".format(define) for define in cmake_extra_defines]

    cmake_args += cmake_extra_args

    # ADO pipelines will store the pipeline build number
    # (e.g. 191101-2300.1.master) and source version in environment
    # variables. If present, use these values to define the
    # WinML/ORT DLL versions.
    build_number = os.getenv("Build_BuildNumber")
    source_version = os.getenv("Build_SourceVersion")
    if build_number and source_version:
        build_matches = re.fullmatch(r"(\d\d)(\d\d)(\d\d)(\d\d)\.(\d+)", build_number)
        if build_matches:
            YY = build_matches.group(2)
            MM = build_matches.group(3)
            DD = build_matches.group(4)

            # Get ORT major and minor number
            with open(os.path.join(source_dir, "VERSION_NUMBER")) as f:
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
                    "-DVERSION_STRING={}.{}.{}.{}".format(ort_major, ort_minor, build_number, source_version[0:7]),
                ]

    for config in configs:
        config_build_dir = get_config_build_dir(build_dir, config)
        os.makedirs(config_build_dir, exist_ok=True)
        if args.use_tvm:
            os.environ["PATH"] = (
                os.path.join(config_build_dir, "_deps", "tvm-build")
                + os.pathsep
                + os.path.join(config_build_dir, "_deps", "tvm-src")
                + os.pathsep
                + os.path.dirname(sys.executable)
                + os.pathsep
                + os.environ["PATH"]
            )
        preinstalled_dir = Path(build_dir) / config
        run_subprocess(
            cmake_args
            + [
                "-Donnxruntime_ENABLE_MEMLEAK_CHECKER="
                + (
                    "ON"
                    if config.lower() == "debug"
                    and not args.use_tvm
                    and not args.use_openvino
                    and not args.use_gdk
                    and not args.enable_msvc_static_runtime
                    and not args.disable_memleak_checker
                    else "OFF"
                ),
                "-DCMAKE_BUILD_TYPE={}".format(config),
                "-DCMAKE_PREFIX_PATH={}/{}/installed".format(build_dir, config)
                if preinstalled_dir.exists() and not (args.arm64 or args.arm64ec or args.arm)
                else "",
            ],
            cwd=config_build_dir,
            cuda_home=cuda_home,
        )


def clean_targets(cmake_path, build_dir, configs):
    for config in configs:
        log.info("Cleaning targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path, "--build", build_dir2, "--config", config, "--target", "clean"]

        run_subprocess(cmd_args)


def build_targets(args, cmake_path, build_dir, configs, num_parallel_jobs, target=None):
    for config in configs:
        log.info("Building targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path, "--build", build_dir2, "--config", config]
        if target:
            cmd_args.extend(["--target", target])

        build_tool_args = []
        if num_parallel_jobs != 1:
            if is_windows() and args.cmake_generator != "Ninja" and not args.build_wasm:
                build_tool_args += [
                    "/maxcpucount:{}".format(num_parallel_jobs),
                    # if nodeReuse is true, msbuild processes will stay around for a bit after the build completes
                    "/nodeReuse:False",
                ]
            elif is_macOS() and args.use_xcode:
                # CMake will generate correct build tool args for Xcode
                cmd_args += ["--parallel", str(num_parallel_jobs)]
            else:
                build_tool_args += ["-j{}".format(num_parallel_jobs)]

        if build_tool_args:
            cmd_args += ["--"]
            cmd_args += build_tool_args

        env = {}
        if args.android:
            env["ANDROID_SDK_ROOT"] = args.android_sdk_path
            env["ANDROID_NDK_HOME"] = args.android_ndk_path

        run_subprocess(cmd_args, env=env)


def add_dir_if_exists(directory, dir_list):
    if os.path.isdir(directory):
        dir_list.append(directory)


def setup_cuda_vars(args):
    cuda_home = ""
    cudnn_home = ""

    if args.use_cuda:
        cuda_home = args.cuda_home if args.cuda_home else os.getenv("CUDA_HOME")
        cudnn_home = args.cudnn_home if args.cudnn_home else os.getenv("CUDNN_HOME")

        cuda_home_valid = cuda_home is not None and os.path.exists(cuda_home)
        cudnn_home_valid = cudnn_home is not None and os.path.exists(cudnn_home)

        if not cuda_home_valid or (not is_windows() and not cudnn_home_valid):
            raise BuildError(
                "cuda_home and cudnn_home paths must be specified and valid.",
                "cuda_home='{}' valid={}. cudnn_home='{}' valid={}".format(
                    cuda_home, cuda_home_valid, cudnn_home, cudnn_home_valid
                ),
            )

    return cuda_home, cudnn_home


def setup_cann_vars(args):
    cann_home = ""

    if args.use_cann:
        cann_home = args.cann_home if args.cann_home else os.getenv("ASCEND_HOME_PATH")

        cann_home_valid = cann_home is not None and os.path.exists(cann_home)

        if not cann_home_valid:
            raise BuildError(
                "cann_home paths must be specified and valid.",
                "cann_home='{}' valid={}.".format(cann_home, cann_home_valid),
            )

    return cann_home


def setup_tensorrt_vars(args):
    tensorrt_home = ""
    if args.use_tensorrt:
        tensorrt_home = args.tensorrt_home if args.tensorrt_home else os.getenv("TENSORRT_HOME")
        tensorrt_home_valid = tensorrt_home is not None and os.path.exists(tensorrt_home)
        if not tensorrt_home_valid:
            raise BuildError(
                "tensorrt_home paths must be specified and valid.",
                "tensorrt_home='{}' valid={}.".format(tensorrt_home, tensorrt_home_valid),
            )

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

    if args.use_migraphx:
        print("migraphx_home = {}".format(args.migraphx_home))
        migraphx_home = args.migraphx_home or os.getenv("MIGRAPHX_HOME") or None

        migraphx_home_not_valid = migraphx_home and not os.path.exists(migraphx_home)

        if migraphx_home_not_valid:
            raise BuildError(
                "migraphx_home paths must be specified and valid.",
                "migraphx_home='{}' valid={}.".format(migraphx_home, migraphx_home_not_valid),
            )
    return migraphx_home or ""


def setup_dml_build(args, cmake_path, build_dir, configs):
    if not args.use_dml:
        return

    if args.dml_path:
        for expected_file in ["bin/DirectML.dll", "lib/DirectML.lib", "include/DirectML.h"]:
            file_path = os.path.join(args.dml_path, expected_file)
            if not os.path.exists(file_path):
                raise BuildError(
                    "dml_path is invalid.", "dml_path='{}' expected_file='{}'.".format(args.dml_path, file_path)
                )
    elif not args.dml_external_project:
        for config in configs:
            # Run the RESTORE_PACKAGES target to perform the initial
            # NuGet setup.
            cmd_args = [
                cmake_path,
                "--build",
                get_config_build_dir(build_dir, config),
                "--config",
                config,
                "--target",
                "RESTORE_PACKAGES",
            ]
            run_subprocess(cmd_args)


def setup_rocm_build(args):
    rocm_home = None
    if args.use_rocm:
        print("rocm_home = {}".format(args.rocm_home))
        rocm_home = args.rocm_home or None
        rocm_home_not_valid = rocm_home and not os.path.exists(rocm_home)
        if rocm_home_not_valid:
            raise BuildError(
                "rocm_home paths must be specified and valid.",
                "rocm_home='{}' valid={}.".format(rocm_home, rocm_home_not_valid),
            )
    return rocm_home or ""


def run_android_tests(args, source_dir, build_dir, config, cwd):
    sdk_tool_paths = android.get_sdk_tool_paths(args.android_sdk_path)
    device_dir = "/data/local/tmp"

    def adb_push(src, dest, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, "push", src, dest], **kwargs)

    def adb_shell(*args, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, "shell", *args], **kwargs)

    def adb_install(*args, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, "install", *args], **kwargs)

    def run_adb_shell(cmd):
        # GCOV_PREFIX_STRIP specifies the depth of the directory hierarchy to strip and
        # GCOV_PREFIX specifies the root directory
        # for creating the runtime code coverage files.
        if args.code_coverage:
            adb_shell(
                "cd {0} && GCOV_PREFIX={0} GCOV_PREFIX_STRIP={1} {2}".format(device_dir, cwd.count(os.sep) + 1, cmd)
            )
        else:
            adb_shell("cd {} && {}".format(device_dir, cmd))

    if args.android_abi == "x86_64":
        with contextlib.ExitStack() as context_stack:
            if args.android_run_emulator:
                avd_name = "ort_android"
                system_image = "system-images;android-{};google_apis;{}".format(args.android_api, args.android_abi)

                android.create_virtual_device(sdk_tool_paths, system_image, avd_name)
                emulator_proc = context_stack.enter_context(
                    android.start_emulator(
                        sdk_tool_paths=sdk_tool_paths,
                        avd_name=avd_name,
                        extra_args=["-partition-size", "2047", "-wipe-data"],
                    )
                )
                context_stack.callback(android.stop_emulator, emulator_proc)

            adb_push("testdata", device_dir, cwd=cwd)
            adb_push(
                os.path.join(source_dir, "cmake", "external", "onnx", "onnx", "backend", "test"), device_dir, cwd=cwd
            )
            adb_push("onnxruntime_test_all", device_dir, cwd=cwd)
            adb_shell("chmod +x {}/onnxruntime_test_all".format(device_dir))
            adb_push("onnx_test_runner", device_dir, cwd=cwd)
            adb_shell("chmod +x {}/onnx_test_runner".format(device_dir))
            run_adb_shell("{0}/onnxruntime_test_all".format(device_dir))

            if args.build_java:
                gradle_executable = "gradle"
                # use the gradle wrapper if it exists, the gradlew should be setup under <repo root>/java
                gradlew_path = os.path.join(source_dir, "java", "gradlew.bat" if is_windows() else "gradlew")
                if os.path.exists(gradlew_path):
                    gradle_executable = gradlew_path
                android_test_path = os.path.join(cwd, "java", "androidtest", "android")
                run_subprocess(
                    [
                        gradle_executable,
                        "--no-daemon",
                        "-DminSdkVer={}".format(args.android_api),
                        "clean",
                        "connectedDebugAndroidTest",
                    ],
                    cwd=android_test_path,
                )

            if args.use_nnapi:
                run_adb_shell("{0}/onnx_test_runner -e nnapi {0}/test".format(device_dir))
            else:
                run_adb_shell("{0}/onnx_test_runner {0}/test".format(device_dir))
            # run shared_lib_test if necessary
            if args.build_shared_lib:
                adb_push("libonnxruntime.so", device_dir, cwd=cwd)
                adb_push("onnxruntime_shared_lib_test", device_dir, cwd=cwd)
                adb_shell("chmod +x {}/onnxruntime_shared_lib_test".format(device_dir))
                run_adb_shell("LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{0} {0}/onnxruntime_shared_lib_test".format(device_dir))


def run_ios_tests(args, source_dir, config, cwd):
    run_subprocess(
        [
            "xcodebuild",
            "test-without-building",
            "-project",
            "./onnxruntime.xcodeproj",
            "-configuration",
            config,
            "-scheme",
            "onnxruntime_test_all_xc",
            "-destination",
            "platform=iOS Simulator,OS=latest,name=iPhone SE (2nd generation)",
        ],
        cwd=cwd,
    )

    run_subprocess(
        [
            "xcodebuild",
            "test-without-building",
            "-project",
            "./onnxruntime.xcodeproj",
            "-configuration",
            config,
            "-scheme",
            "onnxruntime_shared_lib_test_xc",
            "-destination",
            "platform=iOS Simulator,OS=latest,name=iPhone SE (2nd generation)",
        ],
        cwd=cwd,
    )

    if args.build_apple_framework:
        package_test_py = os.path.join(source_dir, "tools", "ci_build", "github", "apple", "test_ios_packages.py")
        framework_info_file = os.path.join(cwd, "framework_info.json")
        dynamic_framework_dir = os.path.join(cwd, config + "-" + args.ios_sysroot)
        static_framework_dir = os.path.join(cwd, config + "-" + args.ios_sysroot, "static_framework")
        # test dynamic framework
        run_subprocess(
            [
                sys.executable,
                package_test_py,
                "--c_framework_dir",
                dynamic_framework_dir,
                "--framework_info_file",
                framework_info_file,
            ],
            cwd=cwd,
        )
        # test static framework
        run_subprocess(
            [
                sys.executable,
                package_test_py,
                "--c_framework_dir",
                static_framework_dir,
                "--framework_info_file",
                framework_info_file,
            ],
            cwd=cwd,
        )


def run_orttraining_test_orttrainer_frontend_separately(cwd):
    class TestNameCollecterPlugin:
        def __init__(self):
            self.collected = set()

        def pytest_collection_modifyitems(self, items):
            for item in items:
                print("item.name: ", item.name)
                test_name = item.name
                start = test_name.find("[")
                if start > 0:
                    test_name = test_name[:start]
                self.collected.add(test_name)

    import pytest

    plugin = TestNameCollecterPlugin()
    test_script_filename = os.path.join(cwd, "orttraining_test_orttrainer_frontend.py")
    pytest.main(["--collect-only", test_script_filename], plugins=[plugin])

    for test_name in plugin.collected:
        run_subprocess(
            [sys.executable, "-m", "pytest", "orttraining_test_orttrainer_frontend.py", "-v", "-k", test_name], cwd=cwd
        )


def run_training_python_frontend_tests(cwd):
    # have to disable due to (with torchvision==0.9.1+cu102 which is required by ortmodule):
    # Downloading http://yann.lecun.com/exdb/mnist/
    # https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    # Failed to download (trying next):
    # HTTP Error 404: Not Found
    # run_subprocess([sys.executable, 'onnxruntime_test_ort_trainer.py'], cwd=cwd)

    run_subprocess([sys.executable, "onnxruntime_test_training_unit_tests.py"], cwd=cwd)
    run_subprocess(
        [
            sys.executable,
            "orttraining_test_transformers.py",
            "BertModelTest.test_for_pretraining_full_precision_list_input",
        ],
        cwd=cwd,
    )
    run_subprocess(
        [
            sys.executable,
            "orttraining_test_transformers.py",
            "BertModelTest.test_for_pretraining_full_precision_dict_input",
        ],
        cwd=cwd,
    )
    run_subprocess(
        [
            sys.executable,
            "orttraining_test_transformers.py",
            "BertModelTest.test_for_pretraining_full_precision_list_and_dict_input",
        ],
        cwd=cwd,
    )

    # TODO: use run_orttraining_test_orttrainer_frontend_separately to work around a sporadic segfault.
    # shall revert to run_subprocess call once the segfault issue is resolved.
    run_orttraining_test_orttrainer_frontend_separately(cwd)
    # run_subprocess([sys.executable, '-m', 'pytest', '-sv', 'orttraining_test_orttrainer_frontend.py'], cwd=cwd)

    run_subprocess([sys.executable, "-m", "pytest", "-sv", "orttraining_test_orttrainer_bert_toy_onnx.py"], cwd=cwd)

    run_subprocess([sys.executable, "-m", "pytest", "-sv", "orttraining_test_checkpoint_storage.py"], cwd=cwd)

    run_subprocess(
        [sys.executable, "-m", "pytest", "-sv", "orttraining_test_orttrainer_checkpoint_functions.py"], cwd=cwd
    )
    # Not technically training related, but it needs torch to be installed.
    run_subprocess([sys.executable, "-m", "pytest", "-sv", "test_pytorch_export_contrib_ops.py"], cwd=cwd)


def run_training_python_frontend_e2e_tests(cwd):
    # frontend tests are to be added here:
    log.info("Running python frontend e2e tests.")

    run_subprocess(
        [sys.executable, "orttraining_run_frontend_batch_size_test.py", "-v"],
        cwd=cwd,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )

    import torch

    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        bert_pretrain_script = "orttraining_run_bert_pretrain.py"
        # TODO: this test will be replaced with convergence test ported from backend
        log.debug(
            "RUN: mpirun -n {} "
            "-x"
            "NCCL_DEBUG=INFO"
            " {} {} {}".format(
                ngpus, sys.executable, bert_pretrain_script, "ORTBertPretrainTest.test_pretrain_convergence"
            )
        )
        run_subprocess(
            [
                "mpirun",
                "-n",
                str(ngpus),
                "-x",
                "NCCL_DEBUG=INFO",
                sys.executable,
                bert_pretrain_script,
                "ORTBertPretrainTest.test_pretrain_convergence",
            ],
            cwd=cwd,
        )

        log.debug("RUN: mpirun -n {} {} orttraining_run_glue.py".format(ngpus, sys.executable))
        run_subprocess(
            ["mpirun", "-n", str(ngpus), "-x", "NCCL_DEBUG=INFO", sys.executable, "orttraining_run_glue.py"], cwd=cwd
        )

    # with orttraining_run_glue.py.
    # 1. we like to force to use single GPU (with CUDA_VISIBLE_DEVICES)
    #   for fine-tune tests.
    # 2. need to run test separately (not to mix between fp16
    #   and full precision runs. this need to be investigated).
    run_subprocess(
        [sys.executable, "orttraining_run_glue.py", "ORTGlueTest.test_bert_with_mrpc", "-v"],
        cwd=cwd,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )

    run_subprocess(
        [sys.executable, "orttraining_run_glue.py", "ORTGlueTest.test_bert_fp16_with_mrpc", "-v"],
        cwd=cwd,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )

    run_subprocess(
        [sys.executable, "orttraining_run_glue.py", "ORTGlueTest.test_roberta_with_mrpc", "-v"],
        cwd=cwd,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )

    run_subprocess(
        [sys.executable, "orttraining_run_glue.py", "ORTGlueTest.test_roberta_fp16_with_mrpc", "-v"],
        cwd=cwd,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )

    run_subprocess(
        [sys.executable, "orttraining_run_multiple_choice.py", "ORTMultipleChoiceTest.test_bert_fp16_with_swag", "-v"],
        cwd=cwd,
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )

    run_subprocess([sys.executable, "onnxruntime_test_ort_trainer_with_mixed_precision.py"], cwd=cwd)

    run_subprocess(
        [sys.executable, "orttraining_test_transformers.py", "BertModelTest.test_for_pretraining_mixed_precision"],
        cwd=cwd,
    )


def run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs):
    for config in configs:
        log.info("Running tests for %s configuration", config)
        cwd = get_config_build_dir(build_dir, config)
        cwd = os.path.abspath(cwd)

        if args.android:
            run_android_tests(args, source_dir, build_dir, config, cwd)
            continue
        elif args.ios:
            run_ios_tests(args, source_dir, config, cwd)
            continue
        dll_path_list = []
        if args.use_tensorrt:
            dll_path_list.append(os.path.join(args.tensorrt_home, "lib"))
        # Adding the torch lib path for loading DLLs for onnxruntime in eager mode
        # This works for Python 3.7 and below, and doesn't work for Python 3.8+
        # User will need to import torch before onnxruntime and it will work for all versions
        if (args.build_eager_mode or args.enable_lazy_tensor) and is_windows():
            import torch

            dll_path_list.append(os.path.join(os.path.dirname(torch.__file__), "lib"))

        dll_path = None
        if len(dll_path_list) > 0:
            dll_path = os.pathsep.join(dll_path_list)

        if not ctest_path:
            if is_windows():
                # Get the "Google Test Adapter" for vstest.
                if not os.path.exists(os.path.join(cwd, "googletestadapter.0.17.1")):
                    run_subprocess(
                        [
                            "nuget.exe",
                            "restore",
                            os.path.join(source_dir, "packages.config"),
                            "-ConfigFile",
                            os.path.join(source_dir, "NuGet.config"),
                            "-PackagesDirectory",
                            cwd,
                        ]
                    )
                cwd2 = os.path.join(cwd, config)
                executables = ["onnxruntime_test_all.exe", "onnxruntime_mlas_test.exe"]
                if args.build_shared_lib:
                    executables.append("onnxruntime_shared_lib_test.exe")
                    executables.append("onnxruntime_global_thread_pools_test.exe")
                    executables.append("onnxruntime_api_tests_without_env.exe")
                run_subprocess(
                    [
                        "vstest.console.exe",
                        "--parallel",
                        "--TestAdapterPath:..\\googletestadapter.0.17.1\\build\\_common",  # noqa
                        "/Logger:trx",
                        "/Enablecodecoverage",
                        "/Platform:x64",
                        "/Settings:%s" % os.path.join(source_dir, "cmake\\codeconv.runsettings"),
                    ]
                    + executables,
                    cwd=cwd2,
                    dll_path=dll_path,
                )
            else:
                executables = ["onnxruntime_test_all", "onnxruntime_mlas_test"]
                if args.build_shared_lib:
                    executables.append("onnxruntime_shared_lib_test")
                    executables.append("onnxruntime_global_thread_pools_test")
                    executables.append("onnxruntime_api_tests_without_env")
                for exe in executables:
                    run_subprocess([os.path.join(cwd, exe)], cwd=cwd, dll_path=dll_path)

        else:
            ctest_cmd = [ctest_path, "--build-config", config, "--verbose", "--timeout", "10800"]
            run_subprocess(ctest_cmd, cwd=cwd, dll_path=dll_path)

        if args.enable_pybind:
            # Disable python tests for TensorRT on Windows due to need to enable placeholder builder
            # to reduce test times.
            if args.use_tensorrt and is_windows():
                return

            python_path = None
            if args.use_tvm:
                python_path = str((Path(build_dir) / config / "_deps" / "tvm-src" / "python").resolve())

            # Disable python tests in a reduced build as we don't know which ops have been included and which
            # models can run.
            if is_reduced_ops_build(args) or args.minimal_build is not None:
                return

            if is_windows():
                cwd = os.path.join(cwd, config)

            run_subprocess(
                [sys.executable, "onnxruntime_test_python.py"], cwd=cwd, dll_path=dll_path, python_path=python_path
            )

            if not args.disable_contrib_ops:
                run_subprocess([sys.executable, "onnxruntime_test_python_sparse_matmul.py"], cwd=cwd, dll_path=dll_path)

            if args.enable_symbolic_shape_infer_tests:
                run_subprocess(
                    [sys.executable, "onnxruntime_test_python_symbolic_shape_infer.py"], cwd=cwd, dll_path=dll_path
                )

            # For CUDA enabled builds test IOBinding feature
            if args.use_cuda:
                # We need to have Torch installed to test the IOBinding feature
                # which currently uses Torch's allocator to allocate GPU memory for testing
                log.info("Testing IOBinding feature")
                run_subprocess([sys.executable, "onnxruntime_test_python_iobinding.py"], cwd=cwd, dll_path=dll_path)

                log.info("Testing CUDA Graph feature")
                run_subprocess([sys.executable, "onnxruntime_test_python_cudagraph.py"], cwd=cwd, dll_path=dll_path)

            if not args.disable_ml_ops and not args.use_tensorrt:
                run_subprocess([sys.executable, "onnxruntime_test_python_mlops.py"], cwd=cwd, dll_path=dll_path)

            # The following test has multiple failures on Windows
            if args.enable_training and args.use_cuda and not is_windows():
                # run basic frontend tests
                run_training_python_frontend_tests(cwd=cwd)

            if args.build_eager_mode:
                # run eager mode test
                args_list = [sys.executable, os.path.join(cwd, "eager_test")]
                run_subprocess(args_list, cwd=cwd, dll_path=dll_path, python_path=cwd)
                if args.test_external_transformer_example:
                    run_subprocess(
                        [
                            sys.executable,
                            os.path.join(
                                source_dir,
                                "orttraining",
                                "orttraining",
                                "test",
                                "external_transformer",
                                "test",
                                "external_transformers_test.py",
                            ),
                        ],
                        cwd=cwd,
                        dll_path=dll_path,
                    )

            try:
                import onnx  # noqa

                onnx_test = True
            except ImportError as error:
                log.exception(error)
                log.warning("onnx is not installed. The ONNX tests will be skipped.")
                onnx_test = False

            if onnx_test:
                # Disable python onnx tests for TensorRT because many tests are
                # not supported yet.
                if args.use_tensorrt:
                    return

                run_subprocess(
                    [sys.executable, "onnxruntime_test_python_backend.py"],
                    cwd=cwd,
                    dll_path=dll_path,
                    python_path=python_path,
                )
                if not args.disable_contrib_ops:
                    run_subprocess(
                        [sys.executable, "-m", "unittest", "discover", "-s", "quantization"], cwd=cwd, dll_path=dll_path
                    )
                    if args.enable_transformers_tool_test:
                        import google.protobuf
                        import numpy

                        numpy_init_version = numpy.__version__
                        pb_init_version = google.protobuf.__version__
                        run_subprocess(
                            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=SCRIPT_DIR
                        )
                        run_subprocess([sys.executable, "-m", "pytest", "transformers"], cwd=cwd)
                        # Restore initial numpy/protobuf version in case other tests use it
                        run_subprocess([sys.executable, "-m", "pip", "install", "numpy==" + numpy_init_version])
                        run_subprocess([sys.executable, "-m", "pip", "install", "protobuf==" + pb_init_version])

                if not args.disable_ml_ops:
                    run_subprocess(
                        [sys.executable, "onnxruntime_test_python_backend_mlops.py"], cwd=cwd, dll_path=dll_path
                    )

                run_subprocess(
                    [
                        sys.executable,
                        os.path.join(source_dir, "onnxruntime", "test", "onnx", "gen_test_models.py"),
                        "--output_dir",
                        "test_models",
                    ],
                    cwd=cwd,
                )

                if not args.skip_onnx_tests:
                    run_subprocess([os.path.join(cwd, "onnx_test_runner"), "test_models"], cwd=cwd)
                    if config != "Debug":
                        run_subprocess([sys.executable, "onnx_backend_test_series.py"], cwd=cwd, dll_path=dll_path)

            if not args.skip_keras_test:
                try:
                    import keras  # noqa
                    import onnxmltools  # noqa

                    onnxml_test = True
                except ImportError:
                    log.warning("onnxmltools and keras are not installed. " "The keras tests will be skipped.")
                    onnxml_test = False
                if onnxml_test:
                    run_subprocess([sys.executable, "onnxruntime_test_python_keras.py"], cwd=cwd, dll_path=dll_path)


def tvm_run_python_tests(build_dir, configs):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows():
            cwd = os.path.join(cwd, config)
        python_path = os.path.join(build_dir, config, "_deps", "tvm-src", "python")
        run_subprocess(
            [sys.executable, "onnxruntime_test_python_tvm.py"], cwd=cwd, python_path=os.path.abspath(python_path)
        )


def run_nodejs_tests(nodejs_binding_dir):
    args = ["npm", "test", "--", "--timeout=90000"]
    if is_windows():
        args = ["cmd", "/c"] + args
    run_subprocess(args, cwd=nodejs_binding_dir)


def build_python_wheel(
    source_dir,
    build_dir,
    configs,
    use_cuda,
    cuda_version,
    use_rocm,
    rocm_version,
    use_dnnl,
    use_tensorrt,
    use_openvino,
    use_tvm,
    use_vitisai,
    use_acl,
    use_armnn,
    use_dml,
    use_cann,
    wheel_name_suffix,
    enable_training,
    nightly_build=False,
    default_training_package_device=False,
    use_ninja=False,
    build_eager_mode=False,
    enable_training_on_device=False,
    enable_rocm_profiling=False,
):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows() and not use_ninja:
            cwd = os.path.join(cwd, config)

        args = [sys.executable, os.path.join(source_dir, "setup.py"), "bdist_wheel"]

        # Any combination of the following arguments can be applied
        if nightly_build:
            args.append("--nightly_build")
        if default_training_package_device:
            args.append("--default_training_package_device")
        if wheel_name_suffix:
            args.append("--wheel_name_suffix={}".format(wheel_name_suffix))
        if enable_training:
            args.append("--enable_training")
        if enable_training_on_device:
            args.append("--enable_training_on_device")
        if build_eager_mode:
            args.append("--disable_auditwheel_repair")
        if enable_rocm_profiling:
            args.append("--enable_rocm_profiling")

        # The following arguments are mutually exclusive
        if use_cuda:
            # The following line assumes no other EP is enabled
            args.append("--wheel_name_suffix=gpu")
            if cuda_version:
                args.append("--cuda_version={}".format(cuda_version))
        elif use_rocm:
            args.append("--use_rocm")
            if rocm_version:
                args.append("--rocm_version={}".format(rocm_version))
        elif use_openvino:
            args.append("--use_openvino")
        elif use_dnnl:
            args.append("--use_dnnl")
        elif use_tvm:
            args.append("--use_tvm")
        elif use_vitisai:
            args.append("--use_vitisai")
        elif use_acl:
            args.append("--use_acl")
        elif use_armnn:
            args.append("--use_armnn")
        elif use_dml:
            args.append("--wheel_name_suffix=directml")
        elif use_cann:
            args.append("--use_cann")

        run_subprocess(args, cwd=cwd)


def derive_linux_build_property():
    if is_windows():
        return '/p:IsLinuxBuild="false"'
    else:
        return '/p:IsLinuxBuild="true"'


def build_nuget_package(
    source_dir,
    build_dir,
    configs,
    use_cuda,
    use_openvino,
    use_tensorrt,
    use_dnnl,
    use_tvm,
    use_winml,
    use_snpe,
    enable_training_on_device,
):
    if not (is_windows() or is_linux()):
        raise BuildError(
            "Currently csharp builds and nuget package creation is only supportted " "on Windows and Linux platforms."
        )

    csharp_build_dir = os.path.join(source_dir, "csharp")
    is_linux_build = derive_linux_build_property()

    # in most cases we don't want/need to include the Xamarin mobile targets, as doing so means the Xamarin
    # mobile workloads must be installed on the machine.
    # they are only included in the Microsoft.ML.OnnxRuntime nuget package
    sln = "OnnxRuntime.DesktopOnly.CSharp.sln"

    # derive package name and execution provider based on the build args
    target_name = "/t:CreatePackage"
    execution_provider = '/p:ExecutionProvider="None"'
    package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime"'
    if enable_training_on_device:
        if use_cuda:
            package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime.Training.Gpu"'
        else:
            package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime.Training"'
    elif use_winml:
        package_name = '/p:OrtPackageId="Microsoft.AI.MachineLearning"'
        target_name = "/t:CreateWindowsAIPackage"
    elif use_openvino:
        execution_provider = '/p:ExecutionProvider="openvino"'
        package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime.OpenVino"'
    elif use_tensorrt:
        execution_provider = '/p:ExecutionProvider="tensorrt"'
        package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime.TensorRT"'
    elif use_dnnl:
        execution_provider = '/p:ExecutionProvider="dnnl"'
        package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime.DNNL"'
    elif use_cuda:
        package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime.Gpu"'
    elif use_tvm:
        execution_provider = '/p:ExecutionProvider="tvm"'
        package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime.Tvm"'
    elif use_snpe:
        execution_provider = '/p:ExecutionProvider="snpe"'
        package_name = '/p:OrtPackageId="Microsoft.ML.OnnxRuntime.Snpe"'
    else:
        # use the solution file that includes Xamarin mobile targets
        sln = "OnnxRuntime.CSharp.sln"

    # set build directory based on build_dir arg
    native_dir = os.path.normpath(os.path.join(source_dir, build_dir))
    ort_build_dir = '/p:OnnxRuntimeBuildDirectory="' + native_dir + '"'

    # dotnet restore
    cmd_args = ["dotnet", "restore", sln, "--configfile", "Nuget.CSharp.config"]
    run_subprocess(cmd_args, cwd=csharp_build_dir)

    # build csharp bindings and create nuget package for each config
    for config in configs:
        if is_linux():
            native_build_dir = os.path.join(native_dir, config)
            cmd_args = ["make", "install", "DESTDIR=.//nuget-staging"]
            run_subprocess(cmd_args, cwd=native_build_dir)

        configuration = '/p:Configuration="' + config + '"'

        if not use_winml:
            cmd_args = ["dotnet", "msbuild", sln, configuration, package_name, is_linux_build, ort_build_dir]
            run_subprocess(cmd_args, cwd=csharp_build_dir)
        else:
            winml_interop_dir = os.path.join(source_dir, "csharp", "src", "Microsoft.AI.MachineLearning.Interop")
            winml_interop_project = os.path.join(winml_interop_dir, "Microsoft.AI.MachineLearning.Interop.csproj")
            winml_interop_project = os.path.normpath(winml_interop_project)
            cmd_args = [
                "dotnet",
                "msbuild",
                winml_interop_project,
                configuration,
                '/p:Platform="Any CPU"',
                ort_build_dir,
                "-restore",
            ]
            run_subprocess(cmd_args, cwd=csharp_build_dir)

        if is_windows():
            if not use_winml:
                # user needs to make sure nuget is installed and added to the path variable
                nuget_exe = "nuget.exe"
            else:
                # this path is setup by cmake/nuget_helpers.cmake for MSVC on Windows
                nuget_exe = os.path.normpath(os.path.join(native_dir, config, "nuget_exe", "src", "nuget.exe"))
        else:
            # user needs to make sure nuget is installed and can be found
            nuget_exe = "nuget"

        nuget_exe_arg = '/p:NugetExe="' + nuget_exe + '"'

        cmd_args = [
            "dotnet",
            "msbuild",
            "OnnxRuntime.CSharp.proj",
            target_name,
            package_name,
            configuration,
            execution_provider,
            is_linux_build,
            ort_build_dir,
            nuget_exe_arg,
        ]
        run_subprocess(cmd_args, cwd=csharp_build_dir)


def run_csharp_tests(source_dir, build_dir, use_cuda, use_openvino, use_tensorrt, use_dnnl, enable_training_on_device):
    # Currently only running tests on windows.
    if not is_windows():
        return
    csharp_source_dir = os.path.join(source_dir, "csharp")

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
    if enable_training_on_device:
        macros += "__TRAINING_ENABLED_NATIVE_BUILD__;"

    define_constants = ""
    if macros != "":
        define_constants = '/p:DefineConstants="' + macros + '"'

    # set build directory based on build_dir arg
    native_build_dir = os.path.normpath(os.path.join(source_dir, build_dir))
    ort_build_dir = '/p:OnnxRuntimeBuildDirectory="' + native_build_dir + '"'

    # Skip pretrained models test. Only run unit tests as part of the build
    # add "--verbosity", "detailed" to this command if required
    cmd_args = [
        "dotnet",
        "test",
        "test\\Microsoft.ML.OnnxRuntime.Tests.NetCoreApp\\Microsoft.ML.OnnxRuntime.Tests.NetCoreApp.csproj",
        "--filter",
        "FullyQualifiedName!=Microsoft.ML.OnnxRuntime.Tests.InferenceTest.TestPreTrainedModels",
        define_constants,
        ort_build_dir,
    ]
    run_subprocess(cmd_args, cwd=csharp_source_dir)


def is_cross_compiling_on_apple(args):
    if not is_macOS():
        return False
    if args.ios:
        return True
    if args.osx_arch != platform.machine():
        return True
    return False


def build_protoc_for_host(cmake_path, source_dir, build_dir, args):
    if (args.arm or args.arm64 or args.arm64ec) and not (is_windows() or is_cross_compiling_on_apple(args)):
        raise BuildError(
            "Currently only support building protoc for Windows host while "
            "cross-compiling for ARM/ARM64/Store and linux cross-compiling iOS"
        )

    log.info("Building protoc for host to be used in cross-compiled build process")
    protoc_build_dir = os.path.join(os.getcwd(), build_dir, "host_protoc")
    os.makedirs(protoc_build_dir, exist_ok=True)
    # Generate step
    cmd_args = [
        cmake_path,
        os.path.join(source_dir, "cmake", "external", "protobuf", "cmake"),
        "-Dprotobuf_BUILD_TESTS=OFF",
        "-Dprotobuf_WITH_ZLIB_DEFAULT=OFF",
        "-Dprotobuf_BUILD_SHARED_LIBS=OFF",
    ]

    is_ninja = args.cmake_generator == "Ninja"
    if args.cmake_generator is not None and not (is_macOS() and args.use_xcode):
        cmd_args += ["-G", args.cmake_generator]
    if is_windows():
        if not is_ninja:
            cmd_args += ["-T", "host=x64"]
    elif is_macOS():
        if args.use_xcode:
            cmd_args += ["-G", "Xcode"]
            # CMake < 3.18 has a bug setting system arch to arm64 (if not specified) for Xcode 12,
            # protoc for host should be built using host architecture
            # Explicitly specify the CMAKE_OSX_ARCHITECTURES for x86_64 Mac.
            cmd_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format("arm64" if platform.machine() == "arm64" else "x86_64")]

    run_subprocess(cmd_args, cwd=protoc_build_dir)
    # Build step
    cmd_args = [cmake_path, "--build", protoc_build_dir, "--config", "Release", "--target", "protoc"]
    run_subprocess(cmd_args)

    # Absolute protoc path is needed for cmake
    config_dir = ""
    suffix = ""

    if (is_windows() and not is_ninja) or (is_macOS() and args.use_xcode):
        config_dir = "Release"

    if is_windows():
        suffix = ".exe"

    expected_protoc_path = os.path.join(protoc_build_dir, config_dir, "protoc" + suffix)

    if not os.path.exists(expected_protoc_path):
        raise BuildError("Couldn't find {}. Host build of protoc failed.".format(expected_protoc_path))

    return expected_protoc_path


def generate_documentation(source_dir, build_dir, configs, validate):
    # Randomly choose one build config
    config = next(iter(configs))
    cwd = get_config_build_dir(build_dir, config)
    if is_windows():
        cwd = os.path.join(cwd, config)

    contrib_op_doc_path = os.path.join(source_dir, "docs", "ContribOperators.md")
    opkernel_doc_path = os.path.join(source_dir, "docs", "OperatorKernels.md")
    shutil.copy(os.path.join(source_dir, "tools", "python", "gen_contrib_doc.py"), cwd)
    shutil.copy(os.path.join(source_dir, "tools", "python", "gen_opkernel_doc.py"), cwd)
    # limit to just com.microsoft (excludes purely internal stuff like com.microsoft.nchwc).
    run_subprocess(
        [sys.executable, "gen_contrib_doc.py", "--output_path", contrib_op_doc_path, "--domains", "com.microsoft"],
        cwd=cwd,
    )
    # we currently limit the documentation created by a build to a subset of EP's.
    # Run get_opkernel_doc.py directly if you need/want documentation from other EPs that are enabled in the build.
    run_subprocess(
        [
            sys.executable,
            "gen_opkernel_doc.py",
            "--output_path",
            opkernel_doc_path,
            "--providers",
            "CPU",
            "CUDA",
            "DML",
        ],
        cwd=cwd,
    )

    if validate:
        try:
            have_diff = False

            def diff_file(path, regenerate_qualifiers=""):
                diff = subprocess.check_output(["git", "diff", path], cwd=source_dir)
                if diff:
                    nonlocal have_diff
                    have_diff = True
                    log.warning(
                        "The updated document {} is different from the checked in version. "
                        "Please regenerate the file{}, or copy the updated version from the "
                        "CI build's published artifacts if applicable.".format(path, regenerate_qualifiers)
                    )
                    log.debug("diff:\n" + str(diff))

            diff_file(opkernel_doc_path, " with CPU, CUDA and DML execution providers enabled")
            diff_file(contrib_op_doc_path)

            if have_diff:
                # Output for the CI to publish the updated md files as an artifact
                print("##vso[task.setvariable variable=DocUpdateNeeded]true")
                raise BuildError("Generated documents have diffs. Check build output for details.")

        except subprocess.CalledProcessError:
            raise BuildError("git diff returned non-zero error code")


def main():
    log.debug("Command line arguments:\n  {}".format(" ".join(shlex.quote(arg) for arg in sys.argv[1:])))

    args = parse_arguments()
    cmake_extra_defines = normalize_arg_list(args.cmake_extra_defines)
    cross_compiling = args.arm or args.arm64 or args.arm64ec or args.android

    # If there was no explicit argument saying what to do, default
    # to update, build and test (for native builds).
    if not (args.update or args.clean or args.build or args.test or args.gen_doc):
        log.debug("Defaulting to running update, build [and test for native builds].")
        args.update = True
        args.build = True
        if cross_compiling:
            args.test = args.android_abi == "x86_64" or args.android_abi == "arm64-v8a"
        else:
            args.test = True

    if args.skip_tests:
        args.test = False

    if args.use_tensorrt:
        args.use_cuda = True

    if args.use_migraphx:
        args.use_rocm = True

    if args.build_wheel or args.gen_doc or args.use_tvm:
        args.enable_pybind = True

    if args.build_csharp or args.build_nuget or args.build_java or args.build_nodejs:
        args.build_shared_lib = True

    if args.build_nuget and cross_compiling:
        raise BuildError("Currently nuget package creation is not supported while cross-compiling")

    if args.enable_pybind and args.disable_rtti:
        raise BuildError("Python bindings use typeid so you can't disable RTTI")

    if args.enable_pybind and args.disable_exceptions:
        raise BuildError("Python bindings require exceptions to be enabled.")

    if args.nnapi_min_api:
        if not args.use_nnapi:
            raise BuildError("Using --nnapi_min_api requires --use_nnapi")
        if args.nnapi_min_api < 27:
            raise BuildError("--nnapi_min_api should be 27+")

    if args.build_wasm_static_lib:
        args.build_wasm = True

    if args.build_wasm:
        if not args.disable_wasm_exception_catching and args.disable_exceptions:
            # When '--disable_exceptions' is set, we set '--disable_wasm_exception_catching' as well
            args.disable_wasm_exception_catching = True
        if args.test and args.disable_wasm_exception_catching and not args.minimal_build:
            raise BuildError("WebAssembly tests need exception catching enabled to run if it's not minimal build")
        if args.test and args.enable_wasm_debug_info:
            # With flag --enable_wasm_debug_info, onnxruntime_test_all.wasm will be very huge (>1GB). This will fail
            # Node.js when trying to load the .wasm file.
            # To debug ONNX Runtime WebAssembly, use ONNX Runtime Web to debug ort-wasm.wasm in browsers.
            raise BuildError("WebAssembly tests cannot be enabled with flag --enable_wasm_debug_info")

        if args.wasm_malloc is not None:
            # mark --wasm_malloc as deprecated
            log.warning(
                "Flag '--wasm_malloc=<Value>' is deprecated. " "Please use '--emscripten_settings MALLOC=<Value>'."
            )

    if args.code_coverage and not args.android:
        raise BuildError("Using --code_coverage requires --android")

    if args.gen_api_doc and len(args.config) != 1:
        raise BuildError("Using --get-api-doc requires a single build config")

    # Disabling unit tests for VAD-F as FPGA only supports
    # models with NCHW layout
    if args.use_openvino == "VAD-F_FP32":
        args.test = False

    # Disabling unit tests for GPU and MYRIAD on nuget creation
    if args.use_openvino != "CPU_FP32" and args.build_nuget:
        args.test = False

    # GDK builds don't support testing
    if args.use_gdk:
        args.test = False

    configs = set(args.config)

    # setup paths and directories
    # cmake_path and ctest_path can be None. For example, if a person only wants to run the tests, he/she doesn't need
    # to have cmake/ctest.
    cmake_path = resolve_executable_path(args.cmake_path)
    ctest_path = None if args.use_vstest else resolve_executable_path(args.ctest_path)
    build_dir = args.build_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))

    # if using cuda, setup cuda paths and env vars
    cuda_home, cudnn_home = setup_cuda_vars(args)

    mpi_home = args.mpi_home
    nccl_home = args.nccl_home

    snpe_root = args.snpe_root

    acl_home = args.acl_home
    acl_libs = args.acl_libs

    armnn_home = args.armnn_home
    armnn_libs = args.armnn_libs

    # if using tensorrt, setup tensorrt paths
    tensorrt_home = setup_tensorrt_vars(args)

    # if using migraphx, setup migraphx paths
    migraphx_home = setup_migraphx_vars(args)

    # if using rocm, setup rocm paths
    rocm_home = setup_rocm_build(args)

    # if using cann, setup cann paths
    cann_home = setup_cann_vars(args)

    if args.update or args.build:
        for config in configs:
            os.makedirs(get_config_build_dir(build_dir, config), exist_ok=True)

    log.info("Build started")

    if args.update:
        if is_reduced_ops_build(args):
            from reduce_op_kernels import reduce_ops

            is_extended_minimal_build_or_higher = args.minimal_build is None or "extended" in args.minimal_build
            for config in configs:
                reduce_ops(
                    config_path=args.include_ops_by_config,
                    build_dir=get_config_build_dir(build_dir, config),
                    enable_type_reduction=args.enable_reduced_operator_type_support,
                    use_cuda=args.use_cuda,
                    is_extended_minimal_build_or_higher=is_extended_minimal_build_or_higher,
                )

        cmake_extra_args = []
        path_to_protoc_exe = args.path_to_protoc_exe
        if not args.skip_submodule_sync:
            update_submodules(source_dir)
        if is_windows():
            cpu_arch = platform.architecture()[0]
            if args.build_wasm:
                cmake_extra_args = ["-G", "Ninja"]
            elif args.cmake_generator == "Ninja":
                if cpu_arch == "32bit" or args.arm or args.arm64 or args.arm64ec:
                    raise BuildError(
                        "To cross-compile with Ninja, load the toolset "
                        "environment for the target processor (e.g. Cross "
                        "Tools Command Prompt for VS)"
                    )
                cmake_extra_args = ["-G", args.cmake_generator]
            elif args.arm or args.arm64 or args.arm64ec:
                # Cross-compiling for ARM(64) architecture
                # First build protoc for host to use during cross-compilation
                if path_to_protoc_exe is None:
                    path_to_protoc_exe = build_protoc_for_host(cmake_path, source_dir, build_dir, args)
                if args.arm:
                    cmake_extra_args = ["-A", "ARM"]
                elif args.arm64:
                    cmake_extra_args = ["-A", "ARM64"]
                elif args.arm64ec:
                    cmake_extra_args = ["-A", "ARM64EC"]
                cmake_extra_args += ["-G", args.cmake_generator]
                # Cannot test on host build machine for cross-compiled
                # builds (Override any user-defined behaviour for test if any)
                if args.test:
                    log.warning(
                        "Cannot test on host build machine for cross-compiled "
                        "ARM(64) builds. Will skip test running after build."
                    )
                    args.test = False
            elif cpu_arch == "32bit" or args.x86:
                cmake_extra_args = ["-A", "Win32", "-T", "host=x64", "-G", args.cmake_generator]
            else:
                if args.msvc_toolset:
                    toolset = "host=x64,version=" + args.msvc_toolset
                else:
                    toolset = "host=x64"
                if args.cuda_version:
                    toolset += ",cuda=" + args.cuda_version
                cmake_extra_args = ["-A", "x64", "-T", toolset, "-G", args.cmake_generator]
            if args.enable_wcos:
                cmake_extra_defines.append("CMAKE_USER_MAKE_RULES_OVERRIDE=wcos_rules_override.cmake")
        elif args.cmake_generator is not None and not (is_macOS() and args.use_xcode):
            cmake_extra_args += ["-G", args.cmake_generator]
        elif is_macOS():
            if args.use_xcode:
                cmake_extra_args += ["-G", "Xcode"]
            if not args.ios and not args.android and args.osx_arch == "arm64" and platform.machine() == "x86_64":
                if args.test:
                    log.warning("Cannot test ARM64 build on X86_64. Will skip test running after build.")
                    args.test = False

        if args.build_wasm:
            emsdk_version = args.emsdk_version
            emsdk_dir = os.path.join(source_dir, "cmake", "external", "emsdk")
            emsdk_file = os.path.join(emsdk_dir, "emsdk.bat") if is_windows() else os.path.join(emsdk_dir, "emsdk")

            log.info("Installing emsdk...")
            run_subprocess([emsdk_file, "install", emsdk_version], cwd=emsdk_dir)
            log.info("Activating emsdk...")
            run_subprocess([emsdk_file, "activate", emsdk_version], cwd=emsdk_dir)

        if (
            args.android or args.ios or args.build_wasm or is_cross_compiling_on_apple(args)
        ) and args.path_to_protoc_exe is None:
            # Cross-compiling for Android, iOS, and WebAssembly
            path_to_protoc_exe = build_protoc_for_host(cmake_path, source_dir, build_dir, args)

        if is_ubuntu_1604():
            if args.arm or args.arm64:
                raise BuildError(
                    "Only Windows ARM(64) cross-compiled builds supported " "currently through this script"
                )
            if not is_docker() and not args.use_acl and not args.use_armnn:
                install_python_deps()

        if args.enable_pybind and is_windows():
            install_python_deps(args.numpy_version)

        if args.use_cuda and args.cuda_version is None:
            if is_windows():
                # cuda_version is used while generating version_info.py on Windows.
                raise BuildError("cuda_version must be specified on Windows.")
            else:
                args.cuda_version = ""
        if args.use_rocm and args.rocm_version is None:
            args.rocm_version = ""

        if args.build_eager_mode:
            eager_root_dir = os.path.join(source_dir, "orttraining", "orttraining", "eager")
            if args.eager_customop_module and not args.eager_customop_header:
                raise Exception("eager_customop_header must be provided when eager_customop_module is")
            elif args.eager_customop_header and not args.eager_customop_module:
                raise Exception("eager_customop_module must be provided when eager_customop_header is")

            def gen_ops(gen_cpp_name: str, header_file: str, ops_module: str, custom_ops: bool):
                gen_cpp_scratch_name = gen_cpp_name + ".working"
                print(
                    f"Generating ORT ATen overrides (output_file: {gen_cpp_name}, header_file: {header_file},"
                    f"ops_module: {ops_module}), custom_ops: {custom_ops}"
                )

                cmd = [
                    sys.executable,
                    os.path.join(os.path.join(eager_root_dir, "opgen", "opgen.py")),
                    "--output_file",
                    gen_cpp_scratch_name,
                    "--ops_module",
                    ops_module,
                    "--header_file",
                    header_file,
                ]

                if custom_ops:
                    cmd += ["--custom_ops"]

                subprocess.check_call(cmd)

                import filecmp

                if not os.path.isfile(gen_cpp_name) or not filecmp.cmp(
                    gen_cpp_name, gen_cpp_scratch_name, shallow=False
                ):
                    os.rename(gen_cpp_scratch_name, gen_cpp_name)
                else:
                    os.remove(gen_cpp_scratch_name)

            def gen_ort_ops():
                # generate native aten ops
                import torch

                regdecs_path = os.path.join(os.path.dirname(torch.__file__), "include/ATen/RegistrationDeclarations.h")

                ops_module = os.path.join(eager_root_dir, "opgen/opgen/atenops.py")
                gen_ops(os.path.join(eager_root_dir, "ort_aten.g.cpp"), regdecs_path, ops_module, False)

                # generate custom ops
                if not args.eager_customop_header:
                    args.eager_customop_header = os.path.realpath(
                        os.path.join(eager_root_dir, "opgen", "CustomOpDeclarations.h")
                    )

                if not args.eager_customop_module:
                    args.eager_customop_module = os.path.join(eager_root_dir, "opgen/opgen/custom_ops.py")

                gen_ops(
                    os.path.join(eager_root_dir, "ort_customops.g.cpp"),
                    args.eager_customop_header,
                    args.eager_customop_module,
                    True,
                )

            gen_ort_ops()
        if args.enable_external_custom_op_schemas and not is_linux():
            raise BuildError("Registering external custom op schemas is only supported on Linux.")

        generate_build_tree(
            cmake_path,
            source_dir,
            build_dir,
            cuda_home,
            cudnn_home,
            rocm_home,
            mpi_home,
            nccl_home,
            tensorrt_home,
            migraphx_home,
            acl_home,
            acl_libs,
            armnn_home,
            armnn_libs,
            snpe_root,
            cann_home,
            path_to_protoc_exe,
            configs,
            cmake_extra_defines,
            args,
            cmake_extra_args,
        )

    if args.clean:
        clean_targets(cmake_path, build_dir, configs)

    # if using DML, perform initial nuget package restore
    setup_dml_build(args, cmake_path, build_dir, configs)

    if args.build:
        if args.parallel < 0:
            raise BuildError("Invalid parallel job count: {}".format(args.parallel))
        num_parallel_jobs = os.cpu_count() if args.parallel == 0 else args.parallel
        build_targets(args, cmake_path, build_dir, configs, num_parallel_jobs, args.target)

    if args.test:
        if args.enable_onnx_tests:
            source_onnx_model_dir = "C:\\local\\models" if is_windows() else "/data/models"
            setup_test_data(source_onnx_model_dir, "models", build_dir, configs)

        run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs)

        # TODO(agladyshev):
        # to support Windows, we need to update .github/workflows/windows.yml
        # and add to the PATH variable the following value: C:Program Files\LLVM\bin
        if args.enable_pybind and args.use_tvm and not is_windows():
            tvm_run_python_tests(build_dir, configs)

        # run node.js binding tests
        if args.build_nodejs and not args.skip_nodejs_tests:
            nodejs_binding_dir = os.path.normpath(os.path.join(source_dir, "js", "node"))
            run_nodejs_tests(nodejs_binding_dir)

    # Build packages after running the tests.
    # NOTE: if you have a test that rely on a file which only get copied/generated during packaging step, it could
    # fail unexpectedly. Similar, if your packaging step forgot to copy a file into the package, we don't know it
    # either.
    if args.build:
        if args.build_wheel:
            nightly_build = bool(os.getenv("NIGHTLY_BUILD") == "1")
            default_training_package_device = bool(os.getenv("DEFAULT_TRAINING_PACKAGE_DEVICE") == "1")
            build_python_wheel(
                source_dir,
                build_dir,
                configs,
                args.use_cuda,
                args.cuda_version,
                args.use_rocm,
                args.rocm_version,
                args.use_dnnl,
                args.use_tensorrt,
                args.use_openvino,
                args.use_tvm,
                args.use_vitisai,
                args.use_acl,
                args.use_armnn,
                args.use_dml,
                args.use_cann,
                args.wheel_name_suffix,
                args.enable_training,
                nightly_build=nightly_build,
                default_training_package_device=default_training_package_device,
                use_ninja=(args.cmake_generator == "Ninja"),
                build_eager_mode=args.build_eager_mode,
                enable_training_on_device=args.enable_training_on_device,
                enable_rocm_profiling=args.enable_rocm_profiling,
            )
        if args.build_nuget:
            build_nuget_package(
                source_dir,
                build_dir,
                configs,
                args.use_cuda,
                args.use_openvino,
                args.use_tensorrt,
                args.use_dnnl,
                args.use_tvm,
                args.use_winml,
                args.use_snpe,
                args.enable_training_on_device,
            )

    if args.test and args.build_nuget:
        run_csharp_tests(
            source_dir,
            build_dir,
            args.use_cuda,
            args.use_openvino,
            args.use_tensorrt,
            args.use_dnnl,
            args.enable_training_on_device,
        )

    if args.gen_doc:
        # special case CI where we create the build config separately to building
        if args.update and not args.build:
            pass
        else:
            # assumes build has occurred for easier use in CI where we don't always build via build.py and need to run
            # documentation generation as a separate task post-build
            generate_documentation(source_dir, build_dir, configs, args.gen_doc == "validate")

    if args.gen_api_doc and (args.build or args.test):
        print("Generating Python doc for ORTModule...")
        docbuild_dir = os.path.join(source_dir, "tools", "doc")
        run_subprocess(
            ["bash", "builddoc.sh", os.path.dirname(sys.executable), source_dir, build_dir, args.config[0]],
            cwd=docbuild_dir,
        )

    log.info("Build complete")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseError as e:
        log.error(str(e))
        sys.exit(1)
