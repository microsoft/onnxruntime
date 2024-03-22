# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.26)

cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

cmake_policy(SET CMP0092 NEW)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0117 NEW)
# Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 OLD)

# Enable Hot Reload for MSVC compilers if supported.
cmake_policy(SET CMP0141 NEW)

# Project
project(onnxruntime C CXX ASM)

# Disable fast-math for Intel oneAPI compiler
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC-like")
    # Using icx-cl compiler driver with MSVC-like arguments
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:precise")
  else()
    # Using icpx compiler driver
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-fast-math")
  endif()
endif()

# Needed for Java
set(CMAKE_C_STANDARD 99)

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(CMakeDependentOption)
include(FetchContent)
include(CheckFunctionExists)

# TODO: update this once all system adapt c++20
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set(CMAKE_CXX_STANDARD 20)
else()
set(CMAKE_CXX_STANDARD 17)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# NOTE: POSITION INDEPENDENT CODE hurts performance, and it only make sense on POSIX systems
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable CTest
enable_testing()
include(Dart)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 8)
  message(FATAL_ERROR  "GCC version must be greater than or equal to 8")
endif()

# Options
option(onnxruntime_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)
option(onnxruntime_GENERATE_TEST_REPORTS "Enable test report generation" OFF)
option(onnxruntime_ENABLE_STATIC_ANALYSIS "Enable static analysis" OFF)
option(onnxruntime_USE_CUSTOM_STATIC_ANALYSIS_RULES "Use a custom SDL Rule. It is mainly for our CI build" OFF)
option(onnxruntime_REDIRECT_STATIC_ANALYSIS_OUTPUTS_TO_FILE "Use a custom SDL Rule. It is mainly for our CI build" OFF)
option(onnxruntime_ENABLE_PYTHON "Enable python buildings" OFF)
# Enable it may cause LNK1169 error
option(onnxruntime_ENABLE_MEMLEAK_CHECKER "Experimental: Enable memory leak checker in Windows debug build" OFF)
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
# Enable ONNX Runtime CUDA EP's internal unit tests that directly access the EP's internal functions instead of through
# OpKernels. When the option is ON, we will have two copies of GTest library in the same process. It is not a typical
# use. If you hit any problem with that, please do not report it to GTest. Turn OFF the following build option instead.
cmake_dependent_option(onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS "Build with CUDA unit tests" OFF "onnxruntime_USE_CUDA;onnxruntime_BUILD_UNIT_TESTS;LINUX" OFF)

option(onnxruntime_USE_CUDA_NHWC_OPS "Build CUDA with NHWC op support" OFF)
option(onnxruntime_CUDA_MINIMAL "Build CUDA without any operations apart from memcpy ops. Usefuel for a very minial TRT build" OFF)
option(onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO "When building with CUDA support, generate device code line number information." OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_COREML "Build with CoreML support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_QNN "Build with QNN support" OFF)
option(onnxruntime_USE_SNPE "Build with SNPE support" OFF)
option(onnxruntime_USE_RKNPU "Build with RKNPU support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NEURAL_SPEED "Build with Neural Speed support" ON)
option(onnxruntime_USE_JSEP "Build with JavaScript implemented kernels support" OFF)
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" ON)
option(onnxruntime_BUILD_CSHARP "Build C# library" OFF)
option(onnxruntime_BUILD_OBJC "Build Objective-C library" OFF)
option(onnxruntime_USE_PREINSTALLED_EIGEN "Use pre-installed EIGEN. Need to provide eigen_SOURCE_PATH if turn this on." OFF)
option(onnxruntime_BUILD_BENCHMARKS "Build ONNXRuntime micro-benchmarks" OFF)
option(onnxruntime_USE_LLVM "Build TVM with LLVM" OFF)

cmake_dependent_option(onnxruntime_USE_FLASH_ATTENTION "Build flash attention kernel for scaled dot product attention" ON "NOT WIN32; onnxruntime_USE_CUDA" OFF)
option(onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION "Build memory efficient attention kernel for scaled dot product attention" ON)

option(onnxruntime_BUILD_FOR_NATIVE_MACHINE "Enable this option for turning on optimization specific to this machine" OFF)
option(onnxruntime_USE_AVX "Use AVX instructions" OFF)
option(onnxruntime_USE_AVX2 "Use AVX2 instructions" OFF)
option(onnxruntime_USE_AVX512 "Use AVX512 instructions" OFF)

option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" OFF)
option(onnxruntime_BUILD_APPLE_FRAMEWORK "Build a macOS/iOS framework" OFF)
option(onnxruntime_ENABLE_MICROSOFT_INTERNAL "Use this option to enable/disable microsoft internal only code" OFF)

option(onnxruntime_USE_VITISAI "Build with Vitis-AI" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(onnxruntime_USE_TENSORRT_BUILTIN_PARSER "Use TensorRT builtin parser" OFF)
option(onnxruntime_ENABLE_LTO "Enable link time optimization" OFF)
option(onnxruntime_CROSS_COMPILING "Cross compiling onnx runtime" OFF)
option(onnxruntime_GCOV_COVERAGE "Compile with options necessary to run code coverage" OFF)
option(onnxruntime_DONT_VECTORIZE "Do not vectorize operations in Eigen" OFF)

option(onnxruntime_USE_FULL_PROTOBUF "Link to libprotobuf instead of libprotobuf-lite when this option is ON" OFF)
option(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS "Enable operator implemented in language other than cpp" OFF)
option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS "Dump debug information about node inputs and outputs when executing the model." OFF)
cmake_dependent_option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB "Build dump debug information about node inputs and outputs with support for sql database." OFF "onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS" OFF)
option(onnxruntime_USE_DML "Build with DirectML support" OFF)
option(onnxruntime_USE_MIGRAPHX "Build with AMDMIGraphX support" OFF)
option(onnxruntime_USE_WINML "Build with WinML support" OFF)
option(onnxruntime_USE_ACL "Build with ACL support" OFF)
option(onnxruntime_USE_ACL_1902 "Build with ACL version 1902 support" OFF)
option(onnxruntime_USE_ACL_1905 "Build with ACL version 1905 support" OFF)
option(onnxruntime_USE_ACL_1908 "Build with ACL version 1908 support" OFF)
option(onnxruntime_USE_ACL_2002 "Build with ACL version 2002 support" OFF)
option(onnxruntime_USE_ACL_2308 "Build with ACL version 2308 support" OFF)
option(onnxruntime_USE_ARMNN "Build with ArmNN support" OFF)
option(onnxruntime_ARMNN_RELU_USE_CPU "Use the CPU implementation for the Relu operator for the ArmNN EP" ON)
option(onnxruntime_ARMNN_BN_USE_CPU "Use the CPU implementation for the Batch Normalization operator for the ArmNN EP" ON)
option(onnxruntime_ENABLE_INSTRUMENT "Enable Instrument with Event Tracing for Windows (ETW)" OFF)
option(onnxruntime_USE_TELEMETRY "Build with Telemetry" OFF)
cmake_dependent_option(onnxruntime_USE_MIMALLOC "Override new/delete and arena allocator with mimalloc" OFF "WIN32;NOT onnxruntime_USE_CUDA;NOT onnxruntime_USE_OPENVINO" OFF)
option(onnxruntime_USE_CANN "Build with CANN support" OFF)
option(onnxruntime_USE_ROCM "Build with AMD GPU support" OFF)
option(onnxruntime_USE_TVM "Build with TVM support" OFF)
option(onnxruntime_TVM_CUDA_RUNTIME "Build TVM with CUDA support" OFF)
option(onnxruntime_TVM_USE_LLVM "Build TVM with LLVM. Set customized path to llvm-config.exe here if need" OFF)
option(onnxruntime_TVM_USE_HASH "Build ipp-crypto library for support hash algorithm. It is defined for TVM only")
option(onnxruntime_USE_XNNPACK "Build with XNNPACK support. Provides an alternative math library on ARM, WebAssembly and x86." OFF)
option(onnxruntime_USE_WEBNN "Build with WebNN support. Enable hardware acceleration in web browsers." OFF)

# Options related to reducing the binary size produced by the build
# XNNPACK EP requires the internal NHWC contrib ops to be available, so this option must be OFF when onnxruntime_USE_XNNPACK is ON
cmake_dependent_option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" OFF "NOT onnxruntime_USE_XNNPACK" OFF)
option(onnxruntime_DISABLE_ML_OPS "Disable traditional ML ops" OFF)
option(onnxruntime_DISABLE_SPARSE_TENSORS "Disable sparse tensors data types" OFF)
option(onnxruntime_DISABLE_OPTIONAL_TYPE "Disable optional type" OFF)
option(onnxruntime_DISABLE_FLOAT8_TYPES "Disable float 8 types" OFF)
option(onnxruntime_MINIMAL_BUILD "Exclude as much as possible from the build. Support ORT format models. No support for ONNX format models." OFF)
cmake_dependent_option(onnxruntime_DISABLE_RTTI "Disable RTTI" ON "NOT onnxruntime_ENABLE_PYTHON;NOT onnxruntime_USE_CUDA" OFF)
# For now onnxruntime_DISABLE_EXCEPTIONS will only work with onnxruntime_MINIMAL_BUILD, more changes (ONNX, non-CPU EP, ...) are required to run this standalone
cmake_dependent_option(onnxruntime_DISABLE_EXCEPTIONS "Disable exception handling. Requires onnxruntime_MINIMAL_BUILD currently." ON "onnxruntime_MINIMAL_BUILD;NOT onnxruntime_ENABLE_PYTHON" OFF)
# Even when onnxruntime_DISABLE_ABSEIL is ON, ONNX Runtime still needs to link to abseil.
option(onnxruntime_DISABLE_ABSEIL "Do not use Abseil data structures in ONNX Runtime source code. Redefine Inlined containers to STD containers." OFF)

option(onnxruntime_EXTENDED_MINIMAL_BUILD "onnxruntime_MINIMAL_BUILD with support for execution providers that compile kernels." OFF)
option(onnxruntime_MINIMAL_BUILD_CUSTOM_OPS "Add custom operator kernels support to a minimal build." OFF)
option(onnxruntime_REDUCED_OPS_BUILD "Reduced set of kernels are registered in build via modification of the kernel registration source files." OFF)
option(onnxruntime_DISABLE_EXTERNAL_INITIALIZERS "Don't allow models to load external data" OFF)

#A special option just for debugging and sanitize check. Please do not enable in option in retail builds.
#The option has no effect on Windows.
option(onnxruntime_USE_VALGRIND "Build with valgrind hacks" OFF)

# A special build option only used for gathering code coverage info
option(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE "Run model tests even in debug mode" OFF)

# options for security fuzzing
# build configuration for fuzz testing is in onnxruntime_fuzz_test.cmake
option(onnxruntime_FUZZ_TEST "Enable Fuzz testing" OFF)

# training options
option(onnxruntime_ENABLE_NVTX_PROFILE "Enable NVTX profile." OFF)
option(onnxruntime_ENABLE_MEMORY_PROFILE "Enable memory profile." OFF)
option(onnxruntime_ENABLE_TRAINING "Enable full training functionality. Includes ORTModule and ORT Training APIs" OFF)
option(onnxruntime_ENABLE_TRAINING_APIS "Enable ort training apis." OFF)
option(onnxruntime_ENABLE_TRAINING_OPS "Include training operators but no training session support." OFF)
option(onnxruntime_ENABLE_TRAINING_E2E_TESTS "Enable training end-to-end tests." OFF)
option(onnxruntime_ENABLE_CPU_FP16_OPS "Build with advanced instruction sets" ON)
option(onnxruntime_USE_NCCL "Build with NCCL support" OFF)
option(onnxruntime_USE_MPI "Build with MPI support" OFF)

# WebAssembly options
option(onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB "Enable this option to create WebAssembly static library" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_THREADS "Enable this option to create WebAssembly byte codes with multi-threads support" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING "Enable this option to turn on exception catching" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_API_EXCEPTION_CATCHING "Enable this option to turn on api exception catching" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING "Enable this option to turn on exception throwing even if the build disabled exceptions support" OFF)
option(onnxruntime_WEBASSEMBLY_RUN_TESTS_IN_BROWSER "Enable this option to run tests in browser instead of Node.js" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO "Enable this option to turn on DWARF format debug info" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_PROFILING "Enable this option to turn on WebAssembly profiling and preserve function names" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL "Enable this option to allow WebAssembly to output optimized model" OFF)

# Enable bitcode for iOS
option(onnxruntime_ENABLE_BITCODE "Enable bitcode for iOS only" OFF)

# build Pytorch's LazyTensor support
cmake_dependent_option(onnxruntime_ENABLE_LAZY_TENSOR "Enable ORT as a LazyTensor backend in Pytorch." ON "onnxruntime_ENABLE_TRAINING" OFF)

# build separate library of schemas of (custom) ops used by ORT (for ONNX to MLIR translation)
option(onnxruntime_BUILD_OPSCHEMA_LIB "Build op schema library" ON)

# option to enable custom operators in onnxruntime-extensions
option(onnxruntime_USE_EXTENSIONS "Build with onnxruntime-extensions for more features" OFF)

# option to enable custom operators in onnxruntime-extensions with a custom path
option(onnxruntime_EXTENSIONS_OVERRIDDEN, "Enable onnxruntime-extensions overridden" OFF)

# Enable registering custom op schemas from shared libraries in python environment.
option(onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS "Enable registering user defined custom op schemas dynamically" OFF)

set(ONNX_CUSTOM_PROTOC_EXECUTABLE "" CACHE STRING "Specify custom protoc executable to build ONNX")

# pre-build python path
option(onnxruntime_PREBUILT_PYTORCH_PATH "Path to pytorch installation dir")

# external transformer src path
option(onnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH "Path to external transformer src dir")

option(onnxruntime_ENABLE_CUDA_PROFILING "Enable CUDA kernel profiling" OFF)
option(onnxruntime_ENABLE_ROCM_PROFILING "Enable ROCM kernel profiling" OFF)

option(onnxruntime_ENABLE_CPUINFO "Enable cpuinfo" ON)

# ATen fallback support
option(onnxruntime_ENABLE_ATEN "Enable ATen fallback" OFF)

# Triton support
option(onnxruntime_ENABLE_TRITON "Enable Triton" OFF)

# composable kernel is managed automatically, unless user want to explicitly disable it, it should not be manually set
option(onnxruntime_USE_COMPOSABLE_KERNEL "Enable composable kernel for ROCm EP" ON)
option(onnxruntime_USE_ROCBLAS_EXTENSION_API "Enable rocblas tuning for ROCm EP" OFF)
option(onnxruntime_USE_TRITON_KERNEL "Enable triton compiled kernel" OFF)
option(onnxruntime_BUILD_KERNEL_EXPLORER "Build Kernel Explorer for testing and profiling GPU kernels" OFF)

option(onnxruntime_BUILD_CACHE "onnxruntime build with cache" OFF)
# https://zeux.io/2010/11/22/z7-everything-old-is-new-again/
cmake_dependent_option(MSVC_Z7_OVERRIDE "replacing /Zi and /ZI with /Z7 when using MSVC with CCache" ON "onnxruntime_BUILD_CACHE; MSVC" OFF)

option(onnxruntime_USE_AZURE "Build with azure inferencing support" OFF)
option(onnxruntime_USE_LOCK_FREE_QUEUE "Build with lock-free task queue for threadpool." OFF)

# ENABLE_TRAINING includes all training functionality
# The following 2 entry points
# 1. ORTModule
# 2. ORT Training APIs
# It includes all the feature additions as well like
# 1. Python OP
# 2. Aten Fallback
# 3. Strided Tensors
# 4. All training ops including communication and  collectives ops
# 5. ONNXBlock (Front end for training preparation when using training apis)
# Some features are only enabled when onnxruntime_ENABLE_PYTHON is ON as they are only relevant
# when using python env
if (onnxruntime_ENABLE_TRAINING)
  set(onnxruntime_ENABLE_TRAINING_APIS ON)
  set(onnxruntime_ENABLE_TRAINING_OPS ON)
  set(onnxruntime_ENABLE_ATEN ON)
  set(onnxruntime_ENABLE_TRITON ON)
  if (NOT APPLE)
    set(onnxruntime_ENABLE_TRAINING_TORCH_INTEROP ON)
  endif()
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
  set(onnxruntime_ENABLE_TRAINING_OPS ON)
  if (onnxruntime_ENABLE_PYTHON AND NOT onnxruntime_ENABLE_TRAINING)
    message(FATAL_ERROR "Standalone On-Device Training build is not supported with Python bindings! "
    "Please use the --enable_training flag instead of the --enable_training_apis flag.")
  endif()
endif()

if (onnxruntime_USE_ROCM)
  if (WIN32)
    message(FATAL_ERROR "ROCM does not support build in Windows!")
  endif()
  if (onnxruntime_USE_CUDA)
    message(FATAL_ERROR "ROCM does not support build with CUDA!")
  endif()

  if (NOT CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER "${onnxruntime_ROCM_HOME}/llvm/bin/clang++")
  endif()

  if (NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101")
  endif()

  file(GLOB rocm_cmake_components ${onnxruntime_ROCM_HOME}/lib/cmake/*)
  list(APPEND CMAKE_PREFIX_PATH ${rocm_cmake_components})
  # Force cmake to accept the configured HIP compiler. Because the configured CMAKE_PREFIX_PATH does not work during
  # enable_language(HIP), we might need to move configuring of CMAKE_PREFIX_PATH to build.py (in the future).
  set(CMAKE_HIP_COMPILER_FORCED ON)

  enable_language(HIP)
  # NOTE: Flags -mllvm -amdgpu-early-inline-all=true are critical for gpu kernel code performance. -mllvm passes the
  # next flag to underlying LLVM instead of clang and -amdgpu-early-inline-all=true is the optimization flag for LLVM.
  # With CMake's enable_language(HIP), additional flags including the proceeding one are propagated from
  # hip-lang::device library. But in some weird cases, the hip-lang::device target may not be properly configured, for
  # example, the CMAKE_PREFIX_PATH might be improperly configured.
  if(NOT DEFINED _CMAKE_HIP_DEVICE_RUNTIME_TARGET)
    message(FATAL_ERROR "HIP Language is not properly configured.")
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:SHELL:-x hip>")

  if (NOT onnxruntime_HIPIFY_PERL)
    find_path(HIPIFY_PERL_PATH
      NAMES hipify-perl
      HINTS
      ${onnxruntime_ROCM_HOME}/bin
      ${onnxruntime_ROCM_HOME}/hip/bin)
    if (HIPIFY_PERL_PATH-NOTFOUND)
      MESSAGE(FATAL_ERROR "hipify-perl not found")
    endif()
    set(onnxruntime_HIPIFY_PERL ${HIPIFY_PERL_PATH}/hipify-perl)
  endif()

  # replicate strategy used by pytorch to get ROCM_VERSION
  # https://github.com/pytorch/pytorch/blob/5c5b71b6eebae76d744261715231093e62f0d090/cmake/public/LoadHIP.cmake
  # with modification
  if (EXISTS "${onnxruntime_ROCM_HOME}/.info/version")
    file(READ "${onnxruntime_ROCM_HOME}/.info/version" ROCM_VERSION_DEV_RAW)
    string(REGEX MATCH "^([0-9]+)\.([0-9]+)\.([0-9]+)-.*$" ROCM_VERSION_MATCH ${ROCM_VERSION_DEV_RAW})
  elseif (EXISTS "${onnxruntime_ROCM_HOME}/include/rocm_version.h")
    file(READ "${onnxruntime_ROCM_HOME}/include/rocm_version.h" ROCM_VERSION_H_RAW)
    string(REGEX MATCH "\"([0-9]+)\.([0-9]+)\.([0-9]+).*\"" ROCM_VERSION_MATCH ${ROCM_VERSION_H_RAW})
  elseif (EXISTS "${onnxruntime_ROCM_HOME}/include/rocm-core/rocm_version.h")
    file(READ "${onnxruntime_ROCM_HOME}/include/rocm-core/rocm_version.h" ROCM_VERSION_H_RAW)
    string(REGEX MATCH "\"([0-9]+)\.([0-9]+)\.([0-9]+).*\"" ROCM_VERSION_MATCH ${ROCM_VERSION_H_RAW})
  endif()

  if (ROCM_VERSION_MATCH)
    set(ROCM_VERSION_DEV_MAJOR ${CMAKE_MATCH_1})
    set(ROCM_VERSION_DEV_MINOR ${CMAKE_MATCH_2})
    set(ROCM_VERSION_DEV_PATCH ${CMAKE_MATCH_3})
    set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
    math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")
  else()
    message(FATAL_ERROR "Cannot determine ROCm version string")
  endif()
  message("\n***** ROCm version from ${onnxruntime_ROCM_HOME}/.info/version ****\n")
  message("ROCM_VERSION_DEV: ${ROCM_VERSION_DEV}")
  message("ROCM_VERSION_DEV_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message("ROCM_VERSION_DEV_MINOR: ${ROCM_VERSION_DEV_MINOR}")
  message("ROCM_VERSION_DEV_PATCH: ${ROCM_VERSION_DEV_PATCH}")
  message("ROCM_VERSION_DEV_INT:   ${ROCM_VERSION_DEV_INT}")
  message("\n***** HIP LANGUAGE CONFIG INFO ****\n")
  message("CMAKE_HIP_COMPILER:      ${CMAKE_HIP_COMPILER}")
  message("CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
  message("CMAKE_HIP_FLAGS:         ${CMAKE_HIP_FLAGS}")
  string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
  message("CMAKE_HIP_FLAGS_${BUILD_TYPE}: ${CMAKE_HIP_FLAGS_${BUILD_TYPE}}")
  add_definitions(-DROCM_VERSION=${ROCM_VERSION_DEV_INT})

  if (onnxruntime_USE_COMPOSABLE_KERNEL AND ROCM_VERSION_DEV VERSION_LESS "5.3")
    message(WARNING "composable kernel is only supported on ROCm >= 5.3")
    set(onnxruntime_USE_COMPOSABLE_KERNEL OFF)
  endif()
endif()



# Single output director for all binaries
set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "Single output directory for all binaries.")


include(FetchContent)

function(set_msvc_c_cpp_compiler_warning_level warning_level)
  if (NOT "${warning_level}" MATCHES "^[0-4]$")
    message(FATAL_ERROR "Expected warning_level value of 0-4, got '${warning_level}'.")
  endif()

  if (MSVC)
    set(warning_flag "/W${warning_level}")
    get_property(opts DIRECTORY PROPERTY COMPILE_OPTIONS)
    # only match the generator expression added by this function
    list(FILTER opts
         EXCLUDE REGEX "^\\$<\\$<OR:\\$<COMPILE_LANGUAGE:C>,\\$<COMPILE_LANGUAGE:CXX>>:/W[0-4]>$")
    list(APPEND opts "$<$<COMPILE_LANGUAGE:CXX,C>:${warning_flag}>")
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "${opts}")
  endif()
endfunction()


if(MSVC_Z7_OVERRIDE)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif(MSVC_Z7_OVERRIDE)

# set default MSVC warning level to 3 for external dependencies
set_msvc_c_cpp_compiler_warning_level(3)

# Fuzz test has only been tested with BUILD_SHARED_LIB option,
# using the MSVC compiler and on windows OS.
if (MSVC AND WIN32 AND onnxruntime_FUZZ_TEST AND onnxruntime_BUILD_SHARED_LIB AND onnxruntime_USE_FULL_PROTOBUF)
  # Fuzz test library dependency, protobuf-mutator,
  # needs the onnx message to be compiled using "non-lite protobuf version"
  set(onnxruntime_FUZZ_ENABLED ON)
endif()

if (onnxruntime_USE_VALGRIND AND NOT WIN32)
  add_definitions(-DRE2_ON_VALGRIND=1)
endif()

if (onnxruntime_ENABLE_NVTX_PROFILE)
  add_definitions(-DENABLE_NVTX_PROFILE=1)
endif()

if (onnxruntime_ENABLE_BITCODE)
  if (NOT (CMAKE_SYSTEM_NAME STREQUAL "iOS"))
    message(FATAL_ERROR "iOS platform required for onnxruntime_ENABLE_BITCODE")
  endif()
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE YES)
  set(CMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE "bitcode")
else()
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)
endif()

if (onnxruntime_ENABLE_MEMORY_PROFILE)
  add_definitions(-DORT_MEMORY_PROFILE=1)
endif()

set(ONNX_ML 1)
if (NOT onnxruntime_ENABLE_PYTHON)
  set(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS OFF)
endif()

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  add_compile_definitions(ENABLE_LANGUAGE_INTEROP_OPS)
endif()

if (NOT (UNIX AND onnxruntime_ENABLE_PYTHON AND onnxruntime_ENABLE_TRAINING AND (NOT onnxruntime_BUILD_SHARED_LIB)))
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    message(WARNING "onnxruntime_ENABLE_TRAINING_TORCH_INTEROP is turned OFF due to incompatible build combinations.")
  endif()
  set(onnxruntime_ENABLE_TRAINING_TORCH_INTEROP OFF)
endif()

if (NOT (UNIX AND onnxruntime_USE_CUDA AND onnxruntime_ENABLE_PYTHON AND onnxruntime_ENABLE_TRAINING AND (NOT onnxruntime_BUILD_SHARED_LIB)))
  if (onnxruntime_ENABLE_TRITON)
    message(WARNING "onnxruntime_ENABLE_TRITON is turned OFF because it's designed to support CUDA training on Linux only currently.")
  endif()
  set(onnxruntime_ENABLE_TRITON OFF)
endif()

set(onnxruntime_REQUIRE_PYTHON_EMBED_LIB OFF)
if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  add_compile_definitions(ENABLE_TRAINING_TORCH_INTEROP)

  # Python::Python is required for building unit test executables.
  if (onnxruntime_BUILD_UNIT_TESTS)
    set(onnxruntime_REQUIRE_PYTHON_EMBED_LIB ON)
  endif()
endif()

if (onnxruntime_ENABLE_TRITON)
  # Need SetOutputMLValue.
  set(onnxruntime_ENABLE_ATEN ON)
  add_compile_definitions(ENABLE_TRITON)

  # Python::Python is required for building unit test executables.
  if (onnxruntime_BUILD_UNIT_TESTS)
    set(onnxruntime_REQUIRE_PYTHON_EMBED_LIB ON)
  endif()
endif()

# General C# properties
if (onnxruntime_BUILD_CSHARP)
  check_language(CSharp)
  if (CMAKE_CSharp_COMPILER)
    enable_language(CSharp)
    set(CMAKE_CSharp_FLAGS ${CMAKE_CSharp_FLAGS} "/langversion:6")
    message(STATUS "CMAKE_Csharp_Compiler = ${CMAKE_CSharp_COMPILER}")
  else()
    message(WARNING "Language Csharp is not found in the system")
  endif()
endif()

if (NOT WIN32)
  #TODO: On Linux we may try https://github.com/microsoft/TraceLogging.git
  if (onnxruntime_ENABLE_INSTRUMENT)
    message(WARNING "Instrument is only supported on Windows now")
    set(onnxruntime_ENABLE_INSTRUMENT OFF)
  endif()
endif()

# 'extended' implies minimal.
if (onnxruntime_EXTENDED_MINIMAL_BUILD AND NOT onnxruntime_MINIMAL_BUILD)
  set(onnxruntime_MINIMAL_BUILD ON)
endif()

include(adjust_global_compile_flags.cmake)

if (APPLE)
  if (NOT CMAKE_OSX_ARCHITECTURES)
    message("Building ONNX Runtime for ${CMAKE_HOST_SYSTEM_PROCESSOR} CPU ARCH")
  endif()
elseif (NOT WIN32 AND NOT APPLE)
  message("Building ONNX Runtime for ${onnxruntime_target_platform} CPU ARCH")
endif()

# We need to link with libatomic on systems that do not have built-in atomics, or
# don't have built-in support for 8 byte atomics
# Derived from https://github.com/protocolbuffers/protobuf/blob/master/cmake/CMakeLists.txt
set(onnxruntime_LINK_LIBATOMIC false)
# We don't need to link libatomic on iOS
if (NOT MSVC AND NOT (CMAKE_SYSTEM_NAME STREQUAL "iOS"))
  include(CheckCXXSourceCompiles)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} -std=c++11)
  check_cxx_source_compiles("
    #include <atomic>
    int main() {
      return std::atomic<int64_t>{};
    }
  " onnxruntime_HAVE_BUILTIN_ATOMICS)
  if (NOT onnxruntime_HAVE_BUILTIN_ATOMICS)
    set(onnxruntime_LINK_LIBATOMIC true)
  endif()
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endif()

set(REPO_ROOT ${PROJECT_SOURCE_DIR}/..)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../onnxruntime)
set(ORTTRAINING_ROOT ${PROJECT_SOURCE_DIR}/../orttraining)
set(ORTTRAINING_SOURCE_DIR ${ORTTRAINING_ROOT}/orttraining)

file (STRINGS "${REPO_ROOT}/VERSION_NUMBER" ORT_VERSION)

find_package(Threads)
# On Windows we directly use Windows APIs to do the job
# Android NDK doesn't provide the iconv lib.
if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
  find_package(Iconv REQUIRED)
  set(ICONV_LIB Iconv::Iconv)
endif()

find_package(Patch)
if (WIN32 AND NOT Patch_FOUND)
    # work around CI machines missing patch from the git install by falling back to the binary in this repo.
    # replicate what happens in https://github.com/Kitware/CMake/blob/master/Modules/FindPatch.cmake but without
    # the hardcoded suffixes in the path to the patch binary.
    find_program(Patch_EXECUTABLE NAMES patch PATHS ${PROJECT_SOURCE_DIR}/external/git.Win32.2.41.03.patch)
    if(Patch_EXECUTABLE)
      set(Patch_FOUND 1)
      if (NOT TARGET Patch::patch)
        add_executable(Patch::patch IMPORTED)
        set_property(TARGET Patch::patch PROPERTY IMPORTED_LOCATION ${Patch_EXECUTABLE})
      endif()
    endif()
endif()
if(Patch_FOUND)
  message("Patch found: ${Patch_EXECUTABLE}")
endif()

if (CMAKE_CROSSCOMPILING)
  message("Doing crosscompiling")
endif()

#Need python to generate def file
if (onnxruntime_BUILD_SHARED_LIB OR onnxruntime_ENABLE_PYTHON)
  if (onnxruntime_ENABLE_PYTHON)
    if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS OR onnxruntime_REQUIRE_PYTHON_EMBED_LIB)
      find_package(Python 3.8 COMPONENTS Interpreter Development NumPy)
    else()
      find_package(Python 3.8 COMPONENTS Interpreter Development.Module NumPy)
    endif()
  else()
    find_package(Python 3.8 COMPONENTS Interpreter)
  endif()
endif()

#Dependencies begin
get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
get_filename_component(ORTTRAINING_ROOT "${ORTTRAINING_ROOT}" ABSOLUTE)
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_INCLUDE_DIR ${REPO_ROOT}/include/onnxruntime)

include(external/onnxruntime_external_deps.cmake)

set(ORT_WARNING_FLAGS)
if (WIN32)
    # class needs to have dll-interface to be used by clients
    list(APPEND ORT_WARNING_FLAGS "/wd4251")
    # issued by thrust nonstandard extension used: nameless struct/union
    list(APPEND ORT_WARNING_FLAGS "/wd4201")
    # structure was padded due to __declspec(align())
    list(APPEND ORT_WARNING_FLAGS "/wd4324")
    # warning C4800: Implicit conversion from 'X' to bool. Possible information loss
    if (onnxruntime_USE_OPENVINO)
       list(APPEND ORT_WARNING_FLAGS "/wd4800")
    endif()
    # operator 'operator-name': deprecated between enumerations of different types
    list(APPEND ORT_WARNING_FLAGS "/wd5054")
    # Enable warning: data member 'member' will be initialized after data member 'member2' / base class 'base_class'
    list(APPEND ORT_WARNING_FLAGS "/w15038")

    # set linker flags to minimize the binary size.
    if (MSVC)
      foreach(type EXE STATIC SHARED)
        if (NOT type MATCHES STATIC)
          # The WinML internal toolchain does not allow link's "additional options" to contain optimization
          # flags (/OPT#); these are already specified using msbuild properties.
          if (NOT DEFINED onnxruntime_DISABLE_LINKER_OPT_FLAGS)
            set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO")
            set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO")
            set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /INCREMENTAL:NO")
          endif()
        endif()
        if (onnxruntime_ENABLE_LTO AND NOT onnxruntime_USE_CUDA)
          set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /LTCG")
        endif()
      endforeach()
    endif()

else()
  check_cxx_compiler_flag(-Wambiguous-reversed-operator HAS_AMBIGUOUS_REVERSED_OPERATOR)
  check_cxx_compiler_flag(-Wbitwise-instead-of-logical HAS_BITWISE_INSTEAD_OF_LOGICAL)
  check_cxx_compiler_flag(-Wcast-function-type HAS_CAST_FUNCTION_TYPE)
  check_cxx_compiler_flag(-Wcatch-value HAS_CATCH_VALUE)
  check_cxx_compiler_flag(-Wclass-memaccess HAS_CLASS_MEMACCESS)
  check_cxx_compiler_flag(-Wdeprecated-anon-enum-enum-conversion HAS_DEPRECATED_ANON_ENUM_ENUM_CONVERSION)
  check_cxx_compiler_flag(-Wdeprecated-builtins HAS_DEPRECATED_BUILTINS)
  check_cxx_compiler_flag(-Wdeprecated-copy HAS_DEPRECATED_COPY)
  check_cxx_compiler_flag(-Wdeprecated-declarations HAS_DEPRECATED_DECLARATIONS)
  check_cxx_compiler_flag(-Wenum-constexpr-conversion HAS_ENUM_CONSTEXPR_CONVERSION)
  check_cxx_compiler_flag(-Wformat-truncation HAS_FORMAT_TRUNCATION)
  check_cxx_compiler_flag(-Wignored-attributes HAS_IGNORED_ATTRIBUTES)
  check_cxx_compiler_flag(-Wmaybe-uninitialized HAS_MAYBE_UNINITIALIZED)
  check_cxx_compiler_flag(-Wmissing-braces HAS_MISSING_BRACES)
  check_cxx_compiler_flag(-Wnonnull-compare HAS_NONNULL_COMPARE)
  check_cxx_compiler_flag(-Wparentheses HAS_PARENTHESES)
  check_cxx_compiler_flag(-Wshorten-64-to-32 HAS_SHORTEN_64_TO_32)
  check_cxx_compiler_flag(-Wstrict-aliasing HAS_STRICT_ALIASING)
  check_nvcc_compiler_flag(-Wstrict-aliasing NVCC_HAS_STRICT_ALIASING)
  check_cxx_compiler_flag(-Wtautological-pointer-compare HAS_TAUTOLOGICAL_POINTER_COMPARE)
  check_cxx_compiler_flag(-Wundefined-var-template HAS_UNDEFINED_VAR_TEMPLATE)
  check_cxx_compiler_flag(-Wunused-but-set-parameter HAS_UNUSED_BUT_SET_PARAMETER)
  check_cxx_compiler_flag(-Wunused-but-set-variable HAS_UNUSED_BUT_SET_VARIABLE)
  check_cxx_compiler_flag(-Wunused-variable HAS_UNUSED_VARIABLE)
  check_cxx_compiler_flag(-Wuseless-cast HAS_USELESS_CAST)
  check_cxx_compiler_flag(-Wstringop-overflow HAS_STRINGOP_OVERFLOW)
  check_function_exists(reallocarray HAS_REALLOCARRAY)
  if (NOT APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_target_platform STREQUAL "aarch64")
   check_cxx_compiler_flag(-march=armv8.2-a+bf16 HAS_ARM64_BFLOAT16)
   if(NOT HAS_ARM64_BFLOAT16)
     message(FATAL_ERROR  "The compiler doesn't support BFLOAT16!!!")
   endif()
   check_cxx_compiler_flag(-march=armv8.2-a+fp16 HAS_ARM64_FLOAT16)
   if(NOT HAS_ARM64_FLOAT16)
     message(FATAL_ERROR  "The compiler doesn't support FLOAT16!!!")
   endif()
  endif()
  if (HAS_TAUTOLOGICAL_POINTER_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    list(APPEND ORT_WARNING_FLAGS -Wno-tautological-pointer-compare)
  endif()
  if (HAS_NONNULL_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    list(APPEND ORT_WARNING_FLAGS -Wno-nonnull-compare)
  endif()

  if (HAS_AMBIGUOUS_REVERSED_OPERATOR)
    list(APPEND ORT_WARNING_FLAGS -Wno-ambiguous-reversed-operator)
  endif()
  if (HAS_DEPRECATED_ANON_ENUM_ENUM_CONVERSION)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-anon-enum-enum-conversion)
  endif()
  if (HAS_UNDEFINED_VAR_TEMPLATE)
    list(APPEND ORT_WARNING_FLAGS -Wno-undefined-var-template)
  endif()
  if (HAS_DEPRECATED_BUILTINS)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-builtins)
  endif()
  #see:https://reviews.llvm.org/D131307
  #It was intended that the 'enum-constexpr-conversion' type warnings can not be silenced by -w
  if(HAS_ENUM_CONSTEXPR_CONVERSION AND NOT Protobuf_FOUND)
    if (TARGET libprotobuf)
      target_compile_options(libprotobuf PRIVATE "-Wno-enum-constexpr-conversion")
    endif()
    if (TARGET libprotobuf-lite)
      target_compile_options(libprotobuf-lite PRIVATE "-Wno-enum-constexpr-conversion")
    endif()
  endif()

  # enable warning(s) that may not be on by default
  if (HAS_SHORTEN_64_TO_32)
    list(APPEND ORT_WARNING_FLAGS -Wshorten-64-to-32)
  endif()
endif()

#names in this var must match the directory names under onnxruntime/core/providers
#ONNXRUNTIME_PROVIDER_NAMES is the list of providers that needs to export additional symbols in the global namespace.
#For example CUDA EP exports "OrtSessionOptionsAppendExecutionProvider_CUDA", which is a global function.
#However, all these things are legacy and deprecated and should be replaced with functions in onnxruntime_c_api.h.
set(ONNXRUNTIME_PROVIDER_NAMES cpu)

set(ORT_PROVIDER_FLAGS)
set(ORT_PROVIDER_CMAKE_FLAGS)

if (onnxruntime_USE_CUDA)
  if (onnxruntime_USE_CUDA_NHWC_OPS)
    add_compile_definitions(ENABLE_CUDA_NHWC_OPS)
  endif()
  enable_language(CUDA)
  message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

  if (onnxruntime_DISABLE_CONTRIB_OPS)
    set(onnxruntime_USE_FLASH_ATTENTION OFF)
    set(onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION OFF)
  endif()
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.6)
    message( STATUS "Turn off flash attention since CUDA compiler version < 11.6")
    set(onnxruntime_USE_FLASH_ATTENTION OFF)
    set(onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION OFF)
  endif()
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.4)
    message( FATAL_ERROR "Failed build due to CUDA compiler version < 11.4")
  endif()
else()
  set(onnxruntime_USE_FLASH_ATTENTION OFF)
  set(onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION OFF)
endif()

if (onnxruntime_USE_CUDA)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_CUDA=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_CUDA=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES cuda)

    if (onnxruntime_USE_FLASH_ATTENTION)
      message( STATUS "Enable flash attention for CUDA EP")
      list(APPEND ORT_PROVIDER_FLAGS -DUSE_FLASH_ATTENTION=1)
      list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_FLASH_ATTENTION=1)
    endif()
    if (onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION)
      message( STATUS "Enable memory efficient attention for CUDA EP")
      list(APPEND ORT_PROVIDER_FLAGS -DUSE_MEMORY_EFFICIENT_ATTENTION=1)
      list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=1)
    endif()
endif()

if (onnxruntime_USE_VITISAI)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_VITISAI=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_VITISAI=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES vitisai)
endif()
if (onnxruntime_USE_DNNL)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_DNNL=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES dnnl)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_DNNL=1)
endif()
if (onnxruntime_USE_OPENVINO)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_OPENVINO=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES openvino)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_OPENVINO=1)
endif()
if (onnxruntime_USE_TENSORRT)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_TENSORRT=1)
    #TODO: remove the following line and change the test code in onnxruntime_shared_lib_test to use the new EP API.
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES tensorrt)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_TENSORRT=1)
endif()
if (onnxruntime_USE_RKNPU)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_RKNPU=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_RKNPU=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES rknpu)
endif()
if (onnxruntime_USE_NNAPI_BUILTIN)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_NNAPI=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_NNAPI_BUILTIN=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES nnapi)
endif()
if (onnxruntime_USE_JSEP)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_JSEP=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_JSEP=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES js)
endif()
if (onnxruntime_USE_QNN)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_QNN=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_QNN=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES qnn)
endif()
if (onnxruntime_USE_SNPE)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_SNPE=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES snpe)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_SNPE=1)
endif()
if (onnxruntime_USE_TVM)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_TVM=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_TVM=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES tvm)
endif()
if (onnxruntime_USE_WINML)
  list(APPEND ORT_PROVIDER_FLAGS  -DUSE_WINML=1)
  list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_WINML=1)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES winml)
endif()
if (onnxruntime_USE_ACL)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_ACL=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_ACL=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES acl)
endif()
if (onnxruntime_USE_DML)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_DML=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_DML=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES dml)
endif()
if (onnxruntime_USE_MIGRAPHX)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_MIGRAPHX=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_MIGRAPHX=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES migraphx)
endif()
if (onnxruntime_USE_ARMNN)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_ARMNN=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_ARMNN=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES armnn)
endif()
if (onnxruntime_USE_ROCM)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_ROCM=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_ROCM=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES rocm)
endif()
if (onnxruntime_USE_COREML)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_COREML=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_COREML=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES coreml)
endif()
if (onnxruntime_USE_XNNPACK)
  list(APPEND ORT_PROVIDER_FLAGS -DUSE_XNNPACK=1)
  list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_XNNPACK=1)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES xnnpack)
endif()
if (onnxruntime_USE_WEBNN)
  list(APPEND ORT_PROVIDER_FLAGS -DUSE_WEBNN=1)
  list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_WEBNN=1)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES webnn)
endif()
if (onnxruntime_USE_CANN)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_CANN=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_CANN=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES cann)
endif()
if (onnxruntime_USE_AZURE)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_AZURE=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_AZURE=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES azure)
endif()
if (onnxruntime_USE_LOCK_FREE_QUEUE)
    add_compile_definitions(USE_LOCK_FREE_QUEUE)
endif()

if (onnxruntime_ENABLE_LAZY_TENSOR)
  # To support LazyTensor, ORT needs to call Python function from C/C++.
  # so onnxruntime_ENABLE_PYTHON is required.
  if (NOT onnxruntime_ENABLE_TRAINING OR NOT onnxruntime_ENABLE_PYTHON)
    message(
        FATAL_ERROR
        "Option onnxruntime_ENABLE_LAZY_TENSOR can only be set when onnxruntime_ENABLE_TRAINING and onnxruntime_ENABLE_PYTHON are enabled")
  endif()
  # TODO: In the future, we can compile LazyTensor into a standalone
  # library target, onnxruntime_lazy_tensor, to make the buid
  # cleaner.
endif()

function(onnxruntime_set_compile_flags target_name)
    if (CPUINFO_SUPPORTED)
      onnxruntime_add_include_to_target(${target_name} cpuinfo::cpuinfo)
    endif()
    if(onnxruntime_ENABLE_LAZY_TENSOR)
      target_compile_definitions(${target_name} PRIVATE ENABLE_LAZY_TENSOR)
    endif()
    if (onnxruntime_ENABLE_CPU_FP16_OPS)
      target_compile_definitions(${target_name} PRIVATE ENABLE_CPU_FP16_TRAINING_OPS)
    endif()
    if(onnxruntime_DISABLE_ABSEIL)
      target_compile_definitions(${target_name} PRIVATE DISABLE_ABSEIL)
    endif()
    if(UNIX)
      target_compile_definitions(${target_name} PRIVATE PLATFORM_POSIX)
    endif()
    target_compile_definitions(${target_name} PRIVATE EIGEN_USE_THREADS)
    if (onnxruntime_DISABLE_CONTRIB_OPS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_CONTRIB_OPS)
    endif()

    if (onnxruntime_DISABLE_ML_OPS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_ML_OPS)
    endif()

    if (onnxruntime_DISABLE_SPARSE_TENSORS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_SPARSE_TENSORS)
    endif()

    if (onnxruntime_DISABLE_OPTIONAL_TYPE)
      target_compile_definitions(${target_name} PRIVATE DISABLE_OPTIONAL_TYPE)
    endif()

    if (onnxruntime_DISABLE_FLOAT8_TYPES)
      target_compile_definitions(${target_name} PRIVATE DISABLE_FLOAT8_TYPES)
    endif()

    if (onnxruntime_ENABLE_ATEN)
      target_compile_definitions(${target_name} PRIVATE ENABLE_ATEN)
    endif()

    if(USE_NEURAL_SPEED)
      target_compile_definitions(${target_name} PRIVATE ORT_NEURAL_SPEED)
    endif()

    set_target_properties(${target_name} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    if (onnxruntime_USE_CUDA)
      # Suppress a "conversion_function_not_usable" warning in gsl/span
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe \"--diag_suppress=conversion_function_not_usable\">")
      target_compile_definitions(${target_name} PRIVATE -DDISABLE_CUSPARSE_DEPRECATED)
    endif()
    if (MSVC)
      foreach(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY}>")
      endforeach()

      foreach(onnxruntime_external_lib IN LISTS onnxruntime_EXTERNAL_LIBRARIES)
        #TODO: the list contains cmake keywords like "debug". We should exclude them.
        if(TARGET ${onnxruntime_external_lib})
          get_target_property(onnxruntime_external_lib_include_dirs ${onnxruntime_external_lib} INTERFACE_INCLUDE_DIRECTORIES)
          foreach(onnxruntime_external_lib_include_dir IN LISTS onnxruntime_external_lib_include_dirs)
            if(onnxruntime_external_lib_include_dir MATCHES "^\\$")
              if(onnxruntime_external_lib_include_dir MATCHES "^\\$<BUILD_INTERFACE:([^>]+)>$")
                string(REGEX REPLACE "^\\$<BUILD_INTERFACE:([^>]+)>$" "\\1" onnxruntime_external_lib_include_dir_cmake "${onnxruntime_external_lib_include_dir}")
                cmake_path(NATIVE_PATH onnxruntime_external_lib_include_dir_cmake NORMALIZE onnxruntime_external_lib_include_dir_native)
                target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${onnxruntime_external_lib_include_dir_native}>")
              endif()
            else()
              cmake_path(NATIVE_PATH onnxruntime_external_lib_include_dir NORMALIZE onnxruntime_external_lib_include_dir_native)
              target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${onnxruntime_external_lib_include_dir_native}>")
            endif()
          endforeach()
        endif()
      endforeach()
      target_compile_definitions(${target_name} PRIVATE -DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
      if (onnxruntime_ENABLE_MEMLEAK_CHECKER)
        target_compile_definitions(${target_name} PRIVATE -DONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
      endif()
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<COMPILE_LANGUAGE:CXX,C>:/utf-8>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /sdl>" "$<$<COMPILE_LANGUAGE:CXX,C>:/sdl>")
      set_target_properties(${target_name}
                      PROPERTIES VS_GLOBAL_CAExcludePath "${ORT_BINARY_DIR};${ORT_SOURCE_DIR}")
      # We do not treat warnings from 3rd-party libraries as errors. In order to do that, we need to add their header files locations to /external:I.
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/experimental:external>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /experimental:external>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:W0>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:W0>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:templates->" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:templates->")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CURRENT_SOURCE_DIR}>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:I${CMAKE_CURRENT_SOURCE_DIR}>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CURRENT_BINARY_DIR}>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:I${CMAKE_CURRENT_BINARY_DIR}>")
      if (onnxruntime_ENABLE_STATIC_ANALYSIS)
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze>" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze>")
        if (onnxruntime_REDIRECT_STATIC_ANALYSIS_OUTPUTS_TO_FILE)
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze:autolog:ext.sarif>" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze:autolog:ext.sarif>")
        endif()
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze:external->" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze:external->")
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6385>" )
        # There are many such warnings from STL:
        # include\list(148): warning C6011: Dereferencing NULL pointer '_Mycont'. : Lines: 146, 147, 148
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6011>" )
      endif()
    else()
      # Enable warning
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wall>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall>")
      target_compile_options(${target_name} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wextra>")
      if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        #external/protobuf/src/google/protobuf/arena.h:445:18: error: unused parameter 'p'
        target_compile_options(${target_name} PRIVATE "-Wno-unused-parameter")
      endif()
      target_compile_definitions(${target_name} PUBLIC -DNSYNC_ATOMIC_CPP11)
      onnxruntime_add_include_to_target(${target_name} nsync::nsync_cpp)
    endif()
    foreach(ORT_FLAG ${ORT_PROVIDER_FLAGS})
      target_compile_definitions(${target_name} PRIVATE ${ORT_FLAG})
    endforeach()
    if (HAS_DEPRECATED_COPY)
      #too many such errors in eigen
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-deprecated-copy>" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>")
    endif()
    foreach(FLAG ${ORT_WARNING_FLAGS})
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${FLAG}>")
    endforeach()
    if (onnxruntime_USE_CUDA)
      foreach(FLAG ${ORT_WARNING_FLAGS})
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options ${FLAG}>")
      endforeach()
      if (NVCC_HAS_STRICT_ALIASING AND "${target_name}" MATCHES "cuda")
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-strict-aliasing>")
      endif()
      if (HAS_STRICT_ALIASING AND NOT "${target_name}" MATCHES "cuda")
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-strict-aliasing>")
      endif()
    endif()
    if (onnxruntime_USE_ROCM)
      # flags are detected with CXX language mode, some flags are not supported with hipclang
      # because we may mix gcc and hipclang
      set(ORT_HIP_WARNING_FLAGS ${ORT_WARNING_FLAGS})
      list(REMOVE_ITEM ORT_HIP_WARNING_FLAGS -Wno-nonnull-compare)

      # float16.h:90:12: error: ‘tmp’ is used uninitialized
      list(APPEND ORT_HIP_WARNING_FLAGS -Wno-uninitialized)
      list(APPEND ORT_HIP_WARNING_FLAGS -Wno-deprecated-copy)

      # some #pragma unroll will fail, do not treat them as error
      # #warning must not be treated as error
      list(APPEND ORT_HIP_WARNING_FLAGS -Wno-error=pass-failed "-Wno-error=#warnings")

      # otherwise error: builtin __has_trivial_assign is deprecated; use __is_trivially_assignable instead
      if (ROCM_VERSION_DEV VERSION_GREATER_EQUAL "5.4")
        list(APPEND ORT_HIP_WARNING_FLAGS "-Wno-deprecated-builtins")
      endif()

      foreach(FLAG ${ORT_HIP_WARNING_FLAGS})
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:SHELL:${FLAG}>")
      endforeach()
    endif()
endfunction()

function(onnxruntime_set_source_file_properties target_name)
  get_target_property(srcs ${target_name} SOURCES)

  # enable ARC for Objective-C/C++
  set(objective_c_cc_srcs ${srcs})
  list(FILTER objective_c_cc_srcs INCLUDE REGEX "\\.mm?$")
  set_property(SOURCE ${objective_c_cc_srcs} APPEND PROPERTY COMPILE_OPTIONS "-fobjc-arc")
endfunction()

function(onnxruntime_configure_target target_name)
  target_link_directories(${target_name} PRIVATE ${onnxruntime_LINK_DIRS})
  onnxruntime_set_compile_flags(${target_name})
  onnxruntime_set_source_file_properties(${target_name})
  if(WIN32 AND onnxruntime_ENABLE_STATIC_ANALYSIS AND onnxruntime_USE_CUSTOM_STATIC_ANALYSIS_RULES)
    set_target_properties(${target_name} PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
  endif()
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${abseil_cpp_SOURCE_DIR})
  if (onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(${target_name} PRIVATE ${ORTTRAINING_ROOT})
  endif()
  if (onnxruntime_ENABLE_LTO)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
  endif()

  if (onnxruntime_BUILD_KERNEL_EXPLORER)
    get_target_property(target_type ${target_name} TYPE)
    if (target_type STREQUAL "MODULE_LIBRARY" OR target_type STREQUAL "SHARED_LIBRARY")
      set_property(TARGET ${target_name}
        APPEND_STRING PROPERTY LINK_FLAGS " -Xlinker --version-script=${ONNXRUNTIME_ROOT}/python/tools/kernel_explorer/version_script.lds ")
    endif()
  endif()

  # Keep BinSkim happy
  if(MSVC AND NOT onnxruntime_target_platform MATCHES "ARM")
    target_link_options(${target_name} PRIVATE "/CETCOMPAT")
  endif()

endfunction()

function(onnxruntime_add_shared_library target_name)
  add_library(${target_name} SHARED ${ARGN})
  onnxruntime_configure_target(${target_name})
endfunction()

function(onnxruntime_add_static_library target_name)
  add_library(${target_name} STATIC ${ARGN})
  onnxruntime_configure_target(${target_name})
endfunction()

#For plugins that are not linked into other targets but may be loaded dynamically at runtime using dlopen-like functionality.
function(onnxruntime_add_shared_library_module target_name)
  if ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin") OR (${CMAKE_SYSTEM_NAME} MATCHES "iOS"))
    add_library(${target_name} SHARED ${ARGN})
  else()
    #On Windows, this target shouldn't generate an import lib, but I don't know how to disable it.
    add_library(${target_name} MODULE ${ARGN})
  endif()

  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_object_library target_name)
  add_library(${target_name} OBJECT ${ARGN})

  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_executable target_name)
  add_executable(${target_name} ${ARGN})
  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
      if(TARGET ${src_target})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
        target_sources(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_SOURCES>)
      endif()
    endforeach()
endfunction()

# ACL
if (onnxruntime_USE_ACL OR onnxruntime_USE_ACL_1902 OR onnxruntime_USE_ACL_1905 OR onnxruntime_USE_ACL_1908 OR onnxruntime_USE_ACL_2002 OR onnxruntime_USE_ACL_2308)
  set(onnxruntime_USE_ACL ON)
  if (onnxruntime_USE_ACL_1902)
    add_definitions(-DACL_1902=1)
  else()
    if (onnxruntime_USE_ACL_1908)
      add_definitions(-DACL_1908=1)
    else()
      if (onnxruntime_USE_ACL_2002)
        add_definitions(-DACL_2002=1)
      else()
	if (onnxruntime_USE_ACL_2308)
	  add_definitions(-DACL_2308=1)
	else()
          add_definitions(-DACL_1905=1)
	endif()
      endif()
    endif()
  endif()

  if (NOT ${onnxruntime_ACL_LIBS} STREQUAL "")
    add_library(arm_compute SHARED IMPORTED)
    set_target_properties(arm_compute PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute.so")

    add_library(arm_compute_core SHARED IMPORTED)
    set_target_properties(arm_compute_core PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_core.so")

    add_library(arm_compute_graph SHARED IMPORTED)
    set_target_properties(arm_compute_graph PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_graph.so")
  endif()

  list(APPEND onnxruntime_EXTERNAL_LIBRARIES arm_compute arm_compute_core arm_compute_graph)

endif()

# ArmNN
if (onnxruntime_USE_ARMNN)
  if (NOT onnxruntime_ARMNN_RELU_USE_CPU)
    add_definitions(-DRELU_ARMNN=1)
  endif()
  if (NOT onnxruntime_ARMNN_BN_USE_CPU)
    add_definitions(-DBN_ARMNN=1)
  endif()

  if (NOT onnxruntime_USE_ACL AND NOT ${onnxruntime_ACL_LIBS} STREQUAL "")
    add_library(arm_compute SHARED IMPORTED)
    set_target_properties(arm_compute PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute.so")

    add_library(arm_compute_core SHARED IMPORTED)
    set_target_properties(arm_compute_core PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_core.so")

    add_library(arm_compute_graph SHARED IMPORTED)
    set_target_properties(arm_compute_graph PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_graph.so")
  endif()

  if (NOT ${onnxruntime_ARMNN_LIBS} STREQUAL "")
    add_library(armnn SHARED IMPORTED)
    set_target_properties(armnn PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ARMNN_LIBS}/libarmnn.so")
  endif()

  list(APPEND onnxruntime_EXTERNAL_LIBRARIES armnn arm_compute arm_compute_core arm_compute_graph)
endif()

if (onnxruntime_USE_DNNL)
  include(dnnl)
  add_compile_definitions(DNNL_OPENMP)
endif()

if (onnxruntime_USE_NEURAL_SPEED AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_USE_TVM)
  include(neural_speed)
  if (USE_NEURAL_SPEED)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES neural_speed::bestla)
  endif()
endif()

# TVM EP
if (onnxruntime_USE_TVM)
  if (NOT TARGET tvm)
    message(STATUS "Include TVM(*).")
    include(tvm)
  endif()

  # ipp-crypto
  if (onnxruntime_TVM_USE_HASH)
    message(STATUS "Include ipp-crypto(*).")
    include(ipp-crypto)
  endif()

  # TVM
  if (onnxruntime_TVM_USE_LLVM)
    set(USE_LLVM "${onnxruntime_TVM_USE_LLVM}" CACHE STRING "Path to LLVM for correct TVM build")
  elseif(onnxruntime_USE_LLVM)
    set(USE_LLVM ON CACHE BOOL "Only defined for TVM")
  endif()

  if (onnxruntime_TVM_CUDA_RUNTIME)
    set(USE_CUDA ON CACHE BOOL "Only defined for TVM" FORCE)
  endif()

  # TODO(vvchernov): customized tvm logger is hidden due to the issue on TVM side (https://github.com/apache/tvm/issues/10139)
  # add_compile_definitions(TVM_LOG_CUSTOMIZE=1)
  # add_library(tvm_custom_logger STATIC ${ONNXRUNTIME_ROOT}/core/providers/tvm/custom_logging.cc)

  set(USE_OPENMP gnu CACHE STRING "Only defined for TVM")
  add_subdirectory(${tvm_SOURCE_DIR} ${tvm_BINARY_DIR} EXCLUDE_FROM_ALL)

  set_target_properties(tvm PROPERTIES FOLDER ${tvm_SOURCE_DIR})
  # target_link_libraries(tvm PUBLIC tvm_custom_logger)

  set(TVM_INCLUDES ${tvm_SOURCE_DIR}/include
    ${tvm_SOURCE_DIR}/3rdparty/dmlc-core/include
    ${tvm_SOURCE_DIR}/3rdparty/dlpack/include
    $<TARGET_PROPERTY:tvm,INTERFACE_INCLUDE_DIRECTORIES>)

  set(onnxruntime_tvm_libs onnxruntime_providers_tvm)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES tvm)
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES tvm)
endif()

# needs to link with stdc++fs in Linux
if (UNIX AND "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 9)
  set(FS_STDLIB stdc++fs)
endif()
list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${FS_STDLIB})

# onnxruntime-extensions
if (onnxruntime_USE_EXTENSIONS)
  include(extensions)
endif()

#Dependencies end. In the next we'll enable "treat warning as error"

#Adjust warning flags
set_msvc_c_cpp_compiler_warning_level(4)

set(onnxruntime_DELAYLOAD_FLAGS "")

include_directories(
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${REPO_ROOT}/include/onnxruntime/core/session
)

if (onnxruntime_ENABLE_TRAINING_APIS)
  include_directories(
    ${REPO_ROOT}/orttraining/orttraining/training_api/include/
  )
endif()

if (onnxruntime_USE_OPENVINO)

  add_definitions(-DUSE_OPENVINO=1)

  if (onnxruntime_USE_OPENVINO_GPU_FP32)
    add_definitions(-DOPENVINO_CONFIG_GPU_FP32=1)
  endif()

  if (onnxruntime_USE_OPENVINO_GPU_FP16)
    add_definitions(-DOPENVINO_CONFIG_GPU_FP16=1)
  endif()

  if (onnxruntime_USE_OPENVINO_CPU_FP32)
    add_definitions(-DOPENVINO_CONFIG_CPU_FP32=1)
  endif()

  if (onnxruntime_USE_OPENVINO_CPU_FP16)
    add_definitions(-DOPENVINO_CONFIG_CPU_FP16=1)
  endif()

  if (onnxruntime_USE_OPENVINO_NPU)
    add_definitions(-DOPENVINO_CONFIG_NPU=1)
  endif()

  if (onnxruntime_USE_OPENVINO_GPU_FP32_NP)
    add_definitions(-DOPENVINO_CONFIG_GPU_FP32=1)
    add_definitions(-DOPENVINO_DISABLE_GRAPH_PARTITION=1)
  endif()

  if (onnxruntime_USE_OPENVINO_GPU_FP16_NP)
    add_definitions(-DOPENVINO_CONFIG_GPU_FP16=1)
    add_definitions(-DOPENVINO_DISABLE_GRAPH_PARTITION=1)
  endif()

  if (onnxruntime_USE_OPENVINO_CPU_FP32_NP)
    add_definitions(-DOPENVINO_CONFIG_CPU_FP32=1)
    add_definitions(-DOPENVINO_DISABLE_GRAPH_PARTITION=1)
  endif()

  if (onnxruntime_USE_OPENVINO_CPU_FP16_NP)
    add_definitions(-DOPENVINO_CONFIG_CPU_FP16=1)
    add_definitions(-DOPENVINO_DISABLE_GRAPH_PARTITION=1)
  endif()

  if (onnxruntime_USE_OPENVINO_NPU_NP)
    add_definitions(-DOPENVINO_CONFIG_NPU=1)
    add_definitions(-DOPENVINO_DISABLE_GRAPH_PARTITION=1)
  endif()

  if (onnxruntime_USE_OPENVINO_HETERO)
    add_definitions(-DOPENVINO_CONFIG_HETERO=1)
    add_definitions(-DDEVICE_NAME="${onnxruntime_USE_OPENVINO_DEVICE}")
  endif()

  if (onnxruntime_USE_OPENVINO_MULTI)
    add_definitions(-DOPENVINO_CONFIG_MULTI=1)
    add_definitions(-DDEVICE_NAME="${onnxruntime_USE_OPENVINO_DEVICE}")
  endif()

  if(onnxruntime_USE_OPENVINO_AUTO)
    add_definitions(-DOPENVINO_CONFIG_AUTO=1)
    add_definitions(-DDEVICE_NAME="${onnxruntime_USE_OPENVINO_DEVICE}")
  endif()

  if($ENV{FIL_ENABLED})
    add_definitions(-DOPENVINO_FIL_ENABLED=1)
  endif()

endif()

if (onnxruntime_USE_VITISAI)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_LIST_DIR}")
endif()

set(ORT_BUILD_INFO "ORT Build Info: ")
find_package(Git)
if (Git_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE ORT_GIT_COMMIT)
  string(STRIP "${ORT_GIT_COMMIT}" ORT_GIT_COMMIT)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE ORT_GIT_BRANCH)
  string(STRIP "${ORT_GIT_BRANCH}" ORT_GIT_BRANCH)
  string(APPEND ORT_BUILD_INFO "git-branch=${ORT_GIT_BRANCH}, git-commit-id=${ORT_GIT_COMMIT}, ")
endif()
string(APPEND ORT_BUILD_INFO "build type=${CMAKE_BUILD_TYPE}")
string(APPEND ORT_BUILD_INFO ", cmake cxx flags: ${CMAKE_CXX_FLAGS}")
configure_file(onnxruntime_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_config.h)
get_property(onnxruntime_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (onnxruntime_GENERATOR_IS_MULTI_CONFIG)
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/Debug/requirements.txt)
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/Release/requirements.txt)
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo/requirements.txt)
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/MinSizeRel/requirements.txt)
else()
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt)
endif()

if (onnxruntime_USE_CUDA)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  set(CMAKE_CUDA_STANDARD 17)
  if(onnxruntime_CUDA_HOME)
    file(TO_CMAKE_PATH CUDAToolkit_ROOT ${onnxruntime_CUDA_HOME})
  endif()
  find_package(CUDAToolkit REQUIRED)
  if(onnxruntime_CUDNN_HOME)
    file(TO_CMAKE_PATH ${onnxruntime_CUDNN_HOME} onnxruntime_CUDNN_HOME)
  endif()
  if (NOT CMAKE_CUDA_ARCHITECTURES)
    if (CMAKE_LIBRARY_ARCHITECTURE STREQUAL "aarch64-linux-gnu")
      # Support for Jetson/Tegra ARM devices
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_53,code=sm_53") # TX1, Nano
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_62,code=sm_62") # TX2
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_72,code=sm_72") # AGX Xavier, NX Xavier
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_87,code=sm_87") # AGX Orin, NX Orin
      endif()
    else()
      # the following compute capabilities are removed in CUDA 11 Toolkit
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=sm_30") # K series
      endif()
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12)
        # 37, 50 still work in CUDA 11 but are marked deprecated and will be removed in future CUDA version.
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_37,code=sm_37") # K80
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
      endif()
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52") # M60
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75") # T series
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80") # A series
      endif()
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90") # H series
      endif()
    endif()
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror default-stream-launch")
  endif()
  if (NOT WIN32)
    list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
  endif()
  # Options passed to cudafe
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=bad_friend_decl\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=expr_has_no_effect\"")

  if (onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>")
  endif()
endif()

if (onnxruntime_USE_TENSORRT)
  # needs to link with stdc++fs in Linux
  if (UNIX)
    if (NOT APPLE)
      set(FS_STDLIB stdc++fs)
    endif()
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${FS_STDLIB})
endif()

if (onnxruntime_USE_MIGRAPHX)
  if (WIN32)
    message(FATAL_ERROR "MIGraphX does not support build in Windows!")
  endif()
  set(AMD_MIGRAPHX_HOME ${onnxruntime_MIGRAPHX_HOME})
endif()

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  add_definitions(-DMICROSOFT_INTERNAL)
endif()

if (onnxruntime_USE_DML)
  if (NOT WIN32)
    message(FATAL_ERROR "The DirectML execution provider is only supported when building for Windows.")
  endif()

  include(dml)
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
  add_compile_definitions(ENABLE_TRAINING_CORE)
  add_compile_definitions(ENABLE_TRAINING_APIS)
endif()

if (onnxruntime_ENABLE_TRAINING_OPS)
  add_compile_definitions(ENABLE_TRAINING_OPS)
endif()

if (onnxruntime_ENABLE_CUDA_PROFILING)
  add_compile_definitions(ENABLE_CUDA_PROFILING)
endif()

if (onnxruntime_ENABLE_ROCM_PROFILING)
  add_compile_definitions(ENABLE_ROCM_PROFILING)
endif()

if (onnxruntime_ENABLE_TRAINING)
  add_compile_definitions(ENABLE_TRAINING_CORE)
  add_compile_definitions(ENABLE_STRIDED_TENSORS)
  add_compile_definitions(ENABLE_TRAINING)

  add_subdirectory(tensorboard EXCLUDE_FROM_ALL)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES tensorboard)
endif()

if (UNIX AND onnxruntime_USE_NCCL)
  # MPI is INDEPENDENT of NCCL for now. You can build NCLL without MPI and launch multi-GPU with your own launcher.
  if (onnxruntime_USE_MPI)
    if (EXISTS "${onnxruntime_MPI_HOME}")
      set(MPI_HOME "${onnxruntime_MPI_HOME}")
    elseif (EXISTS "/bert_ort/openmpi")
      set(MPI_HOME "/bert_ort/openmpi")
    endif()
    find_package(MPI)

    if (MPI_CXX_FOUND)
      message( STATUS "MPI Version: ${MPI_CXX_VERSION}")
      message( STATUS "MPI (include: ${MPI_CXX_INCLUDE_DIRS}, library: ${MPI_CXX_LIBRARIES})" )
      mark_as_advanced(MPI_CXX_INCLUDE_DIRS MPI_CXX_LIBRARIES)
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    else ()
      message(
            FATAL_ERROR
            "MPI is not found. Please define onnxruntime_MPI_HOME to specify the path of MPI. Otherwise, NCCL will be disabled."
            "or you can remove --use_mpi from build args to disable MPI."
          )
    endif()
  endif()

  # Find NCCL
  if (onnxruntime_USE_NCCL)
    if (onnxruntime_USE_CUDA)
      set(NCCL_LIBNAME "nccl")
    elseif (onnxruntime_USE_ROCM)
      set(NCCL_LIBNAME "rccl")
    endif()
    find_path(NCCL_INCLUDE_DIR
      NAMES ${NCCL_LIBNAME}.h
      HINTS
      ${onnxruntime_NCCL_HOME}/include/rccl
      ${onnxruntime_NCCL_HOME}/include
      $ENV{CUDA_ROOT}/include)

    find_library(NCCL_LIBRARY
      NAMES ${NCCL_LIBNAME}
      HINTS
      ${onnxruntime_NCCL_HOME}/lib/x86_64-linux-gnu
      ${onnxruntime_NCCL_HOME}/lib
      $ENV{CUDA_ROOT}/lib64)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

    if (NCCL_FOUND)
      set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIR}/${NCCL_LIBNAME}.h")
      message( STATUS "Determining NCCL version from the header file: ${NCCL_HEADER_FILE}" )
      file (STRINGS ${NCCL_HEADER_FILE} NCCL_MAJOR_VERSION_DEFINED
            REGEX "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
      if (NCCL_MAJOR_VERSION_DEFINED)
        string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+" ""
                NCCL_MAJOR_VERSION ${NCCL_MAJOR_VERSION_DEFINED})
        message( STATUS "NCCL_MAJOR_VERSION: ${NCCL_MAJOR_VERSION}" )
      endif()
      file (STRINGS ${NCCL_HEADER_FILE} NCCL_MINOR_VERSION_DEFINED
            REGEX "^[ \t]*#define[ \t]+NCCL_MINOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
      if (NCCL_MINOR_VERSION_DEFINED)
        string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MINOR[ \t]+" ""
                NCCL_MINOR_VERSION ${NCCL_MINOR_VERSION_DEFINED})
        message(STATUS "NCCL_MINOR_VERSION: ${NCCL_MINOR_VERSION}")
      endif()

      if (NCCL_MAJOR_VERSION_DEFINED AND NCCL_MINOR_VERSION_DEFINED)
        if ("${NCCL_MAJOR_VERSION}.${NCCL_MINOR_VERSION}" VERSION_GREATER_EQUAL "2.7")
          add_definitions(-DUSE_NCCL_P2P=1)
          message( STATUS "NCCL P2P is enabled for supporting ncclSend and ncclRecv." )
        endif()
      endif()

      set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
      set(NCCL_LIBRARIES ${NCCL_LIBRARY})
      message( STATUS "NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})" )
      mark_as_advanced(NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

      list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${NCCL_LIBRARIES})


      add_definitions(-DORT_USE_NCCL=1)
      message( STATUS "NCCL is enabled in Linux GPU Build." )
    else ()
      set(onnxruntime_USE_NCCL OFF)
      message(
          FATAL_ERROR
          "NCCL is not found. Please use --nccl_home to specify the path of NCCL. Otherwise, NCCL is disabled."
        )
    endif()
  endif()
else()
  set(onnxruntime_USE_NCCL OFF)
  set(onnxruntime_USE_MPI OFF)
  message( WARNING "MPI and NCCL are disabled because build is on Windows or USE_NCCL is set to OFF." )
endif()

if (onnxruntime_USE_MPI)
  add_definitions(-DUSE_MPI=1)
endif()

# Default version parts for Microsoft.AI.MachineLearning.dll, onnxruntime.dll, onnxruntime_providers_openvino.dll and onnxruntime_providers_shared.dll in non-ADO pipeline local builds
set(VERSION_MAJOR_PART   0 CACHE STRING "First part of numeric file/product version.")
set(VERSION_MINOR_PART   0 CACHE STRING "Second part of numeric file/product version.")
set(VERSION_BUILD_PART       0 CACHE STRING "Third part of numeric file/product version.")
set(VERSION_PRIVATE_PART     0 CACHE STRING "Fourth part of numeric file/product version.")
set(VERSION_STRING       "Internal Build" CACHE STRING "String representation of file/product version.")

if (WIN32)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${SYS_PATH_LIB})
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES debug Dbghelp)
  # In a onecore build the umbrella libs already contains references to the APIs in advapi32, so in onecore build we do not need to link to advapi32
  # In a non-onecore build, usually we also do not need to link to advapi32 because VC++ by default should have provide everything we need, except when the build target is Windows ARM32.
  # In the future we will add a build option to allow users disabling all API uses from advapi32 because some Windows environments do not have these APIs. For example, some Windows do not have
  # Windows Registry so we cannot query Registry values.
  if(onnxruntime_target_platform STREQUAL "ARM" AND CMAKE_CXX_STANDARD_LIBRARIES MATCHES kernel32.lib)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES advapi32)
  endif()
else()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES nsync::nsync_cpp)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${ICONV_LIB} ${CMAKE_DL_LIBS} Threads::Threads)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES log)
endif()

# check if we need to link against librt on Linux
include(CheckLibraryExists)
include(CheckFunctionExists)
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)

  if (NOT HAVE_CLOCK_GETTIME)
    set(CMAKE_EXTRA_INCLUDE_FILES time.h)
    check_function_exists(clock_gettime HAVE_CLOCK_GETTIME)
    set(CMAKE_EXTRA_INCLUDE_FILES)
  else()
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES rt)
  endif()
endif()


#Now the 'onnxruntime_EXTERNAL_LIBRARIES' variable should be sealed. It will be used in onnxruntime.cmake which will be included in the next.
#The order of the following targets matters. Right depends on left. If target A appears before target B. Then A.cmake can not use variables defined in B.cmake.
set(ONNXRUNTIME_CMAKE_FILES onnxruntime_flatbuffers onnxruntime_common onnxruntime_mlas onnxruntime_graph onnxruntime_framework onnxruntime_util onnxruntime_providers onnxruntime_optimizer onnxruntime_session ${ONNXRUNTIME_EAGER_CMAKE_FILE_NAME})

if (onnxruntime_USE_WINML)
  # WINML uses and depends on the shared lib.  Note:  You can build WINML without DML and you will get a
  # CPU only WINML
  if (NOT onnxruntime_BUILD_SHARED_LIB)
    message(
        FATAL_ERROR
        "Option onnxruntime_USE_WINML can only be used when onnxruntime_BUILD_SHARED_LIB is also enabled")
  endif()
  list(APPEND ONNXRUNTIME_CMAKE_FILES winml)
endif() # if (onnxruntime_USE_WINML)

if (onnxruntime_BUILD_SHARED_LIB OR onnxruntime_BUILD_APPLE_FRAMEWORK)
  if (onnxruntime_BUILD_APPLE_FRAMEWORK AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS")
    message(FATAL_ERROR "onnxruntime_BUILD_APPLE_FRAMEWORK can only be enabled for macOS or iOS.")
  endif()
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime)
endif()

if (onnxruntime_BUILD_JAVA)
  message(STATUS "Java Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_java)
endif()

if (onnxruntime_BUILD_NODEJS)
  message(STATUS "Node.js Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_nodejs)
endif()

if (onnxruntime_ENABLE_PYTHON)
  message(STATUS "Python Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_python)
endif()

if (onnxruntime_BUILD_OBJC)
  message(STATUS "Objective-C Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_objectivec)
endif()

if (onnxruntime_BUILD_UNIT_TESTS)
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_unittests)
endif()

if (onnxruntime_BUILD_WINML_TESTS)
  list(APPEND ONNXRUNTIME_CMAKE_FILES winml_unittests)
endif()

# onnxruntime_training depends on onnxruntime_unittests since onnxruntime_training.cmake uses a variable `TEST_SRC_DIR`
# that is defined in onnxruntime_unittests.cmake
if (onnxruntime_ENABLE_TRAINING)
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_training)
  if (onnxruntime_ENABLE_TRAINING_E2E_TESTS)
    list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_training_e2e_tests)
  endif()
endif()

if (onnxruntime_BUILD_CSHARP)
  message(STATUS "CSharp Build is enabled")
#  set_property(GLOBAL PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "netstandard2.0")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_csharp)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  message(STATUS "WebAssembly Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_webassembly)

  if (onnxruntime_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL)
    add_compile_definitions(ORT_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL)
  endif()
endif()

if(onnxruntime_BUILD_KERNEL_EXPLORER)
  message(STATUS "Kernel Explorer Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_kernel_explorer)
endif()

# When GDK_PLATFORM is set then WINAPI_FAMILY is defined in gdk_toolchain.cmake (along with other relevant flags/definitions).
if (WIN32 AND NOT GDK_PLATFORM AND NOT CMAKE_CROSSCOMPILING)
  if (NOT CMAKE_CXX_STANDARD_LIBRARIES MATCHES kernel32.lib)
    # On onecore, link to the onecore build of the MSVC runtime
    get_filename_component(msvc_path "${CMAKE_C_COMPILER}/../../../.." ABSOLUTE)
    link_directories(BEFORE "${msvc_path}/lib/onecore/${onnxruntime_target_platform}")
    # The .lib files in the MSVC runtime have a DEFAULITLIB entry for onecore.lib, but it shold not cause any conflict with onecoreuap.lib
  endif()
endif()

foreach(target_name ${ONNXRUNTIME_CMAKE_FILES})
  include(${target_name}.cmake)
endforeach()
if (UNIX)
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" ON)
else()
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" OFF)
endif()
if (BUILD_PKGCONFIG_FILES)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.pc.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc @ONLY)
  install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/pkgconfig )
endif()

if (onnxruntime_BUILD_OPSCHEMA_LIB AND onnxruntime_ENABLE_TRAINING)
  # opschema library requires training ops as well
  include(onnxruntime_opschema_lib.cmake)
endif()

if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  add_compile_definitions(DEBUG_NODE_INPUTS_OUTPUTS)
endif()

if (onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS)
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "External custom operator schemas feature is only supported on Linux")
  endif()

  if (NOT ${ONNX_CUSTOM_PROTOC_EXECUTABLE} STREQUAL "")
    message(FATAL_ERROR "External custom operator schemas is not supported with the user specified protoc executable")
  endif()

  if (NOT onnxruntime_ENABLE_TRAINING)
    message(FATAL_ERROR "External custom operator schemas is supported only with --enable-training option")
  endif()

  add_custom_target(install_protobuf ALL DEPENDS ${PROTOBUF_LIB} protobuf::protoc)
  add_custom_command(
    TARGET install_protobuf
    COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR} -P cmake_install.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/protobuf/cmake
    COMMENT "Installing protobuf"
  )
endif()

if(TARGET onnxruntime)
# Install
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  set(PROJECT_CONFIG_CONTENT "@PACKAGE_INIT@\n")
  string(APPEND PROJECT_CONFIG_CONTENT
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Targets.cmake\")")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/PROJECT_CONFIG_FILE" ${PROJECT_CONFIG_CONTENT})
  install(EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
# Create config for find_package()
  configure_package_config_file(
    "${CMAKE_CURRENT_BINARY_DIR}/PROJECT_CONFIG_FILE" ${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${ORT_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(
    FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
endif()

if(DEFINED BUILD_AS_ARM64X)
  set(ARM64X_TARGETS onnxruntime)
  include("${CMAKE_SOURCE_DIR}/arm64x.cmake")
endif()
