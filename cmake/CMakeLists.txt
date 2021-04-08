
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.15")
  cmake_policy(SET CMP0092 NEW)
endif()

# Support OS X versions 10.12+
# This variable is ignored on non-Apple platforms and needs to be set prior to the first project(...) invocation
# TODO: Make the miniumum deployment target MacOSX version configurable
# If the CMAKE_OSX_DEPLOYMENT_TARGET was set (currently valid only for iOS build),
# we will cache its value instead of the default "10.12"
if (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum OS X deployment version for ORT" FORCE)
else()
  set(CMAKE_OSX_DEPLOYMENT_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET} CACHE STRING "Minimum OS X deployment version for ORT" FORCE)
endif()

# Project
project(onnxruntime C CXX)
# Needed for Java
set (CMAKE_C_STANDARD 99)

include(CheckCXXCompilerFlag)
include(CheckLanguage)

# CentOS compiler is old but it does allow certain C++14 features
# such as lambda captures and they are convenient
# On the other hand it does not allow some others.
# So we cant' regulate simply with the standard.
set(CMAKE_CXX_STANDARD 14)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# NOTE: POSITION INDEPENDENT CODE hurts performance, and it only make sense on POSIX systems
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable CTest
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Options
option(onnxruntime_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)
option(onnxruntime_GENERATE_TEST_REPORTS "Enable test report generation" OFF)
option(onnxruntime_ENABLE_STATIC_ANALYSIS "Enable static analysis" OFF)
option(onnxruntime_ENABLE_PYTHON "Enable python buildings" OFF)
# Enable it may cause LNK1169 error
option(onnxruntime_ENABLE_MEMLEAK_CHECKER "Experimental: Enable memory leak checker in Windows debug build" OFF)
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_EIGEN_FOR_BLAS "Use eign for blas" ON)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_RKNPU "Build with RKNPU support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_MKLML "Build the default cpu provider with MKL-ML binary dependency" OFF)
option(onnxruntime_USE_FEATURIZERS "Build ML Featurizers support" OFF)
option(onnxruntime_USE_NGRAPH "Build with nGraph support" OFF)
option(onnxruntime_USE_OPENBLAS "Use openblas" OFF)
option(onnxruntime_DEV_MODE "Enable developer warnings and treat most of them as error." OFF)
option(onnxruntime_USE_JEMALLOC "Use jemalloc" OFF)
option(onnxruntime_MSVC_STATIC_RUNTIME "Compile for the static CRT" OFF)
option(onnxruntime_GCC_STATIC_CPP_RUNTIME "Compile for the static libstdc++" OFF)
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" ON)
option(onnxruntime_BUILD_CSHARP "Build C# library" OFF)
option(onnxruntime_USE_PREINSTALLED_EIGEN "Use pre-installed EIGEN. Need to provide eigen_SOURCE_PATH if turn this on." OFF)
option(onnxruntime_BUILD_BENCHMARKS "Build ONNXRuntime micro-benchmarks" OFF)
option(onnxruntime_USE_TVM "Build tvm for code-gen" OFF)
option(onnxruntime_BUILD_FOR_NATIVE_MACHINE "Enable this option for turning on optimization specific to this machine" OFF)
option(onnxruntime_USE_AVX "Use AVX instructions" OFF)
option(onnxruntime_USE_AVX2 "Use AVX2 instructions" OFF)
option(onnxruntime_USE_AVX512 "Use AVX512 instructions" OFF)
option(onnxruntime_USE_LLVM "Build tvm with LLVM" OFF)
option(onnxruntime_USE_OPENMP "Build with OpenMP support" OFF)
option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" OFF)
option(onnxruntime_ENABLE_MICROSOFT_INTERNAL "Use this option to enable/disable microsoft internal only code" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_VITISAI "Build with Vitis-AI" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(onnxruntime_ENABLE_LTO "Enable link time optimization" OFF)
option(onnxruntime_CROSS_COMPILING "Cross compiling onnx runtime" OFF)

#It's preferred to turn it OFF when onnxruntime is dynamically linked to PROTOBUF
option(onnxruntime_USE_FULL_PROTOBUF "Link to libprotobuf instead of libprotobuf-lite when this option is ON" OFF)
option(tensorflow_C_PACKAGE_PATH "Path to tensorflow C package installation dir")
option(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS "Enable operator implemented in language other than cpp" OFF)
option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS "Dump debug information about node inputs and outputs when executing the model." OFF)
option(onnxruntime_USE_DML "Build with DirectML support" OFF)
option(onnxruntime_USE_MIGRAPHX "Build with AMDMIGraphX support" OFF)
option(onnxruntime_USE_WINML "Build with WinML support" OFF)
option(onnxruntime_USE_ACL "Build with ACL support" OFF)
option(onnxruntime_USE_ACL_1902 "Build with ACL version 1902 support" OFF)
option(onnxruntime_USE_ACL_1905 "Build with ACL version 1905 support" OFF)
option(onnxruntime_USE_ACL_1908 "Build with ACL version 1908 support" OFF)
option(onnxruntime_USE_ACL_2002 "Build with ACL version 2002 support" OFF)
option(onnxruntime_USE_ARMNN "Build with ArmNN support" OFF)
option(onnxruntime_ARMNN_RELU_USE_CPU "Use the CPU implementation for the Relu operator for the ArmNN EP" ON)
option(onnxruntime_ARMNN_BN_USE_CPU "Use the CPU implementation for the Batch Normalization operator for the ArmNN EP" ON)
option(onnxruntime_ENABLE_INSTRUMENT "Enable Instrument with Event Tracing for Windows (ETW)" OFF)
option(onnxruntime_USE_TELEMETRY "Build with Telemetry" OFF)
#The onnxruntime_PREFER_SYSTEM_LIB is mainly designed for package managers like apt/yum/vcpkg.
#Please note, by default Protobuf_USE_STATIC_LIBS is OFF but it's recommended to turn it ON on Windows. You should set it properly when onnxruntime_PREFER_SYSTEM_LIB is ON otherwise you'll hit linkage errors.
#If you have already installed protobuf(or the others) in your system at the default system paths(like /usr/include), then it's better to set onnxruntime_PREFER_SYSTEM_LIB ON. Otherwise onnxruntime may see two different protobuf versions and we won't know which one will be used, the worst case could be onnxruntime picked up header files from one of them but the binaries from the other one.
#It's default OFF because it's experimental now.
option(onnxruntime_PREFER_SYSTEM_LIB "Experimental: Build with the preinstalled libraries in your system" OFF)

# Options related to reducing the binary size produced by the build
option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" OFF)
option(onnxruntime_DISABLE_ML_OPS "Disable traditional ML ops" OFF)
option(onnxruntime_DISABLE_RTTI "Disable RTTI" OFF)
# For now onnxruntime_DISABLE_EXCEPTIONS will only work with onnxruntime_MINIMAL_BUILD, more changes (ONNX, non-CPU EP, ...) are required to run this standalone
option(onnxruntime_DISABLE_EXCEPTIONS "Disable exception handling. Requires onnxruntime_MINIMAL_BUILD currently." OFF)
option(onnxruntime_MINIMAL_BUILD "Exclude as much as possible from the build. Support ORT format models. No support for ONNX format models." OFF)
option(onnxruntime_REDUCED_OPS_BUILD "Reduced set of kernels are registered in build via modification of the kernel registration source files." OFF)
option(onnxruntime_DISABLE_ORT_FORMAT_LOAD "Disable loading an ORT format model when onnxruntime_MINIMAL_BUILD=OFF (i.e. in a full build)." OFF)

# training options
option(onnxruntime_ENABLE_NVTX_PROFILE "Enable NVTX profile." OFF)
option(onnxruntime_ENABLE_TRAINING "Enable training functionality." OFF)
option(onnxruntime_ENABLE_TRAINING_E2E_TESTS "Enable training end-to-end tests." OFF)
option(onnxruntime_USE_HOROVOD "Build with HOROVOD support" OFF)
option(onnxruntime_USE_NCCL "Build with NCCL support" ON)

# LIBTORCH_HOME can be either a path to libtorch downloaded from Pytorch website,
# Pytorch python package installed using Anaconda, Pytorch build folder if you built
# Pytorch from source, and so on.
# This D_GLIBCXX_USE_CXX11_ABI flag should match what is used when compiling Pytorch/LibTorch and
# all other ORT dependencies.
# Do not build using multi-threading when ABI is set. Some packages have their
# own logics to determine this flag and seems resulting race conditions.
if (onnxruntime_USE_TORCH)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -DUSE_TORCH=1 -DTHRUST_IGNORE_CUB_VERSION_CHECK)
  list(APPEND CMAKE_PREFIX_PATH "${onnxruntime_TORCH_HOME}")
  find_package(Torch REQUIRED PATHS "${onnxruntime_TORCH_HOME}")
endif()

# A special build option only used for gathering code coverage info
option(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE "Run model tests even in debug mode" OFF)

# options for security fuzzing
# build configuration for fuzz testing is in onnxruntime_fuzz_test.cmake
option(onnxruntime_FUZZ_TEST "Enable Fuzz testing" OFF)

# Fuzz test has only been tested with BUILD_SHARED_LIB option,
# using the MSVC compiler and on windows OS.
if(MSVC AND WIN32 AND onnxruntime_FUZZ_TEST AND onnxruntime_BUILD_SHARED_LIB AND onnxruntime_USE_FULL_PROTOBUF)
  # Fuzz test library dependency, protobuf-mutator,
  # needs the onnx message to be compiled using "non-lite protobuf version"
  set(onnxruntime_FUZZ_ENABLED ON)
endif()

if (onnxruntime_ENABLE_NVTX_PROFILE)
  add_definitions(-DENABLE_NVTX_PROFILE=1)
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
#nsync tests failed on Mac Build
set(NSYNC_ENABLE_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
set(ONNX_ML 1)
if(NOT onnxruntime_ENABLE_PYTHON)
  set(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS OFF)
endif()
if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  add_compile_definitions(ENABLE_LANGUAGE_INTEROP_OPS)
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

if(NOT WIN32)
  #TODO: On Linux we may try https://github.com/microsoft/TraceLogging
  if(onnxruntime_ENABLE_INSTRUMENT)
    message(WARNING "Instrument is only supported on Windows now")
    set(onnxruntime_ENABLE_INSTRUMENT OFF)
  endif()
else()
  if(WINDOWS_STORE)
    # cmake/external/protobuf/src/google/protobuf/compiler/subprocess.cc and onnxruntime/core/platform/windows/env.cc call a bunch of Win32 APIs.
    # For now, we'll set the API family to desktop globally to expose Win32 symbols in headers; this must be fixed!
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/FI${CMAKE_CURRENT_SOURCE_DIR}/set_winapi_family_desktop.h>")
  endif()
endif()

if(onnxruntime_USE_OPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(${OpenMP_CXX_INCLUDE_DIR})
    # MKLML and NGraph depend on their own OpenMP library that may be different with the compiler's.
    # Disable the options to build mklml/NGraph and OpenMP together.
    if((WIN32 OR APPLE) AND onnxruntime_USE_MKLML)
      message(FATAL_ERROR "Please use only one of onnxruntime_USE_MKLML, onnxruntime_USE_OPENMP")
    endif()
    if(onnxruntime_USE_NGRAPH)
      message(FATAL_ERROR "Please use only one of onnxruntime_USE_NGRAPH, onnxruntime_USE_OPENMP")
    endif()
  else()
    message(WARNING "Flag --use_openmp is specified, but OpenMP is not found in current build environment. Setting it to OFF.")
    set(onnxruntime_USE_OPENMP OFF)
  endif()
endif()

# ORT build with as much excluded as possible. Supports ORT flatbuffers models only.
# Will expose option in build.py when all pieces are available
if(onnxruntime_MINIMAL_BUILD)
  add_compile_definitions(ORT_MINIMAL_BUILD)
  add_compile_definitions(ENABLE_ORT_FORMAT_LOAD)
  
  set(onnxruntime_REDUCED_OPS_BUILD ON)

  if (NOT onnxruntime_ENABLE_PYTHON)
    # Python bindings use typeid so can't automatically disable RTTI in that case
    set(onnxruntime_DISABLE_RTTI ON)
  endif()

  if (MSVC)
    # turn on LTO (which adds some compiler flags and turns on LTCG) unless it's a Debug build to minimize binary size
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      set (onnxruntime_ENABLE_LTO ON)
    endif()

    # undocumented internal flag to allow analysis of a minimal build binary size
    if (ADD_DEBUG_INFO_TO_MINIMAL_BUILD)
      string(APPEND CMAKE_CXX_FLAGS " /Zi")
      string(APPEND CMAKE_C_FLAGS " /Zi")
      string(APPEND CMAKE_SHARED_LINKER_FLAGS " /debug")
    endif()
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    if (CMAKE_HOST_SYSTEM MATCHES "Darwin")
      add_link_options(-Wl, -dead_strip)
    else()
      add_link_options(-Wl,--gc-sections)
    endif()

    if (ADD_DEBUG_INFO_TO_MINIMAL_BUILD)
      string(APPEND CMAKE_CXX_FLAGS " -g")
      string(APPEND CMAKE_C_FLAGS " -g")
    endif()
  endif()
else()
  # support ORT format model loading unless onnxruntime_DISABLE_ORT_FORMAT_LOAD is set
  if (NOT onnxruntime_DISABLE_ORT_FORMAT_LOAD)
    add_compile_definitions(ENABLE_ORT_FORMAT_LOAD)
  endif()
endif()

if(onnxruntime_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_enabled OUTPUT ipo_output)
    if(NOT ipo_enabled)
      message(WARNING "IPO is not supported by this compiler")
      set(onnxruntime_ENABLE_LTO OFF)
    else()
      set (CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()

if (onnxruntime_REDUCED_OPS_BUILD)
  add_compile_definitions(REDUCED_OPS_BUILD)
endif()

if(onnxruntime_DISABLE_RTTI)
  add_compile_definitions(ORT_NO_RTTI GOOGLE_PROTOBUF_NO_RTTI)
  if(MSVC)
    # Disable RTTI and turn usage of dynamic_cast and typeid into errors
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GR->" "$<$<COMPILE_LANGUAGE:CXX>:/we4541>")
  else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>")
  endif()
endif()

# If this is only enabled in an onnxruntime_ORT_MODEL_FORMAT_ONLY build we don't need ONNX changes
# as we (currently) only pull in data_type_utils.cc/h which doesn't throw
if(onnxruntime_DISABLE_EXCEPTIONS)
  if(NOT onnxruntime_MINIMAL_BUILD)
    message(FATAL_ERROR "onnxruntime_MINIMAL_BUILD required for onnxruntime_DISABLE_EXCEPTIONS")
  endif()

  add_compile_definitions("ORT_NO_EXCEPTIONS")
  add_compile_definitions("MLAS_NO_EXCEPTION")

  if(MSVC)
    string(REGEX REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    # Eigen throw_std_bad_alloc calls 'new' instead of throwing which results in a nodiscard warning.
    # It also has unreachable code as there's no good way to avoid EIGEN_EXCEPTIONS being set in macros.h
    # TODO: see if we can limit the code this is disabled for.
    string(APPEND CMAKE_CXX_FLAGS " /wd4834 /wd4702")
    add_compile_definitions("_HAS_EXCEPTIONS=0")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables")
  endif()
endif()

set(REPO_ROOT ${PROJECT_SOURCE_DIR}/..)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../onnxruntime)
set(ORTTRAINING_ROOT ${PROJECT_SOURCE_DIR}/../orttraining)
set(ORTTRAINING_SOURCE_DIR ${ORTTRAINING_ROOT}/orttraining)

file (STRINGS "${REPO_ROOT}/VERSION_NUMBER" ORT_VERSION)

# Guarantee that the Eigen code that you are #including is licensed
# under the MPL2 and possibly more permissive licenses (like BSD).
add_definitions(-DEIGEN_MPL2_ONLY)
if(MSVC)
  add_definitions(-DEIGEN_HAS_CONSTEXPR -DEIGEN_HAS_VARIADIC_TEMPLATES -DEIGEN_HAS_CXX11_MATH -DEIGEN_HAS_CXX11_ATOMIC
          -DEIGEN_STRONG_INLINE=inline)
endif()

if(onnxruntime_CROSS_COMPILING)
  set(CMAKE_CROSSCOMPILING ON)
  check_cxx_compiler_flag(-Wno-error HAS_NOERROR)
  if(HAS_NOERROR)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=attributes")
    string(APPEND CMAKE_C_FLAGS " -Wno-error=attributes")
  endif()
endif()

#must after OpenMP settings
find_package(Threads)

#Set global compile flags for all the source code(including third_party code like protobuf)
#This section must be before any add_subdirectory, otherwise build may fail because /MD,/MT mismatch
if (MSVC)
  if (onnxruntime_MSVC_STATIC_RUNTIME)
    # set all of our submodules to static runtime
    set(ONNX_USE_MSVC_STATIC_RUNTIME ON)
    set(protobuf_MSVC_STATIC_RUNTIME ON CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)

    # In case we are building static libraries, link also the runtime library statically
    # so that MSVCR*.DLL is not required at runtime.
    # https://msdn.microsoft.com/en-us/library/2kzt1wy3.aspx
    # This is achieved by replacing msvc option /MD with /MT and /MDd with /MTd
    # https://gitlab.kitware.com/cmake/community/wikis/FAQ#how-can-i-build-my-msvc-application-with-a-static-runtime
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  else()
    set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
    set(protobuf_WITH_ZLIB  OFF CACHE BOOL "" FORCE)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link protobuf to static runtime libraries" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib for gtest" FORCE)
  endif()

  #Always enable exception handling, even for Windows ARM
  if(NOT onnxruntime_DISABLE_EXCEPTIONS)
    string(APPEND CMAKE_CXX_FLAGS " /EHsc /wd26812")
    string(APPEND CMAKE_C_FLAGS " /EHsc /wd26812")
  endif()

  if(onnxruntime_USE_AVX)
    string(APPEND CMAKE_CXX_FLAGS " /arch:AVX")
    string(APPEND CMAKE_C_FLAGS " /arch:AVX")
  elseif(onnxruntime_USE_AVX2)
    string(APPEND CMAKE_CXX_FLAGS " /arch:AVX2")
    string(APPEND CMAKE_C_FLAGS " /arch:AVX2")
  elseif(onnxruntime_USE_AVX512)
    string(APPEND CMAKE_CXX_FLAGS " /arch:AVX512")
    string(APPEND CMAKE_C_FLAGS " /arch:AVX512")
  endif()

  if (onnxruntime_ENABLE_LTO AND NOT onnxruntime_USE_CUDA)
    SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Gw /GL")
    SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Gw /GL")
    SET (CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /Gw /GL")
  endif()

  # The WinML build tool chain builds ARM/ARM64, and the internal tool chain does not have folders for spectre mitigation libs.
  # WinML performs spectre mitigation differently.
  if (NOT DEFINED onnxruntime_DISABLE_QSPECTRE_CHECK)
    check_cxx_compiler_flag(-Qspectre HAS_QSPECTRE)
    if (HAS_QSPECTRE)
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Qspectre")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qspectre")
    endif()
  endif()
  SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /DYNAMICBASE")
  check_cxx_compiler_flag(-guard:cf HAS_GUARD_CF)
  if (HAS_GUARD_CF)
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /guard:cf")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /guard:cf")
    SET(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /guard:cf")
    SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /guard:cf")
    SET(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} /guard:cf")
    SET(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /guard:cf")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /guard:cf")
  endif()
else()
  if(onnxruntime_BUILD_FOR_NATIVE_MACHINE)
    string(APPEND CMAKE_CXX_FLAGS " -march=native -mtune=native")
    string(APPEND CMAKE_C_FLAGS " -march=native -mtune=native")
  elseif(onnxruntime_USE_AVX)
    string(APPEND CMAKE_CXX_FLAGS " -mavx")
    string(APPEND CMAKE_C_FLAGS " -mavx")
  elseif(onnxruntime_USE_AVX2)
    string(APPEND CMAKE_CXX_FLAGS " -mavx2")
    string(APPEND CMAKE_C_FLAGS " -mavx2")
  elseif(onnxruntime_USE_AVX512)
    string(APPEND CMAKE_CXX_FLAGS " -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")
    string(APPEND CMAKE_C_FLAGS " -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")
  endif()
  if(onnxruntime_GCC_STATIC_CPP_RUNTIME)
    string(APPEND CMAKE_CXX_FLAGS " -static-libstdc++")
  endif()
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    #For Mac compliance
    message("Adding flags for Mac builds")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "iOSCross")
    #For ios compliance
    message("Adding flags for ios builds")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target arm64-apple-darwin-macho")
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "arm")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target armv7a-apple-darwin-macho")
    endif()
endif()

#Dependencies begin
if (onnxruntime_BUILD_UNIT_TESTS)
  if(onnxruntime_PREFER_SYSTEM_LIB)
    find_package(GTest CONFIG)
  endif()
  if(NOT TARGET GTest::gtest)
    message("Use gtest from submodule")
    # gtest and gmock
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/googletest EXCLUDE_FROM_ALL)
    set_target_properties(gmock PROPERTIES FOLDER "External/GTest")
    set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
    set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
    set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")
    add_library(GTest::gmock ALIAS gmock)
    add_library(GTest::gmock_main ALIAS gmock_main)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gtest_main ALIAS gtest_main)
  endif()
endif()

set(ENABLE_DATE_TESTING  OFF CACHE BOOL "" FORCE)
set(USE_SYSTEM_TZ_DB  ON CACHE BOOL "" FORCE)
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)

if(CMAKE_CROSSCOMPILING)
  message("Doing crosscompiling")
endif()

#Need python to generate def file
if(onnxruntime_BUILD_SHARED_LIB OR onnxruntime_ENABLE_PYTHON)
  if(onnxruntime_ENABLE_PYTHON)
    find_package(PythonInterp 3.5 REQUIRED)
    find_package(PythonLibs 3.5 REQUIRED)
  else()
    find_package(PythonInterp 3.4 REQUIRED)
  endif()
endif()

#target begins from here

if(onnxruntime_BUILD_BENCHMARKS)
  if(onnxruntime_PREFER_SYSTEM_LIB)
    find_package(benchmark)
  endif()
  if(NOT TARGET benchmark::benchmark)
    message("Use benchmark from submodule")
    # We will not need to test benchmark lib itself.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
    # We will not need to install benchmark since we link it statically.
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/onnx/third_party/benchmark EXCLUDE_FROM_ALL)
    add_library(benchmark::benchmark ALIAS benchmark)
  endif()
endif()

if(NOT WIN32 AND NOT onnxruntime_PREFER_SYSTEM_LIB)
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/nsync EXCLUDE_FROM_ALL)
endif()
# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

#protobuf begin

#Here we support two build mode:
#1. if ONNX_CUSTOM_PROTOC_EXECUTABLE is set, build Protobuf from source, except protoc.exe. This mode is mainly
#   for cross-compiling
#2. if ONNX_CUSTOM_PROTOC_EXECUTABLE is not set, Compile everything(including protoc) from source code.

if(onnxruntime_PREFER_SYSTEM_LIB)
  set(protobuf_MODULE_COMPATIBLE ON)
  find_package(Protobuf)
endif()


if(Protobuf_FOUND OR Protobuf_FOUND)
  message("Use protobuf from preinstalled system lib")
  if (onnxruntime_USE_FULL_PROTOBUF)
    set(PROTOBUF_LIB protobuf::libprotobuf)
    #We have a check here but most of the cmake users don't know the Protobuf_USE_STATIC_LIBS
    # variable exists and may leave it in a wrong state.
    if(NOT Protobuf_USE_STATIC_LIBS)
      #Indeed here should be a warning, not a fatal error. ONNX Runtime itself can work in such a
      #setting but it may cause compatibility issue when ONNX Runtime is integrated with the other ONNX ecosystem softwares.
      message(FATAL_ERROR "Please enable Protobuf_USE_STATIC_LIBS")
    endif()
  else()
    set(PROTOBUF_LIB protobuf::libprotobuf-lite)
  endif()
else()
  message("Use protobuf from submodule")
  # use protobuf as a submodule
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build protobuf tests" FORCE)
  endif()

  add_subdirectory(${PROJECT_SOURCE_DIR}/external/protobuf/cmake EXCLUDE_FROM_ALL)
  if(TARGET libprotoc)
    set_target_properties(libprotoc PROPERTIES FOLDER "External/Protobuf")
    add_executable(protobuf::protoc ALIAS protoc)
  endif()
  if(TARGET protoc)
    set_target_properties(protoc PROPERTIES FOLDER "External/Protobuf")
  endif()
  if (onnxruntime_USE_FULL_PROTOBUF)
    add_library(protobuf::libprotobuf ALIAS libprotobuf)
    set(PROTOBUF_LIB libprotobuf)
  else()
    add_library(protobuf::libprotobuf ALIAS libprotobuf-lite)
    set(PROTOBUF_LIB libprotobuf-lite)
  endif()
endif()

if(UNIX AND onnxruntime_ENABLE_LTO AND NOT onnxruntime_PREFER_SYSTEM_LIB)
  #https://github.com/protocolbuffers/protobuf/issues/5923
  target_link_options(protoc PRIVATE "-Wl,--no-as-needed")
endif()

include(protobuf_function.cmake)
#protobuf end

if(onnxruntime_USE_FEATURIZERS)
  add_definitions(-DML_FEATURIZERS)
  # Fetch and build featurizers
  include(external/featurizers.cmake)
endif()

if (onnxruntime_DISABLE_CONTRIB_OPS)
  add_definitions(-DDISABLE_CONTRIB_OPS)
endif()

if (onnxruntime_DISABLE_ML_OPS)
  add_definitions(-DDISABLE_ML_OPS)
endif()

if (onnxruntime_USE_CUDA AND "${onnxruntime_CUDNN_HOME}" STREQUAL "")
  message(FATAL_ERROR "onnxruntime_CUDNN_HOME required for onnxruntime_USE_CUDA")
endif()

if (onnxruntime_USE_EIGEN_FOR_BLAS)
  add_definitions(-DUSE_EIGEN_FOR_BLAS)
endif()

if (onnxruntime_USE_OPENBLAS AND "${onnxruntime_OPENBLAS_HOME}" STREQUAL "" AND WIN32)
  # On linux we assume blas is installed via 'apt-get install libopenblas-dev'
  message(FATAL_ERROR "onnxruntime_OPENBLAS_HOME required for onnxruntime_USE_OPENBLAS")
endif()

if (onnxruntime_USE_OPENBLAS AND onnxruntime_USE_EIGEN_FOR_BLAS)
  message(FATAL_ERROR "use one of onnxruntime_USE_OPENBLAS, onnxruntime_USE_EIGEN_FOR_BLAS")
endif()

get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
get_filename_component(ORTTRAINING_ROOT "${ORTTRAINING_ROOT}" ABSOLUTE)
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_INCLUDE_DIR ${REPO_ROOT}/include/onnxruntime)

add_subdirectory(external/date EXCLUDE_FROM_ALL)

if(onnxruntime_PREFER_SYSTEM_LIB)
  find_package(re2)
endif()

set(SAFEINT_INCLUDE_DIR ${REPO_ROOT}/cmake/external/SafeInt)
add_library(safeint_interface INTERFACE)
target_include_directories(safeint_interface INTERFACE ${SAFEINT_INCLUDE_DIR})

if(NOT TARGET re2::re2)
  add_subdirectory(external/re2 EXCLUDE_FROM_ALL)
  set_target_properties(re2 PROPERTIES FOLDER "External/re2")
  add_library(re2::re2 ALIAS re2)
  set(RE2_INCLUDE_DIR ${REPO_ROOT}/cmake/external/re2)
endif()

# bounds checking behavior.
# throw instead of calling terminate if there's a bounds checking violation.
# we make it through via a handler so CUDA does not complain
# The following -DGSL macros are recognized by gsl-lite along with -Dgsl macros
# no bounds checking in release build so no perf cost
# if we enable onnxruntime_DISABLE_EXCEPTIONS, gsl will terminate
if (onnxruntime_DISABLE_EXCEPTIONS)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGSL_TERMINATE_ON_CONTRACT_VIOLATION")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGSL_THROW_ON_CONTRACT_VIOLATION")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")

include(eigen)

#onnxruntime_EXTERNAL_LIBRARIES could contain onnx, onnx_proto,libprotobuf, cuda/cudnn, jemalloc,
# dnnl/mklml, openblas, onnxruntime_codegen_tvm, tvm, nnvm_compiler and pthread
# pthread is always at the last
set(onnxruntime_EXTERNAL_LIBRARIES onnx onnx_proto ${PROTOBUF_LIB} re2::re2)

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()
endfunction()

set(onnxruntime_EXTERNAL_DEPENDENCIES onnx_proto)

# ACL
if (onnxruntime_USE_ACL OR onnxruntime_USE_ACL_1902 OR onnxruntime_USE_ACL_1905 OR onnxruntime_USE_ACL_1908 OR onnxruntime_USE_ACL_2002)
  set(onnxruntime_USE_ACL ON)
  if(onnxruntime_USE_ACL_1902)
    add_definitions(-DACL_1902=1)
  else()
    if(onnxruntime_USE_ACL_1908)
      add_definitions(-DACL_1908=1)
    else()
      if(onnxruntime_USE_ACL_2002)
        add_definitions(-DACL_2002=1)
      else()
        add_definitions(-DACL_1905=1)
      endif()
    endif()
  endif()

  list(APPEND onnxruntime_EXTERNAL_LIBRARIES arm_compute arm_compute_graph arm_compute_core)
endif()

# ArmNN
if (onnxruntime_USE_ARMNN)
  if (NOT onnxruntime_ARMNN_RELU_USE_CPU)
    add_definitions(-DRELU_ARMNN=1)
  endif()
  if (NOT onnxruntime_ARMNN_BN_USE_CPU)
    add_definitions(-DBN_ARMNN=1)
  endif()

  list(APPEND onnxruntime_EXTERNAL_LIBRARIES armnn pthread arm_compute_core arm_compute arm_compute_graph)
endif()

# MKLML
if (onnxruntime_USE_DNNL OR onnxruntime_USE_MKLML)
  include(dnnl)
endif()

# TVM
if (onnxruntime_USE_TVM)
  if (onnxruntime_USE_CUDA)
    set(USE_CUDA ON)
  endif()
  if (onnxruntime_USE_LLVM)
    set(USE_LLVM ON)
    add_definitions(-DUSE_TVM_WITH_LLVM)
  endif()
  if (onnxruntime_USE_OPENMP)
    set(USE_OPENMP "gnu")
  endif()
  if (onnxruntime_USE_MKLML)
    set(USE_OPENMP "intel")
    # make sure MKLML in ORT is used by TVM
    if (WIN32)
      set(OMP_LIBRARY ${MKLML_LIB_DIR}/${IOMP5MD_IMPORT_LIB})
    else()
      set(OMP_LIBRARY ${MKLML_LIB_DIR}/${IOMP5MD_SHARED_LIB})
    endif()
  endif()

  add_subdirectory(${PROJECT_SOURCE_DIR}/external/tvm EXCLUDE_FROM_ALL)
  set_target_properties(tvm PROPERTIES FOLDER "External/tvm")
  set_target_properties(tvm_topi PROPERTIES FOLDER "External/tvm")
  set_target_properties(tvm_runtime PROPERTIES FOLDER "External/tvm")
  set_target_properties(nnvm_compiler PROPERTIES FOLDER "External/tvm")

  if (onnxruntime_USE_MKLML)
    add_dependencies(tvm project_mklml)
    add_dependencies(tvm_topi project_mklml)
    add_dependencies(tvm_runtime project_mklml)
    add_dependencies(nnvm_compiler project_mklml)
  endif()

  set(TVM_INCLUDES ${PROJECT_SOURCE_DIR}/external/tvm/include
    ${PROJECT_SOURCE_DIR}/external/tvm/3rdparty/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/external/tvm/3rdparty/dlpack/include
    $<TARGET_PROPERTY:tvm,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:tvm_topi,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:nnvm_compiler,INTERFACE_INCLUDE_DIRECTORIES>)
  add_definitions(-DUSE_TVM)

  set(onnxruntime_tvm_libs onnxruntime_codegen_tvm)
  # needs to link with stdc++fs in Linux
  if(UNIX)
    if (NOT APPLE)
      set(FS_STDLIB stdc++fs)
    endif()
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES tvm nnvm_compiler ${FS_STDLIB})
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES tvm nnvm_compiler)
endif()

if (APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  #onnx/onnx/proto_utils.h:34:16: error: 'SetTotalBytesLimit' is deprecated: Please use the single
  #parameter version of SetTotalBytesLimit(). The second parameter is ignored.
  #  coded_stream.SetTotalBytesLimit((2048LL << 20) - 1, 512LL << 20);
  #TODO: fix the warning in ONNX and re-enable this flag
  string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated")
  string(APPEND CMAKE_C_FLAGS " -Wno-deprecated")
endif()

# ONNX
if(NOT onnxruntime_USE_FULL_PROTOBUF)
  set(ONNX_USE_LITE_PROTO ON CACHE BOOL "" FORCE)
else()
  set(ONNX_USE_LITE_PROTO OFF CACHE BOOL "" FORCE)
endif()

if (NOT onnxruntime_MINIMAL_BUILD)
  add_subdirectory(external/onnx EXCLUDE_FROM_ALL)
else()
  include(onnx_minimal)
endif()

target_compile_definitions(onnx PUBLIC $<TARGET_PROPERTY:onnx_proto,INTERFACE_COMPILE_DEFINITIONS> PRIVATE "__ONNX_DISABLE_STATIC_REGISTRATION")
if (NOT onnxruntime_USE_FULL_PROTOBUF)
  target_compile_definitions(onnx PUBLIC "__ONNX_NO_DOC_STRINGS")
endif()
set_target_properties(onnx PROPERTIES FOLDER "External/ONNX")
set_target_properties(onnx_proto PROPERTIES FOLDER "External/ONNX")

if (onnxruntime_USE_MKLML)
  add_definitions(-DUSE_MKLML=1 -DUSE_MKLML_FOR_BLAS=1)
  if (WIN32 OR APPLE)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES mklml)
  else()
    if(onnxruntime_USE_OPENMP)
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES mklml_gnu)
    else()
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES mklml_intel)
    endif()
  endif()
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES project_mklml)
  include_directories(${MKLML_INCLUDE_DIR})
  link_directories(${MKLML_LIB_DIR})
endif()

if (onnxruntime_USE_NGRAPH)
  if (NOT onnxruntime_USE_FULL_PROTOBUF)
    message(FATAL_ERROR "Please set onnxruntime_USE_FULL_PROTOBUF=ON for nGraph execution provider.")
  endif()
  add_definitions(-DUSE_NGRAPH=1)
  include(ngraph)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ngraph)
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES project_ngraph)
endif()

# fix a warning in onnx code we can't do anything about
if (MSVC)
   string(APPEND CMAKE_CXX_FLAGS " -DEIGEN_HAS_C99_MATH") # required to be set explicitly to enable Eigen-Unsupported SpecialFunctions
endif()

#Dependencies end. In the next we'll enable "treat warning as error"

if (onnxruntime_RUN_ONNX_TESTS)
  add_definitions(-DORT_RUN_EXTERNAL_ONNX_TESTS)
endif()

# Flatbuffers
# We do not need to build flatc for iOS or Android Cross Compile
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "FLATBUFFERS_BUILD_FLATC" FORCE)
endif()
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "FLATBUFFERS_BUILD_TESTS" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "FLATBUFFERS_INSTALL" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "FLATBUFFERS_BUILD_FLATHASH" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "FLATBUFFERS_BUILD_FLATLIB" FORCE)
add_subdirectory(external/flatbuffers EXCLUDE_FROM_ALL)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES flatbuffers)
list(APPEND onnxruntime_EXTERNAL_LIBRARIES flatbuffers)

#Adjust warning flags
if (WIN32)
    add_definitions(-DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES)
    if(onnxruntime_ENABLE_MEMLEAK_CHECKER)
      add_definitions(-DONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
    endif()
    # parallel build
    # These compiler opitions cannot be forwarded to NVCC, so cannot use add_compiler_options
    string(APPEND CMAKE_CXX_FLAGS " /MP /W4")
    #Compiler bug, we should get such warnings. It will be fixed in a new VC release
    string(APPEND CMAKE_CXX_FLAGS " /wd4127")
    # class needs to have dll-interface to be used by clients
    string(APPEND CMAKE_CXX_FLAGS " /wd4251")
    # issued by thrust nonstandard extension used: nameless struct/union
    string(APPEND CMAKE_CXX_FLAGS " /wd4201")
    if (onnxruntime_ENABLE_STATIC_ANALYSIS)
        string(APPEND CMAKE_CXX_FLAGS
            " /analyze:stacksize 131072"
            # disable warning because there are many occurrences from test macros
            " /wd6326 " # potential comparison of a constant with another constant
        )
    endif()

    # Treat warning as error if onnxruntime_DEV_MODE is ON
    # For cross-compiled ARM64 binaries, there are too many warnings to fix, hence ignore warnings for now
    if (onnxruntime_DEV_MODE AND NOT CMAKE_CROSSCOMPILING)
      # treat warnings as errors
      string(APPEND CMAKE_CXX_FLAGS " /WX")
      foreach(type EXE STATIC SHARED)
        set(CMAKE_${type}_LINKER_FLAGS "${CMAKE_${type}_LINKER_FLAGS} /WX")
      endforeach()
    endif()

    # set linker flags to minimize the binary size.
    if (MSVC)
      foreach(type EXE STATIC SHARED)
        if (NOT type MATCHES STATIC)
          set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /OPT:REF,ICF,LBR")
          set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO")
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF,ICF,LBR")
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO")
          set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /OPT:REF,ICF,LBR")
          set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /INCREMENTAL:NO")
        endif()
        if (onnxruntime_ENABLE_LTO AND NOT onnxruntime_USE_CUDA)
          set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /LTCG")
        endif()
      endforeach()
    endif()
else()
  add_definitions(-DPLATFORM_POSIX)
  # Enable warning and enable space optimization in Linux
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -ffunction-sections -fdata-sections")
  string(APPEND CMAKE_C_FLAGS " -Wall -Wextra -ffunction-sections -fdata-sections")

  if(onnxruntime_DEV_MODE)
    string(APPEND CMAKE_CXX_FLAGS " -Werror")
    string(APPEND CMAKE_C_FLAGS " -Werror")
  endif()
  check_cxx_compiler_flag(-Wunused-but-set-variable HAS_UNUSED_BUT_SET_VARIABLE)
  check_cxx_compiler_flag(-Wunused-parameter HAS_UNUSED_PARAMETER)
  check_cxx_compiler_flag(-Wunused-variable HAS_UNUSED_VARIABLE)
  check_cxx_compiler_flag(-Wcast-function-type HAS_CAST_FUNCTION_TYPE)
  check_cxx_compiler_flag(-Wparentheses HAS_PARENTHESES)
  check_cxx_compiler_flag(-Wuseless-cast HAS_USELESS_CAST)
  check_cxx_compiler_flag(-Wnonnull-compare HAS_NONNULL_COMPARE)
  check_cxx_compiler_flag(-Wtautological-pointer-compare HAS_TAUTOLOGICAL_POINTER_COMPARE)
  check_cxx_compiler_flag(-Wcatch-value HAS_CATCH_VALUE)
  check_cxx_compiler_flag(-Wmissing-braces HAS_MISSING_BRACES)
  check_cxx_compiler_flag(-Wignored-attributes HAS_IGNORED_ATTRIBUTES)
  check_cxx_compiler_flag(-Wdeprecated-copy HAS_DEPRECATED_COPY)
  check_cxx_compiler_flag(-Wdeprecated-declarations HAS_DEPRECATED_DECLARATIONS)
  check_cxx_compiler_flag(-Wclass-memaccess HAS_CLASS_MEMACCESS)
  check_cxx_compiler_flag(-Wmaybe-uninitialized HAS_MAYBE_UNINITIALIZED)

  if(HAS_TAUTOLOGICAL_POINTER_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -Wno-tautological-pointer-compare")
    string(APPEND CMAKE_C_FLAGS_DEBUG " -Wno-tautological-pointer-compare")
  endif()
  if(HAS_NONNULL_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -Wno-nonnull-compare")
    string(APPEND CMAKE_C_FLAGS_DEBUG " -Wno-nonnull-compare")
  endif()
  if(HAS_DEPRECATED_COPY)
    #too many such errors in eigen and gemmlowp
    string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-copy")
  endif()
  if(HAS_PARENTHESES)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-parentheses")
  endif()
endif()
set(onnxruntime_DELAYLOAD_FLAGS "")
if (onnxruntime_USE_JEMALLOC)
  if (onnxruntime_USE_MIMALLOC_STL_ALLOCATOR OR onnxruntime_USE_MIMALLOC_ARENA_ALLOCATOR)
    message( FATAL_ERROR "You cannot specify both jemalloc and mimalloc." )
  endif()

  if (Win32)
    message( FATAL_ERROR "Jemalloc is not supported on Windows." )
  endif()
  include(jemalloc)
  add_definitions(-DUSE_JEMALLOC=1)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${JEMALLOC_STATIC_LIBRARIES})
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES jemalloc)
endif()

include_directories(
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${REPO_ROOT}/include/onnxruntime/core/session
)


if(onnxruntime_USE_OPENVINO)

  add_definitions(-DUSE_OPENVINO=1)

  # Check OpenVINO version for support
  if ($ENV{INTEL_OPENVINO_DIR} MATCHES "2020.2")
    set(OPENVINO_VERSION "2020.2")
    add_definitions(-DOPENVINO_2020_2=1)
  elseif ($ENV{INTEL_OPENVINO_DIR} MATCHES "2020.3")
    set(OPENVINO_VERSION "2020.3")
    add_definitions(-DOPENVINO_2020_3=1)
  elseif ($ENV{INTEL_OPENVINO_DIR} MATCHES "2020.4")
    set(OPENVINO_VERSION "2020.4")
    add_definitions(-DOPENVINO_2020_4=1)
  else()
    message(FATAL_ERROR "Unsupported OpenVINO version: ${INTEL_OPENVINO_DIR}")
  endif()

  if (NOT WIN32)
    if(OPENVINO_VERSION STREQUAL "2020.2")
      include(openvino)
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES ovep_ngraph)
      list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES project_ngraph)
    endif()
  endif()

  if(onnxruntime_USE_OPENVINO_MYRIAD)
    add_definitions(-DOPENVINO_CONFIG_MYRIAD=1)
  endif()

  if(onnxruntime_USE_OPENVINO_GPU_FP32)
    add_definitions(-DOPENVINO_CONFIG_GPU_FP32=1)
  endif()

  if(onnxruntime_USE_OPENVINO_GPU_FP16)
    add_definitions(-DOPENVINO_CONFIG_GPU_FP16=1)
  endif()

  if(onnxruntime_USE_OPENVINO_CPU_FP32)
    add_definitions(-DOPENVINO_CONFIG_CPU_FP32=1)
  endif()

  if(onnxruntime_USE_OPENVINO_VAD_M)
    add_definitions(-DOPENVINO_CONFIG_VAD_M=1)
  endif()

  if(onnxruntime_USE_OPENVINO_VAD_F)
    add_definitions(-DOPENVINO_CONFIG_VAD_F=1)
  endif()

endif()

if (onnxruntime_USE_VITISAI)
  if(WIN32)
    message(FATAL_ERROR "Vitis-AI execution provider is not supported on Windows.")
  else()
    add_definitions(-DUSE_VITISAI=1)
    include(pyxir)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES pyxir)
    list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES pyxir)
  endif()
endif()

if (onnxruntime_USE_OPENBLAS)
  add_definitions(-DUSE_OPENBLAS=1)
  if (WIN32)
    include_directories(${onnxruntime_OPENBLAS_HOME})
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${onnxruntime_OPENBLAS_HOME}/lib/libopenblas.lib)
  else()
    # on linux we assume blas is installed via 'apt-get install libopenblas-dev'
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES openblas)
  endif()
endif()

configure_file(onnxruntime_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_config.h)

if (onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA=1)
  #The following 6 lines are copied from https://gitlab.kitware.com/cmake/cmake/issues/17559
  set( CMAKE_CUDA_FLAGS "" CACHE STRING "" )
  if ( CMAKE_CUDA_FLAGS )
    list(REMOVE_ITEM CMAKE_CUDA_FLAGS "-cudart static")
  endif()
  if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0")
    # cmake 3.17.0 introduces CMAKE_CUDA_RUNTIME which must be used instead of explicit -cudart shared flag
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  else()
    string(APPEND CMAKE_CUDA_FLAGS "-cudart shared")
  endif()
  enable_language(CUDA)
  message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    set(CMAKE_CUDA_STANDARD 14)
  else()
    set(CMAKE_CUDA_STANDARD 11)
  endif()
  file(TO_CMAKE_PATH ${onnxruntime_CUDNN_HOME} onnxruntime_CUDNN_HOME)
  set(ONNXRUNTIME_CUDA_LIBRARIES ${CUDA_LIBRARIES})

  if (onnxruntime_ENABLE_NVTX_PROFILE)
    list(APPEND ONNXRUNTIME_CUDA_LIBRARIES cublas cudnn curand cufft nvToolsExt)
  else()
    list(APPEND ONNXRUNTIME_CUDA_LIBRARIES cublas cudnn curand cufft)
  endif()

  if (WIN32)
    link_directories(${onnxruntime_CUDNN_HOME}/lib/x64)

    # delayload causes crash on exit, so disable for now
    #file(GLOB cuda_dll_paths "${onnxruntime_CUDA_HOME}/bin/cublas64_*" "${onnxruntime_CUDA_HOME}/bin/cudart64_*" "${onnxruntime_CUDA_HOME}/bin/curand64_*" "${onnxruntime_CUDA_HOME}/bin/cufft64_*")
    #set(onnxruntime_DELAYLOAD_FLAGS "${onnxruntime_DELAYLOAD_FLAGS} /DELAYLOAD:cudnn64_7.dll")
    #foreach(cuda_dll_path ${cuda_dll_paths})
    #    get_filename_component(cuda_dll_file_name ${cuda_dll_path} NAME)
    #    set(onnxruntime_DELAYLOAD_FLAGS "${onnxruntime_DELAYLOAD_FLAGS} /DELAYLOAD:${cuda_dll_file_name}")
    #endforeach(cuda_dll_path)

  else()
    link_directories(${onnxruntime_CUDNN_HOME}/lib64)
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${ONNXRUNTIME_CUDA_LIBRARIES})
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL "aarch64-linux-gnu")
      # Support for Jetson/Tegra ARM devices
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_53,code=sm_53") # TX1, Nano
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_62,code=sm_62") # TX2
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_72,code=sm_72") # AGX Xavier, NX Xavier
    else()
      # the following compute capabilities are removed in CUDA 11 Toolkit
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
        if (NOT onnxruntime_USE_TORCH)
          set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=sm_30") # K series
        endif()
      endif()
      # 37, 50 still work in CUDA 11 but are marked deprecated and will be removed in future CUDA version.
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_37,code=sm_37") # K80
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series

      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52") # M60
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80") # A series
      endif()
    endif()
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --default-stream legacy")
  if (NOT WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr --compiler-options -fPIC")
  endif()
  # Options passed to cudafe
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=bad_friend_decl\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=expr_has_no_effect\"")
endif()

if (onnxruntime_USE_MIGRAPHX)
  if (WIN32)
    message(FATAL_ERROR "MIGraphX does not support build in Windows!")
  endif()
  set(AMD_MIGRAPHX_HOME ${onnxruntime_MIGRAPHX_HOME})
  add_definitions(-DUSE_MIGRAPHX=1)
endif()

if (onnxruntime_USE_TVM)
  if (WIN32 AND MSVC)
    # wd4100: identifier' : unreferenced formal parameter
    # wd4244: conversion from 'int' to 'char', possible loss of data
    # wd4251: class X needs to have dll-interface to be used by clients of class Y
    # wd4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
    # wd4275: non dll-interface class X used as base for dll-interface class Y
    # wd4389: signed/unsigned mismatch
    # wd4456: declaration of X hides previous local declaration
    set(DISABLED_WARNINGS_FOR_TVM "/wd4100" "/wd4244" "/wd4251" "/wd4267" "/wd4275" "/wd4389" "/wd4456")
  else()
    set(DISABLED_WARNINGS_FOR_TVM "-Wno-error=ignored-qualifiers" "-Wno-unused-parameter" "-Wno-ignored-qualifiers")
    if(HAS_UNUSED_PARAMETER)
      list(APPEND DISABLED_WARNINGS_FOR_TVM "-Wno-error=unused-parameter")
    endif()
    if(HAS_CATCH_VALUE)
      #TODO: send a PR to TVM and fix it
      list(APPEND DISABLED_WARNINGS_FOR_TVM "-Wno-error=catch-value")
    endif()
  endif()
  include(onnxruntime_codegen.cmake)
endif()

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  add_definitions(-DMICROSOFT_INTERNAL)
endif()

if (onnxruntime_USE_DML)
  if(NOT WIN32)
    message(FATAL_ERROR "The DirectML execution provider is only supported when building for Windows.")
  endif()

  add_definitions(-DUSE_DML=1)
  include(dml)
endif()

if (onnxruntime_ENABLE_TRAINING)
  add_compile_definitions(ENABLE_TRAINING)

  if (UNIX)
    # Find MPI
    find_path(MPI_INCLUDE_DIR
      NAMES mpi.h
      HINTS
      ${onnxruntime_MPI_HOME}/include
      /bert_ort/openmpi/include)

    set(MPI_LIBNAME "mpi")

    find_library(MPI_LIBRARY
      NAMES ${MPI_LIBNAME}
      HINTS
      ${onnxruntime_MPI_HOME}/lib
      /bert_ort/openmpi/lib)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(MPI DEFAULT_MSG MPI_INCLUDE_DIR MPI_LIBRARY)

    if (MPI_FOUND)
      if(NOT DEFINED onnxruntime_MPI_HOME)
        execute_process(COMMAND mpirun --version OUTPUT_VARIABLE MPIRUN_OUTPUT)
      else()
        execute_process(COMMAND ${onnxruntime_MPI_HOME}/bin/mpirun --version OUTPUT_VARIABLE MPIRUN_OUTPUT)
      endif(NOT DEFINED onnxruntime_MPI_HOME)
      string( REGEX MATCH "[0-9]+.[0-9]+.[0-9]" MPI_VERSION ${MPIRUN_OUTPUT})
      message( STATUS "MPI Version: ${MPI_VERSION}")

      set(MPI_INCLUDE_DIRS ${MPI_INCLUDE_DIR})
      set(MPI_LIBRARIES ${MPI_LIBRARY})
      message( STATUS "MPI (include: ${MPI_INCLUDE_DIRS}, library: ${MPI_LIBRARIES})" )
      mark_as_advanced(MPI_INCLUDE_DIRS MPI_LIBRARIES)

      list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${MPI_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    else ()
      set(onnxruntime_USE_NCCL OFF)
      set(onnxruntime_USE_HOROVOD OFF)
      message( WARNING "MPI is not found. Please use --mpi_home to specify the path of MPI. Otherwise, NCCL or HOROVOD will be disabled." )
    endif()

    # Find NCCL and MPI
    if (onnxruntime_USE_NCCL AND MPI_FOUND)
      find_path(NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS
        ${onnxruntime_NCCL_HOME}/include
        $ENV{CUDA_ROOT}/include)

      set(NCCL_LIBNAME "nccl")

      find_library(NCCL_LIBRARY
        NAMES ${NCCL_LIBNAME}
        HINTS
        ${onnxruntime_NCCL_HOME}/lib/x86_64-linux-gnu
        ${onnxruntime_NCCL_HOME}/lib
        $ENV{CUDA_ROOT}/lib64)

      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

      if (NCCL_FOUND)
        set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIR}/nccl.h")
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

        add_definitions(-DUSE_NCCL=1)
        message( STATUS "NCCL is enabled in Linux GPU Build." )
      else ()
        set(onnxruntime_USE_NCCL OFF)
        message( WARNING "NCCL is not found. Please use --nccl_home to specify the path of NCCL. Otherwise, NCCL is disabled." )
      endif()
    endif()
  endif()

  if (onnxruntime_USE_HOROVOD AND MPI_FOUND)
    if (WIN32)
      message( FATAL_ERROR "Horovod is not supported on Windows." )
    elseif (UNIX)
      add_definitions(-DUSE_HOROVOD=1)
      set(HOROVOD_ROOT ${PROJECT_SOURCE_DIR}/external/horovod)
      set(HOROVOD_INCLUDE_DIRS  "${HOROVOD_ROOT}/horovod/common")
      add_subdirectory(horovod EXCLUDE_FROM_ALL)
      # use external/horovod/third_party/gloo/cmake/Modules/Findnccl.cmake to locate nccl lib path
      list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/horovod/third_party/gloo/cmake/Modules/)
      find_package(nccl REQUIRED)
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES horovod)
    endif()
  endif()

  add_subdirectory(tensorboard EXCLUDE_FROM_ALL)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES tensorboard)
endif()

#names in this var must match the directory names under onnxruntime/core/providers
set(ONNXRUNTIME_PROVIDER_NAMES cpu)
foreach(target_name onnxruntime_common onnxruntime_graph onnxruntime_framework onnxruntime_util onnxruntime_providers onnxruntime_optimizer onnxruntime_session onnxruntime_mlas)
  include(${target_name}.cmake)
  if (MSVC)
    target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /sdl>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/sdl>")
  else()
    target_compile_definitions(${target_name} PUBLIC -DNSYNC_ATOMIC_CPP11)
    target_include_directories(${target_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
  endif()
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
endforeach()

foreach(provider_name ${ONNXRUNTIME_PROVIDER_NAMES})
  if(NOT provider_name STREQUAL "cpu" AND NOT provider_name STREQUAL "winml")
    if (MSVC)
      target_compile_options(onnxruntime_providers_${provider_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
      target_compile_options(onnxruntime_providers_${provider_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /sdl>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/sdl>")
    else()
      target_compile_definitions(onnxruntime_providers_${provider_name} PUBLIC -DNSYNC_ATOMIC_CPP11)
      target_include_directories(onnxruntime_providers_${provider_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
    endif()
  endif()
endforeach()



if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES log)
endif()

if (onnxruntime_USE_OPENMP)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES OpenMP::OpenMP_CXX)
endif()

if(WIN32)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${SYS_PATH_LIB} Shlwapi)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES debug Dbghelp)
else()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES nsync_cpp)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${CMAKE_DL_LIBS} Threads::Threads)
endif()

if (WIN32)
  if (WINDOWS_STORE)
    # Setting WINAPI_FAMILY, WINVER and _WIN32_WINNT restrict the APIs exposed in Windows headers to those available
    # in store or desktop, and that support at least the version of Windows specified
    # add_compile_definitions(WINAPI_FAMILY=2)
    add_compile_definitions(WINVER=0x0A00 _WIN32_WINNT=0x0A00)  # Support Windows 10
    # /ZW adds /FUplatform.winmd and /FUwindows.winmd. windows.winmd, in turn, overrides the include path for
    # the cppwinrt headers, which we set to use the WinML built ones.
    # Instead, we use /ZW:nostdlib and force include platform.winml only.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:SHELL:/ZW:nostdlib /FUplatform.winmd>")
    get_filename_component(msvc_path "${CMAKE_C_COMPILER}/../../../.." ABSOLUTE)
    link_directories("${msvc_path}/lib/${onnxruntime_target_platform}/store")
    add_link_options(/APPCONTAINER)
  else()
    add_compile_definitions(WINVER=0x0601)  # Support Windows 7 and newer
    # add_compile_definitions(WINAPI_FAMILY=3)
    # FIXME adding _WIN32_WINNT=0x0601 and WINAPI_FAMILY is desirable. However, it hides some symbols that are used in WRL
    # headers and breaks the build of WinML. We should have separate WINVER/_WIN32_WINNT definitions for WinML
  endif()
endif()

# Default version parts for Microsoft.AI.MachineLearning.dll and onnxruntime.dll in non-ADO pipeline local builds
set(VERSION_MAJOR_PART   0 CACHE STRING "First part of numeric file/product version.")
set(VERSION_MINOR_PART   0 CACHE STRING "Second part of numeric file/product version.")
set(VERSION_BUILD_PART       0 CACHE STRING "Third part of numeric file/product version.")
set(VERSION_PRIVATE_PART     0 CACHE STRING "Fourth part of numeric file/product version.")
set(VERSION_STRING       "Internal Build" CACHE STRING "String representation of file/product version.")
include(wil.cmake)

if (onnxruntime_USE_WINML)
  # WINML uses and depends on the shared lib.  Note:  You can build WINML without DML and you will get a
  # CPU only WINML
  if (NOT onnxruntime_BUILD_SHARED_LIB)
    message(
        FATAL_ERROR
        "Option onnxruntime_USE_WINML can only be used when onnxruntime_BUILD_SHARED_LIB is also enabled")
  endif()
  include(winml.cmake)
endif() # if(onnxruntime_USE_WINML)


#The following files may use the 'onnxruntime_libs' and 'onnxruntime_EXTERNAL_LIBRARIES' vars

if (onnxruntime_BUILD_SHARED_LIB)
  include(onnxruntime.cmake)
endif()

if (onnxruntime_BUILD_JAVA)
  message(STATUS "Java Build is enabled")
  include(onnxruntime_java.cmake)
endif()

if (onnxruntime_BUILD_NODEJS)
  message(STATUS "Node.js Build is enabled")
  include(onnxruntime_nodejs.cmake)
endif()

# some of the tests rely on the shared libs to be
# built; hence the ordering
if (onnxruntime_BUILD_UNIT_TESTS)
  if (onnxruntime_ENABLE_PYTHON)
    include(onnxruntime_python.cmake)
  endif()
  include(onnxruntime_unittests.cmake)
endif()

if (onnxruntime_ENABLE_TRAINING)
  include(onnxruntime_training.cmake)
  if (onnxruntime_ENABLE_TRAINING_E2E_TESTS)
    include(onnxruntime_training_e2e_tests.cmake)
  endif()
endif()

if (onnxruntime_BUILD_CSHARP)
  message(STATUS "CSharp Build is enabled")
#  set_property(GLOBAL PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "netstandard2.0")
  include(onnxruntime_csharp.cmake)
endif()

if (WINDOWS_STORE)
  target_link_options(onnxruntime PRIVATE /DYNAMICBASE /NXCOMPAT /APPCONTAINER)
  target_link_options(winml_dll PRIVATE /DYNAMICBASE /NXCOMPAT /APPCONTAINER)

  if (onnxruntime_target_platform STREQUAL "x86")
    target_link_options(onnxruntime PRIVATE /SAFESEH)
    target_link_options(winml_dll PRIVATE /SAFESEH)
  endif()
endif()

include(flake8.cmake)
