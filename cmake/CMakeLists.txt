
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0069 NEW)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.15")
  cmake_policy(SET CMP0092 NEW)
endif()

# Project
project(onnxruntime C CXX)

include(CheckCXXCompilerFlag)
include(CheckLanguage)

# CentOS compiler is old but it does allow certain C++14 features
# such as lambda captures and they are convinient
# On the other hand it does not allow some others.
# So we cant' regulate simply with the standard.
set(CMAKE_CXX_STANDARD 14)

# General C# prperties
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

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# NOTE: POSITION INDEPENDENT CODE hurts performance, and it only make sense on POSIX systems
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable CTest
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
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
option(onnxruntime_USE_NNAPI "Build with DNNLibrary for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_MKLML "Build the default cpu provider with MKL-ML binary dependency" OFF)
option(onnxruntime_USE_FEATURIZERS "Build ML Featurizers support" OFF)
option(onnxruntime_USE_NGRAPH "Build with nGraph support" OFF)
option(onnxruntime_USE_OPENBLAS "Use openblas" OFF)
option(onnxruntime_DEV_MODE "Enable developer warnings and treat most of them as error." OFF)
option(onnxruntime_USE_JEMALLOC "Use jemalloc" OFF)
option(onnxruntime_MSVC_STATIC_RUNTIME "Compile for the static CRT" OFF)
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" ON)
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
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(onnxruntime_ENABLE_LTO "Enable link time optimization" OFF)
option(onnxruntime_CROSS_COMPILING "Cross compiling onnx runtime" OFF)

#It's preferred to turn it OFF when onnxruntime is dynamically linked to PROTOBUF
option(onnxruntime_USE_FULL_PROTOBUF "Link to libprotobuf instead of libprotobuf-lite when this option is ON" OFF)
option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" OFF)
option(tensorflow_C_PACKAGE_PATH "Path to tensorflow C package installation dir")
option(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS "Enable operator implemented in language other than cpp" OFF)
option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS "Dump node input shapes and output data to standard output when executing the model." OFF)
option(onnxruntime_USE_DML "Build with DirectML support" OFF)
option(onnxruntime_USE_WINML "Build with WinML support" OFF)
option(onnxruntime_USE_ACL "Build with ACL support" OFF)
option(onnxruntime_USE_ACL_1902 "Build with ACL version 1902 support" OFF)
option(onnxruntime_USE_ACL_1905 "Build with ACL version 1905 support" OFF)
option(onnxruntime_USE_ACL_1908 "Build with ACL version 1908 support" OFF)
option(onnxruntime_ENABLE_INSTRUMENT "Enable Instrument with Event Tracing for Windows (ETW)" OFF)
option(onnxruntime_USE_TELEMETRY "Build with Telemetry" OFF)
#The onnxruntime_PREFER_SYSTEM_LIB is mainly designed for package managers like apt/yum/vcpkg.
#Please note, by default Protobuf_USE_STATIC_LIBS is OFF but it's recommended to turn it ON on Windows. You should set it properly when onnxruntime_PREFER_SYSTEM_LIB is ON otherwise you'll hit linkage errors.
#If you have already installed protobuf(or the others) in your system at the default system paths(like /usr/include), then it's better to set onnxruntime_PREFER_SYSTEM_LIB ON. Otherwise onnxruntime may see two different protobuf versions and we won't know which one will be used, the worst case could be onnxruntime picked up header files from one of them but the binaries from the other one.
#It's default OFF because it's experimental now.
option(onnxruntime_PREFER_SYSTEM_LIB "Experimental: Build with the preinstalled libraries in your system" OFF)



# training options
option(onnxruntime_ENABLE_TRAINING "Enable training functionality." OFF)
option(onnxruntime_ENABLE_TRAINING_E2E_TESTS "Enable training end-to-end tests." OFF)
option(onnxruntime_USE_HOROVOD "Build with HOROVOD support" OFF)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
#nsync tests failed on Mac Build
set(NSYNC_ENABLE_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
set(ONNX_ML 1)
if(NOT onnxruntime_ENABLE_PYTHON)
  set(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS OFF)
endif()

if(NOT WIN32)
  #TODO: On Linux we may try https://github.com/microsoft/TraceLogging
  if(onnxruntime_ENABLE_INSTRUMENT)
    message(WARNING "Instrument is only supported on Windows now")
    set(onnxruntime_ENABLE_INSTRUMENT OFF)
  endif()
else()
  check_cxx_compiler_flag(/d2FH4- HAS_D2FH4)
  if (HAS_D2FH4)
    message("Disabling /d2FH4")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /d2FH4-")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /d2FH4-")
  endif()
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
    # MKLML and NGraph depend on their own OpenMP library that may be different with the compiler's.
    # Disable the options to build mklml/NGraph and OpenMP together.
    if(onnxruntime_USE_MKLML)
      message(FATAL_ERROR "Please use only one of onnxruntime_USE_MKLML, onnxruntime_USE_OPENMP")
    endif()
    if(onnxruntime_USE_NGRAPH)
      message(FATAL_ERROR "Please use only one of onnxruntime_USE_NGRAPH, onnxruntime_USE_OPENMP")
    endif()
  else()
    set(onnxruntime_USE_OPENMP OFF)
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
  string(APPEND CMAKE_CXX_FLAGS " /EHsc /wd26812")
  string(APPEND CMAKE_C_FLAGS " /EHsc /wd26812")
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

#Dependencies
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

if(NOT WIN32)
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
  find_package(protobuf)
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

if(UNIX AND onnxruntime_ENABLE_LTO)
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS")
# no bounds checking in release build so no perf cost
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")

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
if (onnxruntime_USE_ACL OR onnxruntime_USE_ACL_1902 OR onnxruntime_USE_ACL_1905 OR onnxruntime_USE_ACL_1908)
  set(onnxruntime_USE_ACL ON)
  if(onnxruntime_USE_ACL_1902)
    add_definitions(-DACL_1902=1)
  else()
    if(onnxruntime_USE_ACL_1908)
      add_definitions(-DACL_1908=1)
    else()
      add_definitions(-DACL_1905=1)
    endif()
  endif()

  list(APPEND onnxruntime_EXTERNAL_LIBRARIES arm_compute arm_compute_graph arm_compute_core)
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

if (APPLE)
  #onnx/onnx/proto_utils.h:34:16: error: 'SetTotalBytesLimit' is deprecated: Please use the single
  #parameter version of SetTotalBytesLimit(). The second parameter is ignored.
  #  coded_stream.SetTotalBytesLimit((2048LL << 20) - 1, 512LL << 20);
  #TODO: fix the warning in ONNX and re-enable this flag
  string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated")
  string(APPEND CMAKE_C_FLAGS " -Wno-deprecated")
endif()
# ONNX
add_subdirectory(onnx)

#set_target_properties(gen_onnx_proto PROPERTIES FOLDER "External/ONNX")
# fix a warning in onnx code we can't do anything about
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_HAS_C99_MATH") # required to be set explicitly to enable Eigen-Unsupported SpecialFunctions
endif()

if (onnxruntime_RUN_ONNX_TESTS)
  add_definitions(-DORT_RUN_EXTERNAL_ONNX_TESTS)
endif()

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
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF,ICF,LBR")
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO")
          #TODO: the "/LTCG" switch should be controlled by onnxruntime_ENABLE_LTO
        endif()
        if (onnxruntime_ENABLE_LTO AND NOT onnxruntime_USE_CUDA)
          set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /LTCG")
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

if (onnxruntime_USE_MKLML)
  add_definitions(-DUSE_MKLML=1 -DUSE_MKLML_FOR_BLAS=1)
  if (WIN32 OR APPLE)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES mklml)
  else()
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES mklml_intel)
  endif()
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES project_mklml)
  include_directories(${MKLML_INCLUDE_DIR})
  link_directories(${MKLML_LIB_DIR})
endif()

if (onnxruntime_USE_DNNL)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES dnnl)
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES project_dnnl)
  link_directories(${DNNL_LIB_DIR})
endif()

if (onnxruntime_USE_NGRAPH)
  #if (onnxruntime_USE_OPENMP)
  #  message(FATAL_ERROR "Please set onnxruntime_USE_OPENMP=OFF for nGraph execution provider.")
  #endif()
  if (NOT onnxruntime_USE_FULL_PROTOBUF)
    message(FATAL_ERROR "Please set onnxruntime_USE_FULL_PROTOBUF=ON for nGraph execution provider.")
  endif()
  add_definitions(-DUSE_NGRAPH=1)
  include(ngraph)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ngraph)
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES project_ngraph)
endif()

if(onnxruntime_USE_OPENVINO)

  if (NOT onnxruntime_USE_FULL_PROTOBUF)
    message(FATAL_ERROR "Please set onnxruntime_USE_FULL_PROTOBUF=ON for OpenVINO execution provider.")
  endif()

  add_definitions(-DUSE_OPENVINO=1)
  if (NOT WIN32)
    include(openvino)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES ovep_ngraph)
    list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES project_ngraph)
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
  set(CMAKE_CUDA_STANDARD 11)
  file(TO_CMAKE_PATH ${onnxruntime_CUDNN_HOME} onnxruntime_CUDNN_HOME)
  set(ONNXRUNTIME_CUDA_LIBRARIES ${CUDA_LIBRARIES})
  list(APPEND ONNXRUNTIME_CUDA_LIBRARIES cublas cudnn curand cufft)
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

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=sm_30") # K series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --default-stream legacy")
  if (NOT WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr --compiler-options -fPIC")
  endif()
  # Options passed to cudafe
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=bad_friend_decl\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=expr_has_no_effect\"")
endif()

if (onnxruntime_USE_TENSORRT)
  if (WIN32)
    set(onnxruntime_DELAYLOAD_FLAGS "${onnxruntime_DELAYLOAD_FLAGS} /DELAYLOAD:nvinfer.dll /DELAYLOAD:nvinfer_plugin.dll")
  else()
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  endif()
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
    set(DISABLED_WARNINGS_FOR_TVM "-Wno-error=ignored-qualifiers")
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
  if (onnxruntime_USE_HOROVOD)
    if (WIN32)
      message( FATAL_ERROR "Horovod is not supported on Windows." )
    elseif (UNIX)
      find_package(MPI REQUIRED)
      add_definitions(-DUSE_HOROVOD=1)
      set(HOROVOD_ROOT ${PROJECT_SOURCE_DIR}/external/horovod)
      set(HOROVOD_INCLUDE_DIRS  "${HOROVOD_ROOT}/horovod/common" ${MPI_CXX_INCLUDE_PATH})
      add_subdirectory(horovod EXCLUDE_FROM_ALL)
      # use external/horovod/third_party/gloo/cmake/Modules/Findnccl.cmake to locate nccl lib path
      list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/horovod/third_party/gloo/cmake/Modules/)
      find_package(nccl REQUIRED)  
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES horovod ${NCCL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    endif()
  endif()

  if (onnxruntime_USE_CUDA)
    if (WIN32)
      message(WARNING "NCCL is not supported on Windows GPU Build." )
    elseif (UNIX)
      message( "NCCL is enabled in Linux GPU Build." )
      add_definitions(-DUSE_NCCL=1)
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

if(WIN32)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${SYS_PATH_LIB} Shlwapi)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES debug Dbghelp)
else()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES nsync_cpp)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${CMAKE_DL_LIBS} Threads::Threads)
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

# some of the tests rely on the shared libs to be
# built; hence the ordering
if (onnxruntime_BUILD_UNIT_TESTS)
  if (onnxruntime_ENABLE_PYTHON)
    if(UNIX)
      set(CMAKE_SKIP_BUILD_RPATH ON)
    endif()
    include(onnxruntime_python.cmake)
  endif()
  # we need to make sure this is turned off since it
  # turned ON by the previous step when building a shared lib
  set(CMAKE_SKIP_BUILD_RPATH OFF)
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

if (onnxruntime_BUILD_FOR_WINDOWS_STORE)
  set_target_properties(onnxruntime PROPERTIES VS_GLOBAL_AppContainerApplication true)
  set_target_properties(winml_dll PROPERTIES VS_GLOBAL_AppContainerApplication true)

  target_link_options(onnxruntime PRIVATE /DYNAMICBASE /NXCOMPAT /APPCONTAINER)
  target_link_options(winml_dll PRIVATE /DYNAMICBASE /NXCOMPAT /APPCONTAINER)

  if (onnxruntime_target_platform STREQUAL "x86")
    target_link_options(onnxruntime PRIVATE /SAFESEH)
    target_link_options(winml_dll PRIVATE /SAFESEH)
  endif()
endif()
