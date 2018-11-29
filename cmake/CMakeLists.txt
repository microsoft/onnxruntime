# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.11)

# Project
project(onnxruntime C CXX)
include(CheckCXXCompilerFlag)
include(CheckLanguage)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# General C# prperties
if (onnxruntime_BUILD_CSHARP)
  check_language(CSharp)
  if (CMAKE_CSharp_COMPILER)
    enable_language(CSharp)
    set(CMAKE_DOTNET_TARGET_FRAMEWORK_VERSION v4.6.1)
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
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_EIGEN_FOR_BLAS "Use eign for blas" ON)
option(onnxruntime_USE_MLAS "Use optimized blas library for GEMM and 2D Convolution" OFF)
option(onnxruntime_USE_MKLDNN "Build with MKL-DNN support" OFF)
option(onnxruntime_USE_MKLML "Build MKL-DNN with MKL-ML binary dependency" OFF)
option(onnxruntime_USE_OPENBLAS "Use openblas" OFF)
option(onnxruntime_DEV_MODE "Enable developer warnings and treat most of them as error." OFF)
option(onnxruntime_USE_PREBUILT_PB "Use prebuilt protobuf library" OFF)
option(onnxruntime_USE_JEMALLOC "Use jecmalloc" OFF)
option(onnxruntime_MSVC_STATIC_RUNTIME "Compile for the static CRT" OFF)
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" ON)
option(onnxruntime_USE_PREINSTALLED_EIGEN "Use pre-installed EIGEN. Need to provide eigen_SOURCE_PATH if turn this on." OFF)
option(onnxruntime_BUILD_BENCHMARKS "Build ONNXRuntime micro-benchmarks" OFF)
option(onnxruntime_USE_TVM "Build tvm for code-gen" OFF)
option(onnxruntime_USE_LLVM "Build tvm with LLVM" OFF)
option(onnxruntime_USE_OPENMP "Build with OpenMP support" OFF)
option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" OFF)
option(onnxruntime_ENABLE_MICROSOFT_INTERNAL "Use this option to enable/disable microsoft internal only code" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nupha" OFF)
option(onnxruntime_USE_BRAINSLICE "Build with BrainSlice" OFF)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
set(ONNX_ML 1)

set(REPO_ROOT ${PROJECT_SOURCE_DIR}/..)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../onnxruntime)
file (STRINGS "${REPO_ROOT}/VERSION_NUMBER" VERSION_NUMBER)

if (MSVC)
  if (onnxruntime_MSVC_STATIC_RUNTIME)
    # set all of our submodules to static runtime
    set(ONNX_USE_MSVC_STATIC_RUNTIME ON)
    set(protobuf_MSVC_STATIC_RUNTIME ON)
    set(gtest_force_shared_crt OFF)

    # In case we are building static libraries, link also the runtime library statically
    # so that MSVCR*.DLL is not required at runtime.
    # https://msdn.microsoft.com/en-us/library/2kzt1wy3.aspx
    # This is achieved by replacing msvc option /MD with /MT and /MDd with /MTd
    # http://www.cmake.org/Wiki/CMake_FAQ#How_can_I_build_my_MSVC_application_with_a_static_runtime.3F
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE__FLAGS_RELWITHDEBINFO)
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
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  #Disable 4100 globally. Too many this kind errors in protobuf
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100")
else()
  # Enable OpenMP for Non-Windows only. WinML team disallows use of OpenMP.
  find_package(OpenMP)
  if (OPENMP_FOUND)
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      add_definitions(-DUSE_OPENMP)
  endif()
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    #For Mac compliance
    message("Adding flags for Mac builds")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")
endif()

find_package(PNG)
set(ENABLE_DATE_TESTING  OFF CACHE BOOL "" FORCE)
set(USE_SYSTEM_TZ_DB  ON CACHE BOOL "" FORCE)

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
    find_package(PythonLibs 3.4 REQUIRED)
  endif()
endif()

if(onnxruntime_BUILD_BENCHMARKS)
  if(NOT TARGET benchmark)
    # We will not need to test benchmark lib itself.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
    # We will not need to install benchmark since we link it statically.
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/onnx/third_party/benchmark EXCLUDE_FROM_ALL)
  endif()
endif()

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

#Here we support three build mode:
#1. (recommended)onnxruntime_USE_PREBUILT_PB is set (ONNX_CUSTOM_PROTOC_EXECUTABLE should also be set)
#   We will not build protobuf, will use a prebuilt binary instead. This mode can also support cross-compiling
#2. onnxruntime_USE_PREBUILT_PB is not set but ONNX_CUSTOM_PROTOC_EXECUTABLE is set
#   Build Protobuf from source, except protoc.exe. This mode is mainly for cross-compiling
#3. both onnxruntime_USE_PREBUILT_PB and ONNX_CUSTOM_PROTOC_EXECUTABLE are not set
#   Compile everything from source code. Slowest option.

if(onnxruntime_USE_PREBUILT_PB)
  get_filename_component(
     _PROTOBUF_INSTALL_PREFIX
     ${ONNX_CUSTOM_PROTOC_EXECUTABLE}
     DIRECTORY)
  get_filename_component(
    _PROTOBUF_INSTALL_PREFIX
    ${_PROTOBUF_INSTALL_PREFIX}/..
    REALPATH)
  if(WIN32)
    include(${_PROTOBUF_INSTALL_PREFIX}/cmake/protobuf-config.cmake)
  else()
    include(${_PROTOBUF_INSTALL_PREFIX}/lib64/cmake/protobuf/protobuf-config.cmake)
  endif()
  include(protobuf_function.cmake)
else()
  # use protobuf as a submodule
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/protobuf/cmake EXCLUDE_FROM_ALL)
  set_target_properties(libprotobuf PROPERTIES FOLDER "External/Protobuf")
  set_target_properties(libprotobuf-lite PROPERTIES FOLDER "External/Protobuf")
  set_target_properties(libprotoc PROPERTIES FOLDER "External/Protobuf")
  set_target_properties(protoc PROPERTIES FOLDER "External/Protobuf")
  add_library(protobuf::libprotobuf ALIAS libprotobuf)
  add_executable(protobuf::protoc ALIAS protoc)
  include(protobuf_function.cmake)
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

# if ON put all the unit tests in a single project so that code coverage is more comprehensive.
# defaulting to that and most likely removing option to have separate unit test projects in the near future.
set(SingleUnitTestProject ON)
if (onnxruntime_SPLIT_UNIT_TEST_PROJECTS)
  set(SingleUnitTestProject OFF)
endif()

get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_INCLUDE_DIR ${REPO_ROOT}/include/onnxruntime)

add_subdirectory(external/date EXCLUDE_FROM_ALL)
add_subdirectory(external/gsl EXCLUDE_FROM_ALL)
add_library(date ALIAS tz)
add_library(gsl ALIAS GSL)

set(date_INCLUDE_DIR $<TARGET_PROPERTY:tz,INTERFACE_INCLUDE_DIRECTORIES>)

# bounds checking behavior.
# throw instead of calling terminate if there's a bounds checking violation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGSL_THROW_ON_CONTRACT_VIOLATION")
# no bounds checking in release build so no perf cost
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")

include(eigen)
set(onnxruntime_EXTERNAL_LIBRARIES protobuf::libprotobuf)

# gtest and gmock
add_subdirectory(${PROJECT_SOURCE_DIR}/external/googletest EXCLUDE_FROM_ALL)
set_target_properties(gmock PROPERTIES FOLDER "External/GTest")
set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()
endfunction()

# TVM
if (onnxruntime_USE_TVM)
  if (onnxruntime_USE_CUDA)
    set(USE_CUDA ON)
  endif()
  if (onnxruntime_USE_LLVM)
    set(USE_LLVM ON)
    add_definitions(-DUSE_TVM_WITH_LLVM)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/tvm EXCLUDE_FROM_ALL)
  set_target_properties(tvm PROPERTIES FOLDER "External/tvm")
  set_target_properties(tvm_topi PROPERTIES FOLDER "External/tvm")
  set_target_properties(tvm_runtime PROPERTIES FOLDER "External/tvm")
  set_target_properties(nnvm_compiler PROPERTIES FOLDER "External/tvm")

  set(TVM_INCLUDES ${PROJECT_SOURCE_DIR}/external/tvm/include
    ${PROJECT_SOURCE_DIR}/external/tvm/3rdparty/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/external/tvm/3rdparty/dlpack/include
    $<TARGET_PROPERTY:tvm,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:tvm_topi,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:nnvm_compiler,INTERFACE_INCLUDE_DIRECTORIES>)
  add_definitions(-DUSE_TVM)

  set(onnxruntime_tvm_libs
    onnxruntime_codegen_tvm
    tvm
    nnvm_compiler)

  set(onnxruntime_tvm_dependencies
    tvm
    nnvm_compiler
    onnxruntime_codegen_tvm)
endif()

# ONNX
add_subdirectory(onnx)
set_target_properties(onnx PROPERTIES FOLDER "External/ONNX")
set_target_properties(onnx_proto PROPERTIES FOLDER "External/ONNX")
#set_target_properties(gen_onnx_proto PROPERTIES FOLDER "External/ONNX")
# fix a warning in onnx code we can't do anything about
if (MSVC)
    target_compile_options(onnx_proto PRIVATE /wd4146) # unary minus operator applied to unsigned type
endif()
set(onnxruntime_EXTERNAL_DEPENDENCIES gsl onnx_proto)

if (onnxruntime_RUN_ONNX_TESTS)
  add_definitions(-DONNXRUNTIME_RUN_EXTERNAL_ONNX_TESTS)
endif()

add_definitions(-DUSE_MLAS)
#Adjust warning flags
if (WIN32)
    add_definitions(-DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES)
    # parallel build
    # These compiler opitions cannot be forwarded to NVCC, so cannot use add_compiler_options
    string(APPEND CMAKE_CXX_FLAGS " /MP")
    string(APPEND CMAKE_CXX_FLAGS
      " /wd4503" # Decorated name length exceeded.
      " /wd4127" # conditional expression is constant.
      " /wd4146" # unary minus operator applied to unsigned type. Needed for Protobuf
    )
    if (onnxruntime_ENABLE_STATIC_ANALYSIS)
        string(APPEND CMAKE_CXX_FLAGS
            " /analyze:WX- "
            # disable warning because there are many occurrences from test macros
            " /wd6326 " # potential comparison of a constant with another constant
        )
    endif()

    # set compile warning level to 3 on CUDA build but 4 on CPU-only build
    if(onnxruntime_USE_CUDA)
        #CMake hardcoded /W3 in its 'Windows-NVIDIA-CUDA.cmake'. We'd better keep consistent with it.
        #Change it to /W4 will result build failure
        string(APPEND CMAKE_CXX_FLAGS " /W3")
    else()
        string(APPEND CMAKE_CXX_FLAGS " /W4")
    endif()

    #only treat warning as error on x64 platform. For x86, right no there are too many warnings to fix
    if (CMAKE_SIZEOF_VOID_P EQUAL 8 AND onnxruntime_DEV_MODE)
      # treat warnings as errors
      string(APPEND CMAKE_CXX_FLAGS " /WX")
      foreach(type EXE STATIC SHARED)
        set(CMAKE_${type}_LINKER_FLAGS "${CMAKE_${type}_LINKER_FLAGS} /WX")
      endforeach()
    endif()
else()
  add_definitions(-DPLATFORM_POSIX)
  # Enable warning in Linux
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
  string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
  if(onnxruntime_DEV_MODE)
    string(APPEND CMAKE_CXX_FLAGS " -Werror")
    string(APPEND CMAKE_C_FLAGS " -Werror")
  endif()
  check_cxx_compiler_flag(-Wunused-but-set-variable HAS_UNUSED_BUT_SET_VARIABLE)
  check_cxx_compiler_flag(-Wunused-parameter HAS_UNUSED_PARAMETER)
  check_cxx_compiler_flag(-Wcast-function-type HAS_CAST_FUNCTION_TYPE)
  check_cxx_compiler_flag(-Wparentheses HAS_PARENTHESES)
  check_cxx_compiler_flag(-Wnull-dereference HAS_NULL_DEREFERENCE)
  check_cxx_compiler_flag(-Wuseless-cast HAS_USELESS_CAST)
  check_cxx_compiler_flag(-Wnonnull-compare HAS_NONNULL_COMPARE)
  check_cxx_compiler_flag(-Wtautological-pointer-compare HAS_TAUTOLOGICAL_POINTER_COMPARE)
  check_cxx_compiler_flag(-Wcatch-value HAS_CATCH_VALUE)
  if(HAS_NULL_DEREFERENCE)
    string(APPEND CMAKE_CXX_FLAGS " -Wnull-dereference")
    string(APPEND CMAKE_C_FLAGS " -Wnull-dereference")
  endif()
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
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=sign-compare -Wno-error=comment")
  if(onnxruntime_USE_CUDA)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=reorder")
  endif()
  if(HAS_PARENTHESES)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-parentheses")
  endif()
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=invalid-partial-specialization -Wno-error=missing-braces -Wno-error=inconsistent-missing-override")
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
    set(DISABLED_WARNINGS_FOR_TVM "-Wno-error=extra" "-Wno-error=ignored-qualifiers")
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

if (onnxruntime_USE_JEMALLOC)
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
  $<TARGET_PROPERTY:GSL,INTERFACE_INCLUDE_DIRECTORIES>
)

if (onnxruntime_USE_MKLDNN)
  add_definitions(-DUSE_MKLDNN=1)
  include(mkldnn)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES mkldnn)
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES mkldnn)
  link_directories(${MKLDNN_LIB_DIR})
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
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  file(TO_CMAKE_PATH ${onnxruntime_CUDA_HOME} CUDA_TOOLKIT_ROOT_DIR)
  file(TO_CMAKE_PATH ${onnxruntime_CUDNN_HOME} onnxruntime_CUDNN_HOME)
  find_package(CUDA 9.0 REQUIRED)
  include(cub)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  if (WIN32)
    link_directories(${onnxruntime_CUDNN_HOME}/lib/x64)
    set(ONNXRUNTIME_CUDA_LIBRARIES cudnn cublas)
  else()
    link_directories(${onnxruntime_CUDNN_HOME}/lib64)
    set(ONNXRUNTIME_CUDA_LIBRARIES cudnn_static cublas_static culibos)
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${ONNXRUNTIME_CUDA_LIBRARIES})
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES cub)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=sm_30") # K series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
  if (NOT WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")
  endif()
endif()

#names in this var must match the directory names under onnxruntime/core/providers
set(ONNXRUNTIME_PROVIDER_NAMES cpu)

include(onnxruntime_common.cmake)
include(onnxruntime_graph.cmake)
include(onnxruntime_framework.cmake)
include(onnxruntime_util.cmake)
include(onnxruntime_providers.cmake)
include(onnxruntime_session.cmake)
include(onnxruntime_mlas.cmake)


if (onnxruntime_BUILD_SHARED_LIB)
  include(onnxruntime.cmake)
endif()

if (onnxruntime_ENABLE_PYTHON)
if(UNIX)
  set(CMAKE_SKIP_BUILD_RPATH ON)
endif()
include(onnxruntime_python.cmake)
endif()

if (onnxruntime_BUILD_CSHARP)
  message(STATUS "CSharp Build is enabled")
#  set_property(GLOBAL PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "netstandard2.0")
  include(onnxruntime_csharp.cmake)
endif()

# some of the tests rely on the shared libs to be
# built; hence the ordering
if (onnxruntime_BUILD_UNIT_TESTS)
  # we need to make sure this is turned off since it
  # turned ON by the previous step when building a shared lib
  set(CMAKE_SKIP_BUILD_RPATH OFF)
  include(onnxruntime_unittests.cmake)
endif()