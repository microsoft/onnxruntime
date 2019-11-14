# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime C CXX)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

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
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NSYNC "Build with NSYNC support. This option only takes effect on Linux" OFF)
option(onnxruntime_USE_EIGEN_FOR_BLAS "Use eign for blas" ON)
option(onnxruntime_USE_NNAPI "Build with DNNLibrary for Android NNAPI support" OFF)
option(onnxruntime_USE_MKLDNN "Build with MKL-DNN support" OFF)
option(onnxruntime_USE_MKLML "Build MKL-DNN with MKL-ML binary dependency" OFF)
option(onnxruntime_USE_GEMMLOWP "Build with gemmlowp for quantized gemm" OFF)
option(onnxruntime_USE_AUTOML "Build AutoML support" ON)
option(onnxruntime_USE_NGRAPH "Build with nGraph support" OFF)
option(onnxruntime_USE_OPENBLAS "Use openblas" OFF)
option(onnxruntime_DEV_MODE "Enable developer warnings and treat most of them as error." OFF)
option(onnxruntime_USE_JEMALLOC "Use jecmalloc" OFF)
option(onnxruntime_MSVC_STATIC_RUNTIME "Compile for the static CRT" OFF)
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" ON)
option(onnxruntime_USE_PREINSTALLED_EIGEN "Use pre-installed EIGEN. Need to provide eigen_SOURCE_PATH if turn this on." OFF)
option(onnxruntime_BUILD_BENCHMARKS "Build ONNXRuntime micro-benchmarks" OFF)
option(onnxruntime_USE_TVM "Build tvm for code-gen" OFF)
option(onnxruntime_BUILD_FOR_NATIVE_MACHINE "Enable this option for turning on optimization specific to this machine" OFF)
option(onnxruntime_USE_LLVM "Build tvm with LLVM" OFF)
option(onnxruntime_USE_OPENMP "Build with OpenMP support" OFF)
option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" OFF)
option(onnxruntime_ENABLE_MICROSOFT_INTERNAL "Use this option to enable/disable microsoft internal only code" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_BRAINSLICE "Build with BrainSlice" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(onnxruntime_ENABLE_LTO "Enable link time optimization" ON)
option(onnxruntime_CROSS_COMPILING "Cross compiling onnx runtime" OFF)
option(onnxruntime_BUILD_SERVER "Build ONNX Runtime Server" OFF)
option(onnxruntime_USE_FULL_PROTOBUF "Use full protobuf" OFF)
option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" OFF)
option(tensorflow_C_PACKAGE_PATH "Path to tensorflow C package installation dir")
option(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS "Enable operator implemented in language other than cpp" OFF)
option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS "Dump node input shapes and output data to standard output when executing the model." OFF)
option(onnxruntime_USE_DML "Build with DirectML support" OFF)
option(onnxruntime_USE_ACL "Build with ACL support" OFF)
option(onnxruntime_ENABLE_INSTRUMENT "Enable Instrument with Event Tracing for Windows (ETW)" OFF)

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
endif()

if(onnxruntime_USE_OPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DUSE_OPENMP)
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
  #TODO: figure out why nsync doesn't work
  if(onnxruntime_USE_NSYNC)
    message(WARNING "IPO is not supported when nsync is in use")
    set(onnxruntime_ENABLE_LTO OFF)
  else()
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_enabled OUTPUT ipo_output)
    if(NOT ipo_enabled)
      message(WARNING "IPO is not supported by this compiler")
      set(onnxruntime_ENABLE_LTO OFF)
    endif()
  endif()
endif()


set(REPO_ROOT ${PROJECT_SOURCE_DIR}/..)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../onnxruntime)
file (STRINGS "${REPO_ROOT}/VERSION_NUMBER" ORT_VERSION)

# Guarantee that the Eigen code that you are #including is licensed
# under the MPL2 and possibly more permissive licenses (like BSD).
add_definitions(-DEIGEN_MPL2_ONLY)

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
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  #Disable 4100 globally. Too many this kind errors in protobuf
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100")
  if (NOT onnxruntime_USE_CUDA)
    SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Gw /GL")
    SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Gw /GL")
  endif()
  check_cxx_compiler_flag(-Qspectre HAS_QSPECTRE)
  if (HAS_QSPECTRE)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Qspectre")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qspectre")
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
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native -mtune=native")
    string(APPEND CMAKE_C_FLAGS_RELEASE " -march=native -mtune=native")
    string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -march=native -mtune=native")
    string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -march=native -mtune=native")
  endif()
  if(onnxruntime_BUILD_x86)
    set (CMAKE_SYSTEM_PROCESSOR "x86")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse2 -mfpmath=sse -Wno-narrowing")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse -Wno-narrowing")
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
    find_package(PythonLibs 3.4 REQUIRED)
  endif()
endif()

#target begins from here

if(onnxruntime_BUILD_BENCHMARKS)
  if(NOT TARGET benchmark)
    # We will not need to test benchmark lib itself.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
    # We will not need to install benchmark since we link it statically.
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/onnx/third_party/benchmark EXCLUDE_FROM_ALL)
  endif()
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/external/nsync EXCLUDE_FROM_ALL)
# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

#Here we support two build mode:
#1. if ONNX_CUSTOM_PROTOC_EXECUTABLE is set, build Protobuf from source, except protoc.exe. This mode is mainly
#   for cross-compiling
#2. if ONNX_CUSTOM_PROTOC_EXECUTABLE is not set, Compile everything(including protoc) from source code.


# use protobuf as a submodule
add_subdirectory(${PROJECT_SOURCE_DIR}/external/protobuf/cmake EXCLUDE_FROM_ALL)
set_target_properties(libprotobuf PROPERTIES FOLDER "External/Protobuf")
set_target_properties(libprotobuf-lite PROPERTIES FOLDER "External/Protobuf")
set_target_properties(libprotoc PROPERTIES FOLDER "External/Protobuf")
set_target_properties(protoc PROPERTIES FOLDER "External/Protobuf")
if (onnxruntime_USE_FULL_PROTOBUF)
  add_library(protobuf::libprotobuf ALIAS libprotobuf)
else()
  add_library(protobuf::libprotobuf ALIAS libprotobuf-lite)
endif()
add_executable(protobuf::protoc ALIAS protoc)
include(protobuf_function.cmake)

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
add_subdirectory(external/re2 EXCLUDE_FROM_ALL)
set_target_properties(re2 PROPERTIES FOLDER "External/re2")

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
# mkldnn/mklml, openblas, onnxruntime_codegen_tvm, tvm, nnvm_compiler and pthread
# pthread is always at the last
set(onnxruntime_EXTERNAL_LIBRARIES onnx onnx_proto protobuf::libprotobuf re2)

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()
endfunction()

set(onnxruntime_EXTERNAL_DEPENDENCIES onnx_proto re2)

# ACL
if (onnxruntime_USE_ACL)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES arm_compute acl arm_compute_graph arm_compute_core)
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

    # treat warning as error only on x64 platform.
    # For x86 and cross-compiled ARM64 binaries, there are too many warnings to fix, hence ignore warnings for now
    if (CMAKE_SIZEOF_VOID_P EQUAL 8 AND onnxruntime_DEV_MODE AND NOT CMAKE_CROSSCOMPILING)
      # treat warnings as errors
      string(APPEND CMAKE_CXX_FLAGS " /WX")
      foreach(type EXE STATIC SHARED)
        set(CMAKE_${type}_LINKER_FLAGS "${CMAKE_${type}_LINKER_FLAGS} /WX")
      endforeach()
    endif()

    # set linker flags to minimize the binary size.
    if (MSVC)
      foreach(type EXE SHARED)
        set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF,ICF,LBR")
        set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO")
        if (NOT onnxruntime_USE_CUDA)
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
  check_cxx_compiler_flag(-Wcast-function-type HAS_CAST_FUNCTION_TYPE)
  check_cxx_compiler_flag(-Wparentheses HAS_PARENTHESES)
  check_cxx_compiler_flag(-Wuseless-cast HAS_USELESS_CAST)
  check_cxx_compiler_flag(-Wnonnull-compare HAS_NONNULL_COMPARE)
  check_cxx_compiler_flag(-Wtautological-pointer-compare HAS_TAUTOLOGICAL_POINTER_COMPARE)
  check_cxx_compiler_flag(-Wcatch-value HAS_CATCH_VALUE)
  check_cxx_compiler_flag(-Wmissing-braces HAS_MISSING_BRACES)
  check_cxx_compiler_flag(-Wignored-attributes HAS_IGNORED_ATTRIBUTES)
  check_cxx_compiler_flag(-Wdeprecated-copy HAS_DEPRECATED_COPY)
  check_cxx_compiler_flag(-Wclass-memaccess HAS_CLASS_MEMACCESS)

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
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -Wno-deprecated-copy")
    string(APPEND CMAKE_C_FLAGS_DEBUG " -Wno-deprecated-copy")
  endif()
  if(HAS_PARENTHESES)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-parentheses")
  endif()
endif()

if (onnxruntime_USE_JEMALLOC)
  if (onnxruntime_USE_MIMALLOC)
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

if (onnxruntime_USE_MKLDNN OR onnxruntime_USE_MKLML)
  include(mkldnn)
endif()

if(onnxruntime_USE_GEMMLOWP)
  add_definitions(-DUSE_GEMMLOWP=1)
endif()

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

if (onnxruntime_USE_MKLDNN)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES mkldnn)
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES project_mkldnn)
  link_directories(${MKLDNN_LIB_DIR})
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
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES ngraph)
endif()

if(onnxruntime_USE_OPENVINO)

  add_definitions(-DUSE_OPENVINO=1)

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
  string(APPEND CMAKE_CUDA_FLAGS "-cudart shared")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 11)
  file(TO_CMAKE_PATH ${onnxruntime_CUDNN_HOME} onnxruntime_CUDNN_HOME)
  set(ONNXRUNTIME_CUDA_LIBRARIES ${CUDA_LIBRARIES})
  list(APPEND ONNXRUNTIME_CUDA_LIBRARIES cublas cudnn)
  if (WIN32)
    link_directories(${onnxruntime_CUDNN_HOME}/lib/x64)

    file(GLOB cuda_dll_paths "${onnxruntime_CUDA_HOME}/bin/cublas64_*" "${onnxruntime_CUDA_HOME}/bin/cudart64_*")
    foreach(cuda_dll_path ${cuda_dll_paths})
        get_filename_component(cuda_dll_file_name ${cuda_dll_path} NAME)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DELAYLOAD:${cuda_dll_file_name}")
    endforeach(cuda_dll_path)

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DELAYLOAD:cudnn64_7.dll")
  else()
    link_directories(${onnxruntime_CUDNN_HOME}/lib64)
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${ONNXRUNTIME_CUDA_LIBRARIES})

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=sm_30") # K series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream legacy")
  if (NOT WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")
  endif()
endif()

if (onnxruntime_USE_TENSORRT)
  if (WIN32)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DELAYLOAD:nvinfer.dll")
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

#names in this var must match the directory names under onnxruntime/core/providers
set(ONNXRUNTIME_PROVIDER_NAMES cpu)

include(onnxruntime_common.cmake)
include(onnxruntime_graph.cmake)
include(onnxruntime_framework.cmake)
include(onnxruntime_util.cmake)
#The next file will define 'onnxruntime_libs' cmake var
include(onnxruntime_providers.cmake)
include(onnxruntime_optimizer.cmake)
include(onnxruntime_session.cmake)
include(onnxruntime_mlas.cmake)

if(onnxruntime_USE_AUTOML)
  add_definitions(-DMICROSOFT_AUTOML)
  # Build shared featurizer library
  include(onnxruntime_automl_featurizers.cmake)
endif()

if(WIN32)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES Shlwapi)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES debug Dbghelp)
else()
  if(onnxruntime_USE_NSYNC)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES nsync_cpp)
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${CMAKE_DL_LIBS} Threads::Threads)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES log)
endif()

#The following files may use the 'onnxruntime_libs' and 'onnxruntime_EXTERNAL_LIBRARIES' vars

if (onnxruntime_BUILD_SHARED_LIB)
  include(onnxruntime.cmake)
endif()

if (onnxruntime_BUILD_SERVER)
  include(onnxruntime_server.cmake)
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
  # gtest and gmock
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/googletest EXCLUDE_FROM_ALL)
  set_target_properties(gmock PROPERTIES FOLDER "External/GTest")
  set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
  set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
  set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")
  if(UNIX)
    target_compile_options(gmock PRIVATE "-w")
    target_compile_options(gtest PRIVATE "-w")
  endif()
  include(onnxruntime_unittests.cmake)
endif()

if (onnxruntime_BUILD_CSHARP)
  message(STATUS "CSharp Build is enabled")
#  set_property(GLOBAL PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "netstandard2.0")
  include(onnxruntime_csharp.cmake)
endif()
