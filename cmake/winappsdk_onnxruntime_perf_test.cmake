# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT onnxruntime_BUILD_WINAPPSDK_PERF_TEST)
  message(FATAL_ERROR "onnxruntime_BUILD_WINAPPSDK_PERF_TEST is OFF")
endif()

if(NOT WIN32)
  message(FATAL_ERROR "winappsdk_onnxruntime_perf_test is only supported on Windows")
endif()

if(NOT MSVC)
  message(FATAL_ERROR "winappsdk_onnxruntime_perf_test is only supported with MSVC")
endif()

if(NOT onnxruntime_BUILD_SHARED_LIB)
  message(FATAL_ERROR "winappsdk_onnxruntime_perf_test requires onnxruntime_BUILD_SHARED_LIB to be ON")
endif()

if(onnxruntime_USE_CUDA OR onnxruntime_USE_NV OR onnxruntime_USE_TENSORRT)
  message(FATAL_ERROR "Unexpected - CUDA/NV/TensorRT usage in winappsdk_onnxruntime_perf_test")
endif()

if(NOT CPPWINRT_VERSION)
  message(FATAL_ERROR "Requires CPPWINRT_VERSION to be set")
endif()

message(STATUS "Using CPPWINRT_VERSION: ${CPPWINRT_VERSION}")

# [WinAppSDK]  Fetch and setup all the WinAppSDK dependencies
include(FetchContent)

FetchContent_Declare(
  NuGetCMakePackage
  GIT_REPOSITORY https://github.com/mschofie/NuGetCMakePackage
  GIT_TAG dc9e92672c6eb1c11f0d29d4f94731b3404cc096
)

FetchContent_MakeAvailable(NuGetCMakePackage)

add_nuget_packages(
  PACKAGES
  Microsoft.Windows.CppWinRT ${CPPWINRT_VERSION}
  Microsoft.Windows.ImplementationLibrary 1.0.250325.1
  Microsoft.WindowsAppSDK.Runtime 2.0.0-experimental3
  Microsoft.WindowsAppSDK.Foundation 2.0.8-experimental
  Microsoft.WindowsAppSDK.InteractiveExperiences 1.8.251104001
  Microsoft.WindowsAppSDK.ML 2.0.44-experimental
)

find_package(Microsoft.WindowsAppSDK.ML CONFIG REQUIRED)
find_package(Microsoft.Windows.ImplementationLibrary CONFIG REQUIRED)


#-------------------------------------------------------------------------------

# Source files
set(winappsdk_onnxruntime_perf_test_src_dir ${TEST_SRC_DIR}/perftest)

set(winappsdk_onnxruntime_perf_test_src_patterns
  "${winappsdk_onnxruntime_perf_test_src_dir}/*.cc"
  "${winappsdk_onnxruntime_perf_test_src_dir}/*.h")

list(APPEND winappsdk_onnxruntime_perf_test_src_patterns
  "${winappsdk_onnxruntime_perf_test_src_dir}/windows/*.cc"
  "${winappsdk_onnxruntime_perf_test_src_dir}/windows/*.h")

file(GLOB winappsdk_onnxruntime_perf_test_src CONFIGURE_DEPENDS
  ${winappsdk_onnxruntime_perf_test_src_patterns}
)

# EXE
onnxruntime_add_executable(winappsdk_onnxruntime_perf_test
  ${winappsdk_onnxruntime_perf_test_src}
  ${winappsdk_onnxruntime_perf_test_src_dir}/windows/app.manifest
  ${ONNXRUNTIME_ROOT}/core/platform/path_lib.cc
)

target_compile_options(winappsdk_onnxruntime_perf_test PRIVATE ${disabled_warnings})

target_include_directories(winappsdk_onnxruntime_perf_test PRIVATE ${onnx_test_runner_src_dir} ${ONNXRUNTIME_ROOT}
  ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir}
  ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(winappsdk_onnxruntime_perf_test
  PRIVATE
  ORT_API_MANUAL_INIT
  BUILD_WINAPPSDK_PERF_TEST
  MICROSOFT_WINDOWSAPPSDK_ML_DISABLE_AUTOINITIALIZE
)

# ABSL_FLAGS_STRIP_NAMES is set to 1 by default to disable flag registration when building for Android, iPhone, and "embedded devices".
# See the issue: https://github.com/abseil/abseil-cpp/issues/1875
# We set it to 0 for all builds to be able to use ABSL flags for onnxruntime_perf_test.
target_compile_definitions(winappsdk_onnxruntime_perf_test PRIVATE ABSL_FLAGS_STRIP_NAMES=0)

target_link_libraries(winappsdk_onnxruntime_perf_test
  PRIVATE
  onnx_test_runner_common
  onnxruntime_test_utils
  onnxruntime_common
  onnxruntime_flatbuffers
  onnx_test_data_proto
  absl::flags
  absl::flags_parse
  ${onnxruntime_EXTERNAL_LIBRARIES}

  Threads::Threads

  Microsoft.Windows.CppWinRT
  Microsoft.Windows.ImplementationLibrary
  Microsoft.WindowsAppSDK.Foundation
  Microsoft.WindowsAppSDK.ML

  # Microsoft.WindowsAppSDK.ML_Framework # Not needed, as target provides it's own custom implementaiton.
  onecoreuap.lib # but you need this for the appmodel APIs.
)
