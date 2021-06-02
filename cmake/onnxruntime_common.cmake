# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(onnxruntime_common_src_patterns
    "${ONNXRUNTIME_INCLUDE_DIR}/core/common/*.h"
    "${ONNXRUNTIME_INCLUDE_DIR}/core/common/logging/*.h"
    "${ONNXRUNTIME_INCLUDE_DIR}/core/platform/*.h"
    "${ONNXRUNTIME_ROOT}/core/common/*.h"
    "${ONNXRUNTIME_ROOT}/core/common/*.cc"
    "${ONNXRUNTIME_ROOT}/core/common/logging/*.h"
    "${ONNXRUNTIME_ROOT}/core/common/logging/*.cc"
    "${ONNXRUNTIME_ROOT}/core/common/logging/sinks/*.h"
    "${ONNXRUNTIME_ROOT}/core/common/logging/sinks/*.cc"
    "${ONNXRUNTIME_ROOT}/core/platform/env.h"
    "${ONNXRUNTIME_ROOT}/core/platform/env.cc"
    "${ONNXRUNTIME_ROOT}/core/platform/env_time.h"
    "${ONNXRUNTIME_ROOT}/core/platform/env_time.cc"
    "${ONNXRUNTIME_ROOT}/core/platform/path_lib.h"
    "${ONNXRUNTIME_ROOT}/core/platform/path_lib.cc"
    "${ONNXRUNTIME_ROOT}/core/platform/scoped_resource.h"
    "${ONNXRUNTIME_ROOT}/core/platform/telemetry.h"
    "${ONNXRUNTIME_ROOT}/core/platform/telemetry.cc"
    "${ONNXRUNTIME_ROOT}/core/platform/logging/make_platform_default_log_sink.h"
    "${ONNXRUNTIME_ROOT}/core/platform/logging/make_platform_default_log_sink.cc"
)

if(WIN32)
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/windows/*.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/*.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/logging/*.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/logging/*.cc"
    )
    # Windows platform adapter code uses advapi32, which isn't linked in by default in desktop ARM
    if (NOT WINDOWS_STORE)
        list(APPEND onnxruntime_EXTERNAL_LIBRARIES advapi32)
    endif()
else()
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/posix/*.h"
         "${ONNXRUNTIME_ROOT}/core/platform/posix/*.cc"
    )

    if (onnxruntime_USE_SYSLOG)
        list(APPEND onnxruntime_common_src_patterns
            "${ONNXRUNTIME_ROOT}/core/platform/posix/logging/*.h"
            "${ONNXRUNTIME_ROOT}/core/platform/posix/logging/*.cc"
        )
    endif()

    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
        list(APPEND onnxruntime_common_src_patterns
            "${ONNXRUNTIME_ROOT}/core/platform/android/logging/*.h"
            "${ONNXRUNTIME_ROOT}/core/platform/android/logging/*.cc"
        )
    endif()

    if (APPLE)
        list(APPEND onnxruntime_common_src_patterns
            "${ONNXRUNTIME_ROOT}/core/platform/apple/logging/*.h"
            "${ONNXRUNTIME_ROOT}/core/platform/apple/logging/*.mm"
            )
    endif()
endif()

if(CMAKE_GENERATOR_PLATFORM)
    # Multi-platform generator
    set(onnxruntime_target_platform ${CMAKE_GENERATOR_PLATFORM})
else()
    set(onnxruntime_target_platform ${CMAKE_SYSTEM_PROCESSOR})
endif()
if(onnxruntime_target_platform STREQUAL "ARM64")
    set(onnxruntime_target_platform "ARM64")
elseif(onnxruntime_target_platform STREQUAL "ARM64EC")
    set(onnxruntime_target_platform "ARM64EC")
elseif(onnxruntime_target_platform STREQUAL "ARM" OR CMAKE_GENERATOR MATCHES "ARM")
    set(onnxruntime_target_platform "ARM")
elseif(onnxruntime_target_platform STREQUAL "x64" OR onnxruntime_target_platform STREQUAL "x86_64" OR onnxruntime_target_platform STREQUAL "AMD64" OR CMAKE_GENERATOR MATCHES "Win64")
    set(onnxruntime_target_platform "x64")
elseif(onnxruntime_target_platform STREQUAL "Win32" OR onnxruntime_target_platform STREQUAL "x86" OR onnxruntime_target_platform STREQUAL "i386" OR onnxruntime_target_platform STREQUAL "i686")
    set(onnxruntime_target_platform "x86")
endif()

if(onnxruntime_target_platform STREQUAL "ARM64EC")
    if (MSVC)
        link_directories("$ENV{VCINSTALLDIR}/Tools/MSVC/$ENV{VCToolsVersion}/lib/ARM64EC")
        link_directories("$ENV{VCINSTALLDIR}/Tools/MSVC/$ENV{VCToolsVersion}/ATLMFC/lib/ARM64EC")
        link_libraries(softintrin.lib)
        add_compile_options("/bigobj")
    endif()
endif()

file(GLOB onnxruntime_common_src CONFIGURE_DEPENDS
    ${onnxruntime_common_src_patterns}
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_common_src})

onnxruntime_add_static_library(onnxruntime_common ${onnxruntime_common_src})

if (onnxruntime_USE_CUDA)
  # Some files, like the provider_bridge_ort include files that depend on cuda headers
  target_include_directories(onnxruntime_common PUBLIC ${onnxruntime_CUDA_HOME}/include)
endif()

if (onnxruntime_USE_TELEMETRY)
  set_target_properties(onnxruntime_common PROPERTIES COMPILE_FLAGS "/FI${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows/TraceLoggingConfigPrivate.h")
endif()

if (onnxruntime_USE_MIMALLOC_STL_ALLOCATOR OR onnxruntime_USE_MIMALLOC_ARENA_ALLOCATOR)
    if(onnxruntime_USE_CUDA OR onnxruntime_USE_OPENVINO)
        message(WARNING "Ignoring directive to use mimalloc on unimplemented targets")
    elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
        # Some of the non-windows targets see strange runtime failures
        message(WARNING "Ignoring request to link to mimalloc - only windows supported")
    else()
        include(external/mimalloc.cmake)
        list(APPEND onnxruntime_EXTERNAL_LIBRARIES mimalloc-static)
        list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES mimalloc-static)
        target_link_libraries(onnxruntime_common mimalloc-static)
    endif()
endif()

onnxruntime_add_include_to_target(onnxruntime_common date_interface wil)
target_include_directories(onnxruntime_common
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS}
    # propagate include directories of dependencies that are part of public interface
    PUBLIC
        ${OPTIONAL_LITE_INCLUDE_DIR})

target_link_libraries(onnxruntime_common safeint_interface Boost::mp11)

if(NOT WIN32)
  target_include_directories(onnxruntime_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
endif()

if(NOT onnxruntime_USE_OPENMP)
  target_compile_definitions(onnxruntime_common PUBLIC EIGEN_USE_THREADS)
endif()
add_dependencies(onnxruntime_common ${onnxruntime_EXTERNAL_DEPENDENCIES})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/common  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
set_target_properties(onnxruntime_common PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_common PROPERTIES FOLDER "ONNXRuntime")

if(WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(onnxruntime_common PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
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
    target_link_libraries(onnxruntime_common rt)
  endif()
endif()

if (onnxruntime_WINML_NAMESPACE_OVERRIDE STREQUAL "Windows")
  target_compile_definitions(onnxruntime_common PRIVATE "BUILD_INBOX=1")
endif()

# check if we need to link against libatomic due to std::atomic usage by the threadpool code
# e.g. Raspberry Pi requires this
if (onnxruntime_LINK_LIBATOMIC)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES atomic)
endif()

if(APPLE)
  target_link_libraries(onnxruntime_common "-framework Foundation")
endif()
