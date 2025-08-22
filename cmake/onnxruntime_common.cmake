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
    "${ONNXRUNTIME_ROOT}/core/platform/check_intel.h"
    "${ONNXRUNTIME_ROOT}/core/platform/check_intel.cc"
    "${ONNXRUNTIME_ROOT}/core/platform/device_discovery.h"
    "${ONNXRUNTIME_ROOT}/core/platform/device_discovery_common.cc"
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
    "${ONNXRUNTIME_ROOT}/core/quantization/*.h"
    "${ONNXRUNTIME_ROOT}/core/quantization/*.cc"
)

if(WIN32)
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/windows/debug_alloc.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/debug_alloc.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/dll_load_error.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/dll_load_error.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/env_time.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/env.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/env.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/hardware_core_enumerator.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/hardware_core_enumerator.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/stacktrace.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/telemetry.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/telemetry.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/logging/*.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/logging/*.cc"
    )

else()
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/posix/env_time.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/posix/env.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/posix/stacktrace.cc"
    )

    # logging files
    if (onnxruntime_USE_SYSLOG)
        list(APPEND onnxruntime_common_src_patterns
            "${ONNXRUNTIME_ROOT}/core/platform/posix/logging/*.h"
            "${ONNXRUNTIME_ROOT}/core/platform/posix/logging/*.cc"
        )
    endif()

    if (ANDROID)
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

# platform-specific device discovery files
if (WIN32)
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/windows/device_discovery.cc")
elseif (LINUX)
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/linux/device_discovery.cc")
elseif (APPLE)
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/apple/device_discovery.cc")
else()
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/device_discovery_default.cc")
endif()

if(onnxruntime_target_platform STREQUAL "ARM64EC")
    if (MSVC)
        link_directories("$ENV{VCINSTALLDIR}/Tools/MSVC/$ENV{VCToolsVersion}/lib/ARM64EC")
        link_directories("$ENV{VCINSTALLDIR}/Tools/MSVC/$ENV{VCToolsVersion}/ATLMFC/lib/ARM64EC")
        link_libraries(softintrin.lib)
        add_compile_options("$<$<NOT:$<COMPILE_LANGUAGE:ASM_MARMASM>>:/bigobj>")
    endif()
endif()

if(onnxruntime_target_platform STREQUAL "ARM64")
    if (MSVC)
        add_compile_options("$<$<NOT:$<COMPILE_LANGUAGE:ASM_MARMASM>>:/bigobj>")
    endif()
endif()

file(GLOB onnxruntime_common_src CONFIGURE_DEPENDS
    ${onnxruntime_common_src_patterns}
    )

# Remove new/delete intercept. To deal with memory leaks
# Use either non-mimalloc build OR use mimalloc built-in features.
if(WIN32 AND onnxruntime_USE_MIMALLOC)
    list(REMOVE_ITEM onnxruntime_common_src
    "${ONNXRUNTIME_ROOT}/core/platform/windows/debug_alloc.cc"
    "${ONNXRUNTIME_ROOT}/core/platform/windows/debug_alloc.h")
endif()

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_common_src})

onnxruntime_add_static_library(onnxruntime_common ${onnxruntime_common_src})
if(WIN32)
  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set_property(TARGET onnxruntime_common PROPERTY CXX_STANDARD 23)
    target_compile_options(onnxruntime_common PRIVATE "/Zc:char8_t-")
  endif()
endif()

if(NOT WIN32 AND NOT APPLE AND NOT ANDROID AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set_source_files_properties(
      ${ONNXRUNTIME_ROOT}/core/common/spin_pause.cc
      PROPERTIES COMPILE_FLAGS "-mwaitpkg"
    )
endif()

if (onnxruntime_USE_TELEMETRY)
  set_target_properties(onnxruntime_common PROPERTIES COMPILE_FLAGS "/FI${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows/TraceLoggingConfigPrivate.h")
endif()
if (onnxruntime_USE_MIMALLOC)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES mimalloc-static)
  onnxruntime_add_static_library(onnxruntime_mimalloc_shim "${ONNXRUNTIME_ROOT}/core/platform/windows/mimalloc/mimalloc_overloads.cc")
  target_link_libraries(onnxruntime_mimalloc_shim PRIVATE mimalloc-static)
  target_link_libraries(onnxruntime_common PRIVATE onnxruntime_mimalloc_shim)
endif()

if (MSVC)
  set(ABSEIL_NATVIS_FILE "abseil-cpp.natvis")
  target_sources(
      onnxruntime_common
      INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/${ABSEIL_NATVIS_FILE}>)
endif()


if (MSVC)
    set(EIGEN_NATVIS_FILE ${eigen_SOURCE_DIR}/debug/msvc/eigen.natvis)
    if (EXISTS ${EIGEN_NATVIS_FILE})
      target_sources(
          onnxruntime_common
          INTERFACE $<BUILD_INTERFACE:${EIGEN_NATVIS_FILE}>)
    endif()
endif()

onnxruntime_add_include_to_target(onnxruntime_common date::date ${WIL_TARGET} Eigen3::Eigen)
target_include_directories(onnxruntime_common
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT}
    # propagate include directories of dependencies that are part of public interface
    PUBLIC
        ${OPTIONAL_LITE_INCLUDE_DIR})


target_link_libraries(onnxruntime_common PUBLIC safeint_interface ${GSL_TARGET} ${ABSEIL_LIBS} date::date)

add_dependencies(onnxruntime_common ${onnxruntime_EXTERNAL_DEPENDENCIES})

set_target_properties(onnxruntime_common PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_common PROPERTIES FOLDER "ONNXRuntime")


if (onnxruntime_WINML_NAMESPACE_OVERRIDE STREQUAL "Windows")
  target_compile_definitions(onnxruntime_common PRIVATE "BUILD_INBOX=1")
endif()

# check if we need to link against libatomic due to std::atomic usage by the threadpool code
# e.g. Raspberry Pi requires this
if (onnxruntime_LINK_LIBATOMIC)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES atomic)
endif()

if(APPLE)
  target_link_libraries(onnxruntime_common PRIVATE "-framework Foundation")
endif()

if(MSVC)
  if(onnxruntime_target_platform STREQUAL "ARM64")
    set(ARM64 TRUE)
  elseif (onnxruntime_target_platform STREQUAL "ARM")
    set(ARM TRUE)
  elseif(onnxruntime_target_platform STREQUAL "x64")
    set(X64 TRUE)
  elseif(onnxruntime_target_platform STREQUAL "x86")
    set(X86 TRUE)
  endif()
elseif(APPLE)
  if(CMAKE_OSX_ARCHITECTURES_LEN LESS_EQUAL 1)
    set(X64 TRUE)
  endif()
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
      set(ARM TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
      set(ARM64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
      set(X86_64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
      set(X86 TRUE)
    endif()
  else()
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} -dumpmachine
      OUTPUT_VARIABLE dumpmachine_output
      ERROR_QUIET
    )
    if(dumpmachine_output MATCHES "^arm64.*")
      set(ARM64 TRUE)
    elseif(dumpmachine_output MATCHES "^arm.*")
      set(ARM TRUE)
    elseif(dumpmachine_output MATCHES "^aarch64.*")
      set(ARM64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64.*")
      set(RISCV64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
      set(X86 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
      set(X86_64 TRUE)
    endif()
  endif()
endif()

if (RISCV64 OR ARM64 OR ARM OR X86 OR X64 OR X86_64)
    # Link cpuinfo if supported
    if (CPUINFO_SUPPORTED)
      onnxruntime_add_include_to_target(onnxruntime_common cpuinfo::cpuinfo)
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES cpuinfo::cpuinfo ${ONNXRUNTIME_CLOG_TARGET_NAME})
    endif()
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/common  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
  install(TARGETS onnxruntime_common EXPORT ${PROJECT_NAME}Targets
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
