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
    "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry_sha256.h"
    "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry_sha256.cc"
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

    # Telemetry for non-Windows platforms (enabled by USE_TELEMETRY)
    if (onnxruntime_USE_TELEMETRY)
        list(APPEND onnxruntime_common_src_patterns
             "${ONNXRUNTIME_ROOT}/core/platform/posix/device_id.h"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/device_id.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry.h"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry_context.h"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry_no_throw.h"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry_sampling.h"
        )
    endif()

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
         "${ONNXRUNTIME_ROOT}/core/platform/linux/device_discovery.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/linux/pci_device_discovery.h")
elseif (APPLE)
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/apple/device_discovery.cc")
else()
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/device_discovery_default.cc")
endif()

# Raw /bigobj is a cl.exe option. Do not apply it to CUDA sources; nvcc treats a
# standalone /bigobj as an input file on Windows ARM64 CUDA 13.1.
set(onnxruntime_msvc_bigobj_compile_option
    "$<$<AND:$<NOT:$<COMPILE_LANGUAGE:ASM_MARMASM>>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/bigobj>")

if(onnxruntime_target_platform STREQUAL "ARM64EC")
    if (MSVC)
        link_directories("$ENV{VCINSTALLDIR}/Tools/MSVC/$ENV{VCToolsVersion}/lib/ARM64EC")
        link_directories("$ENV{VCINSTALLDIR}/Tools/MSVC/$ENV{VCToolsVersion}/ATLMFC/lib/ARM64EC")
        link_libraries(softintrin.lib)
        add_compile_options("${onnxruntime_msvc_bigobj_compile_option}")
    endif()
endif()

if(onnxruntime_target_platform STREQUAL "ARM64")
    if (MSVC)
        add_compile_options("${onnxruntime_msvc_bigobj_compile_option}")
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
  # windows/telemetry.cc's svchost service-name fallback uses CommandLineToArgvW (shell32), which is
  # only compiled on the desktop partition (guarded with WINAPI_PARTITION_DESKTOP there). Restrict the
  # explicit shell32 link to desktop Windows: GDK lists shell32.lib in nodefault_libs (excluded via
  # /NODEFAULTLIB), and non-desktop partitions (UWP/WindowsStore) neither use nor ship it.
  if(NOT GDK_PLATFORM AND NOT CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
    target_link_libraries(onnxruntime_common PRIVATE shell32)
    # shell32.dll statically imports user32.dll, which is unavailable under Win32k lockdown. Delay-load
    # shell32.dll so the load-time dependency on user32.dll is deferred until the call is actually made,
    # letting onnxruntime.dll load in lockdown processes.
    if(onnxruntime_ENABLE_DELAY_LOADING_WIN_DLLS)
      list(APPEND onnxruntime_DELAYLOAD_FLAGS "/DELAYLOAD:shell32.dll")
    endif()
  endif()
endif()

if(NOT WIN32 AND NOT APPLE AND NOT ANDROID AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set_source_files_properties(
      ${ONNXRUNTIME_ROOT}/core/common/spin_pause.cc
      PROPERTIES COMPILE_FLAGS "-mwaitpkg"
    )
endif()

if (onnxruntime_USE_TELEMETRY)
  if(WIN32)
    set(ONNXRUNTIME_TELEMETRY_CONFIG_HEADER
        "${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows/TraceLoggingConfigPrivate.h")
    if(EXISTS "${ONNXRUNTIME_TELEMETRY_CONFIG_HEADER}")
      set_target_properties(
        onnxruntime_common
        PROPERTIES COMPILE_FLAGS "/FI${ONNXRUNTIME_TELEMETRY_CONFIG_HEADER}")
    endif()
  else()
    target_compile_definitions(onnxruntime_common PRIVATE USE_POSIX_TELEMETRY)
    # Optional tenant-token override written into a generated header in the build tree (kept off the
    # compiler command line, so the token never appears in compile_commands.json or build logs). It may be
    # supplied either as -DONNXRUNTIME_TELEMETRY_TENANT_TOKEN=... or via an
    # ONNXRUNTIME_TELEMETRY_TENANT_TOKEN environment variable — the latter lets callers inject a token without
    # it ever appearing on any command line. When unset, telemetry.cc uses the encoded in-repo default.
    if(NOT ONNXRUNTIME_TELEMETRY_TENANT_TOKEN AND DEFINED ENV{ONNXRUNTIME_TELEMETRY_TENANT_TOKEN})
      set(ONNXRUNTIME_TELEMETRY_TENANT_TOKEN "$ENV{ONNXRUNTIME_TELEMETRY_TENANT_TOKEN}")
    endif()
    # Ignore an unexpanded build-system macro (e.g. the literal "$(ONNXRUNTIME_TELEMETRY_TENANT_TOKEN)")
    # so the build falls back to the in-repo default instead of embedding the macro text as a bogus token.
    if(ONNXRUNTIME_TELEMETRY_TENANT_TOKEN MATCHES "^\\$\\(")
      set(ONNXRUNTIME_TELEMETRY_TENANT_TOKEN "")
    endif()
    if(ONNXRUNTIME_TELEMETRY_TENANT_TOKEN)
      set(ONNXRUNTIME_TELEMETRY_TENANT_TOKEN_DEFINE "#define ORT_TELEMETRY_TENANT_TOKEN \"${ONNXRUNTIME_TELEMETRY_TENANT_TOKEN}\"")
    else()
      set(ONNXRUNTIME_TELEMETRY_TENANT_TOKEN_DEFINE "")
    endif()
    set(_ort_telemetry_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_telemetry")
    configure_file(
      "${REPO_ROOT}/cmake/onnxruntime_telemetry_tenant_token.h.in"
      "${_ort_telemetry_gen_dir}/onnxruntime_telemetry_tenant_token.h"
      @ONLY)
    target_include_directories(onnxruntime_common PRIVATE "${_ort_telemetry_gen_dir}")
  endif()
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

if(CPUINFO_SUPPORTED)
  # Link cpuinfo if supported
  onnxruntime_add_include_to_target(onnxruntime_common cpuinfo::cpuinfo)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES cpuinfo::cpuinfo)
endif()

# Link telemetry library (1DS SDK) for non-Windows platforms
if(onnxruntime_USE_TELEMETRY AND NOT WIN32)
  if(onnxruntime_TELEMETRY_USES_EXTERNAL_PACKAGE AND TARGET MSTelemetry::mat)
    # The vcpkg package target propagates its include
    # directories and transitive dependencies (curl/sqlite3/zlib/nlohmann-json), so no
    # manual include paths or system libraries are required here.
    target_link_libraries(onnxruntime_common PRIVATE MSTelemetry::mat)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES MSTelemetry::mat)
  elseif(TARGET mat)
    # Link mat directly. In a shared build its resolved dependency set is absorbed into
    # libonnxruntime; in a static build mat -- and the bundled static archives it links -- are shipped
    # and exported below so a downstream find_package(onnxruntime) resolves them.
    target_link_libraries(onnxruntime_common PRIVATE mat)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES mat)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET libcurl_static)
      # Prevent shared-library consumers from re-exporting the embedded transport symbols. This does
      # not namespace static symbols; static ORT consumers must not co-link another curl/mbedTLS copy.
      string(CONCAT _onnxruntime_telemetry_build_exclude_libs
        "LINKER:--exclude-libs="
        "$<TARGET_FILE_NAME:libcurl_static>:"
        "$<TARGET_FILE_NAME:mbedtls>:"
        "$<TARGET_FILE_NAME:mbedx509>:"
        "$<TARGET_FILE_NAME:mbedcrypto>:"
        "$<TARGET_FILE_NAME:everest>:"
        "$<TARGET_FILE_NAME:p256m>")
      string(CONCAT _onnxruntime_telemetry_install_exclude_libs
        "LINKER:--exclude-libs="
        "$<TARGET_FILE_NAME:onnxruntime::libcurl_static>:"
        "$<TARGET_FILE_NAME:onnxruntime::mbedtls>:"
        "$<TARGET_FILE_NAME:onnxruntime::mbedx509>:"
        "$<TARGET_FILE_NAME:onnxruntime::mbedcrypto>:"
        "$<TARGET_FILE_NAME:onnxruntime::everest>:"
        "$<TARGET_FILE_NAME:onnxruntime::p256m>")
      target_link_options(onnxruntime_common INTERFACE
        "$<BUILD_INTERFACE:${_onnxruntime_telemetry_build_exclude_libs}>"
        "$<INSTALL_INTERFACE:${_onnxruntime_telemetry_install_exclude_libs}>")
    endif()
    # mat propagates its public include dir as a normal (non-SYSTEM) include, so onnxruntime_common's
    # -Wall -Wextra -Werror would apply to the SDK's headers (they trip -Werror=unused-parameter in
    # NullObjects.hpp / LogManagerProvider.hpp). Re-add the SDK include dirs as SYSTEM to exempt them.
    if(DEFINED cpp_client_telemetry_SOURCE_DIR)
      target_include_directories(onnxruntime_common SYSTEM PRIVATE
        ${cpp_client_telemetry_SOURCE_DIR}/lib/include/public
        ${cpp_client_telemetry_SOURCE_DIR}/lib/include/mat
        ${cpp_client_telemetry_SOURCE_DIR}/lib
      )
    endif()
    # Platform-specific system libraries required only for the Apple static-package path.
    if(APPLE AND NOT onnxruntime_BUILD_SHARED_LIB)
      if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # mat already links the SDK's bundled sqlite3/zlib archives, so no system SQLite is needed here.
        # A bare sqlite3 name would reach Xcode as -framework SQLite3, which the iOS SDK does not provide.
        target_link_libraries(onnxruntime_common PRIVATE
          "-framework CoreFoundation"
          "-framework Security"
        )
      else()
        target_link_libraries(onnxruntime_common PRIVATE
          "-framework CoreFoundation"
          "-framework Security"
          z
          sqlite3
        )
      endif()
    endif()

    if (NOT onnxruntime_BUILD_SHARED_LIB)
      # Static package: ship mat and the static archives it links so the exported package is
      # self-contained. These targets are optional because their availability depends on platform.
      install(TARGETS mat EXPORT ${PROJECT_NAME}Targets
              ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
              LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
              FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
      foreach(_mat_bundled_dep
          sqlite3_bundled
          zlib_bundled
          libcurl_static
          mbedtls
          mbedx509
          mbedcrypto
          everest
          p256m)
        if(TARGET ${_mat_bundled_dep})
          install(TARGETS ${_mat_bundled_dep} EXPORT ${PROJECT_NAME}Targets
                  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()
      endforeach()
    endif()
  else()
    message(FATAL_ERROR "Telemetry enabled but no 1DS SDK target ('MSTelemetry::mat' or 'mat') was found")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Every supported Linux telemetry path uses static curl/mbedTLS. Select a readable CA bundle
    # at runtime instead of embedding a build-machine path in the curl configuration.
    target_compile_definitions(onnxruntime_common PRIVATE ORT_TELEMETRY_USES_STATIC_CURL)
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
