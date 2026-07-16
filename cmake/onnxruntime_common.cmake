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

    # Telemetry for non-Windows platforms (enabled by USE_TELEMETRY)
    if (onnxruntime_USE_TELEMETRY)
        list(APPEND onnxruntime_common_src_patterns
             "${ONNXRUNTIME_ROOT}/core/platform/posix/device_id.h"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/device_id.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry.h"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/telemetry.cc"
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
  # windows/telemetry.cc's svchost service-name fallback uses CommandLineToArgvW (shell32), which is
  # only compiled on the desktop partition (guarded with WINAPI_PARTITION_DESKTOP there). Restrict the
  # explicit shell32 link to desktop Windows: GDK lists shell32.lib in nodefault_libs (excluded via
  # /NODEFAULTLIB), and non-desktop partitions (UWP/WindowsStore) neither use nor ship it.
  if(NOT GDK_PLATFORM AND NOT CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
    target_link_libraries(onnxruntime_common PRIVATE shell32)
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
    set_target_properties(onnxruntime_common PROPERTIES COMPILE_FLAGS "/FI${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows/TraceLoggingConfigPrivate.h")
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
  if(TARGET MSTelemetry::mat)
    # vcpkg port (cpp-client-telemetry): the imported target propagates its include
    # directories and transitive dependencies (curl/sqlite3/zlib/nlohmann-json), so no
    # manual include paths or system libraries are required here.
    target_link_libraries(onnxruntime_common PRIVATE MSTelemetry::mat)
  elseif(TARGET mat)
    # Link mat directly. In a shared build its resolved dependency set is absorbed into
    # libonnxruntime; in a static build mat -- and the bundled static archives it links -- are shipped
    # and exported below so a downstream find_package(onnxruntime) resolves them.
    target_link_libraries(onnxruntime_common PRIVATE mat)
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
        # iOS links the bundled sqlite3/zlib archives (shipped below); only the frameworks the SDK's
        # Apple PAL needs are added here.
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
      # Static package: ship mat and the bundled static archives it links so the exported package is
      # self-contained. The bundled deps are optional -- which exist depends on platform (the vendored
      # sqlite3/zlib built for Android, iOS, and non-Apple static via MATSDK_BUNDLE_VENDORED_DEPS) and
      # may fold into libmat.a in a future SDK version -- so install each only if its target exists.
      install(TARGETS mat EXPORT ${PROJECT_NAME}Targets
              ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
              LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
              FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
      foreach(_mat_bundled_dep sqlite3_bundled zlib_bundled)
        if(TARGET ${_mat_bundled_dep})
          install(TARGETS ${_mat_bundled_dep} EXPORT ${PROJECT_NAME}Targets
                  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()
      endforeach()
    endif()
  else()
    message(FATAL_ERROR "Telemetry enabled but no 1DS SDK target ('MSTelemetry::mat' or 'mat') was found")
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
