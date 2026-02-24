# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

# Pass to build
set(ABSL_PROPAGATE_CXX_STD 1)
set(BUILD_TESTING 0)
set(ABSL_BUILD_TESTING OFF)
set(ABSL_BUILD_TEST_HELPERS OFF)
set(ABSL_USE_EXTERNAL_GOOGLETEST ON)

# Both abseil and xnnpack create a target called memory, which
# results in a duplicate target if ABSL_ENABLE_INSTALL is on.
if (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
    set(ABSL_ENABLE_INSTALL ON)
endif()

if(Patch_FOUND)
  if (WIN32)
    set(ABSL_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/abseil/absl_windows.patch &&
                           ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/abseil/absl_cuda_warnings.patch)
  else()
    set(ABSL_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/abseil/absl_cuda_warnings.patch)
  endif()
else()
  set(ABSL_PATCH_COMMAND "")
endif()

# NB! Advancing Abseil version changes its internal namespace,
# currently absl::lts_20250512 which affects abseil-cpp.natvis debugger
# visualization file, that must be adjusted accordingly, unless we eliminate
# that namespace at build time.
onnxruntime_fetchcontent_declare(
    abseil_cpp
    URL ${DEP_URL_abseil_cpp}
    URL_HASH SHA1=${DEP_SHA1_abseil_cpp}
    EXCLUDE_FROM_ALL
    PATCH_COMMAND ${ABSL_PATCH_COMMAND}
    FIND_PACKAGE_ARGS 20250814 NAMES absl
)

onnxruntime_fetchcontent_makeavailable(abseil_cpp)
FetchContent_GetProperties(abseil_cpp)
if(abseil_cpp_SOURCE_DIR)
  set(ABSEIL_SOURCE_DIR ${abseil_cpp_SOURCE_DIR})
endif()

# abseil_cpp_SOURCE_DIR is non-empty if we build it from source
message(STATUS "Abseil source dir:" ${ABSEIL_SOURCE_DIR})
# abseil_cpp_VERSION  is non-empty if we find a preinstalled ABSL
if(abseil_cpp_VERSION)
  message(STATUS "Abseil version:" ${abseil_cpp_VERSION})
endif()
if (GDK_PLATFORM)
  # Abseil considers any partition that is NOT in the WINAPI_PARTITION_APP a viable platform
  # for Win32 symbolize code (which depends on dbghelp.lib); this logic should really be flipped
  # to only include partitions that are known to support it (e.g. DESKTOP). As a workaround we
  # tell Abseil to pretend we're building an APP.
  target_compile_definitions(absl_symbolize PRIVATE WINAPI_FAMILY=WINAPI_FAMILY_DESKTOP_APP)
endif()

# TODO: since multiple ORT's dependencies depend on Abseil, the list below would vary from version to version.
# We'd better to not manually manage the list.
# This list is generated using tools/python/resolve_absl_deps_dynamic.py for Abseil version 20250814.0
# The libraries are topologically sorted for correct static linking order.
set(ABSEIL_LIBS
absl::synchronization
absl::tracing_internal
absl::time
absl::time_zone
absl::civil_time
absl::symbolize
absl::demangle_internal
absl::demangle_rust
absl::stacktrace
absl::debugging_internal
absl::malloc_internal
absl::kernel_timeout_internal
absl::graphcycles_internal
absl::str_format
absl::str_format_internal
absl::flat_hash_set
absl::flat_hash_map
absl::algorithm_container
absl::raw_hash_map
absl::raw_hash_set
absl::prefetch
absl::hashtablez_sampler
absl::hashtable_debug_hooks
absl::hashtable_control_bytes
absl::hash_policy_traits
absl::common_policy_traits
absl::hash_container_defaults
absl::hash_function_defaults
absl::cord
absl::inlined_vector
absl::inlined_vector_internal
absl::span
absl::crc_cord_state
absl::crc32c
absl::cordz_update_tracker
absl::cordz_update_scope
absl::cordz_info
absl::cordz_functions
absl::cord_internal
absl::container_common
absl::container_memory
absl::hash
absl::variant
absl::optional
absl::strings
absl::charset
absl::strings_internal
absl::string_view
absl::int128
absl::compare
absl::function_ref
absl::any_invocable
absl::city
absl::bits
absl::endian
absl::flags
absl::fixed_array
absl::weakly_mixed_integer
absl::memory
absl::meta
absl::throw_delegate
absl::iterator_traits_internal
absl::algorithm
absl::compressed_tuple
absl::utility
absl::base
absl::type_traits
absl::spinlock_wait
absl::raw_logging_internal
absl::errno_saver
absl::nullability
absl::log_severity
absl::dynamic_annotations
absl::base_internal
absl::atomic_hook
absl::core_headers
absl::config
absl::absl_log
absl::log_internal_log_impl
absl::absl_check
absl::log_internal_check_impl)
