# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

# Pass to build
set(ABSL_PROPAGATE_CXX_STD 1)
set(BUILD_TESTING 0)
set(ABSL_BUILD_TESTING OFF)
set(ABSL_BUILD_TEST_HELPERS OFF)
set(ABSL_USE_EXTERNAL_GOOGLETEST ON)
if(Patch_FOUND AND WIN32)
  set(ABSL_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/abseil/absl_windows.patch)
else()
  set(ABSL_PATCH_COMMAND "")
endif()
if(WIN32 AND NOT Patch_FOUND)
  #see https://github.com/google/re2/issues/425 and https://github.com/google/re2/issues/436
  set(ABSL_ENABLE_INSTALL ON)
endif()
# NB! Advancing Abseil version changes its internal namespace,
# currently absl::lts_20230125 which affects abseil-cpp.natvis debugger
# visualization file, that must be adjusted accordingly, unless we eliminate
# that namespace at build time.
FetchContent_Declare(
    abseil_cpp
    URL ${DEP_URL_abseil_cpp}
    URL_HASH SHA1=${DEP_SHA1_abseil_cpp}
    PATCH_COMMAND ${ABSL_PATCH_COMMAND}
    FIND_PACKAGE_ARGS NAMES absl
)

onnxruntime_fetchcontent_makeavailable(abseil_cpp)
FetchContent_GetProperties(abseil_cpp)
set(ABSEIL_SOURCE_DIR ${abseil_cpp_SOURCE_DIR})
message(STATUS "Abseil source dir:" ${ABSEIL_SOURCE_DIR})

if (GDK_PLATFORM)
  # Abseil considers any partition that is NOT in the WINAPI_PARTITION_APP a viable platform
  # for Win32 symbolize code (which depends on dbghelp.lib); this logic should really be flipped
  # to only include partitions that are known to support it (e.g. DESKTOP). As a workaround we
  # tell Abseil to pretend we're building an APP.
  target_compile_definitions(absl_symbolize PRIVATE WINAPI_FAMILY=WINAPI_FAMILY_DESKTOP_APP)
endif()

# TODO: since multiple ORT's dependencies depend on Abseil, the list below would vary from version to version.
# We'd better to not manually manage the list.
set(ABSEIL_LIBS absl::base
absl::city
absl::core_headers
absl::fixed_array
absl::flags
absl::flat_hash_map
absl::flat_hash_set
absl::hash
absl::inlined_vector
absl::low_level_hash
absl::node_hash_map
absl::node_hash_set
absl::optional
absl::raw_hash_set
absl::raw_logging_internal
absl::span
absl::str_format
absl::strings
absl::synchronization
absl::throw_delegate
absl::time)
