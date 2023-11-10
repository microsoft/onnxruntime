# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

# Pass to build
set(ABSL_PROPAGATE_CXX_STD 1)
set(BUILD_TESTING 0)

if(Patch_FOUND AND WIN32)
  set(ABSL_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/abseil/absl_windows.patch)
else()
  set(ABSL_PATCH_COMMAND "")
endif()

# NB! Advancing Abseil version changes its internal namespace,
# currently absl::lts_20211102 which affects abseil-cpp.natvis debugger
# visualization file, that must be adjusted accordingly, unless we eliminate
# that namespace at build time.
FetchContent_Declare(
    abseil_cpp
    URL ${DEP_URL_abseil_cpp}
    URL_HASH SHA1=${DEP_SHA1_abseil_cpp}
    PATCH_COMMAND ${ABSL_PATCH_COMMAND}
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

if(NOT onnxruntime_DISABLE_ABSEIL)
  set(ABSEIL_LIBS absl::inlined_vector absl::flat_hash_set
    absl::flat_hash_map absl::node_hash_set absl::node_hash_map absl::base absl::throw_delegate absl::raw_hash_set
    absl::hash absl::city absl::low_level_hash absl::raw_logging_internal)
endif()