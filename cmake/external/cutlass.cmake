include(FetchContent)
onnxruntime_fetchcontent_declare(
  cutlass
  URL ${DEP_URL_cutlass}
  URL_HASH SHA1=${DEP_SHA1_cutlass}
  EXCLUDE_FROM_ALL
  PATCH_COMMAND ${Patch_EXECUTABLE} --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/cutlass/cutlass_4.4.2.patch
)

# We only consume CUTLASS as a header-only dependency. Avoid FetchContent_MakeAvailable here
# because CUTLASS ships its own CMakeLists.txt that adds many test/example/tool targets and
# may override compile flags. Calling FetchContent_Populate downloads the source tree without
# invoking add_subdirectory.
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  if(POLICY CMP0169)
    # CMake >= 3.30 deprecates the single-argument form of FetchContent_Populate. Keep using
    # the OLD policy locally so we can populate without inviting CUTLASS targets into the build.
    cmake_policy(PUSH)
    cmake_policy(SET CMP0169 OLD)
    FetchContent_Populate(cutlass)
    cmake_policy(POP)
  else()
    FetchContent_Populate(cutlass)
  endif()
endif()
