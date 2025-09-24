include(FetchContent)
onnxruntime_fetchcontent_declare(
  cutlass
  URL ${DEP_URL_cutlass}
  URL_HASH SHA1=${DEP_SHA1_cutlass}
  EXCLUDE_FROM_ALL
  PATCH_COMMAND ${Patch_EXECUTABLE} --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/cutlass/fix_kMinComputeCapability_linkage.patch
)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_MakeAvailable(cutlass)
endif()
