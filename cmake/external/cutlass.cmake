if (onnxruntime_USE_FLASH_ATTENTION)
  include(FetchContent)
  FetchContent_Declare(
    cutlass
    URL ${DEP_URL_cutlass}
    URL_HASH SHA1=${DEP_SHA1_cutlass}
    PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/cutlass/cutlass.patch
  )

  FetchContent_GetProperties(cutlass)
  if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
  endif()
endif()
