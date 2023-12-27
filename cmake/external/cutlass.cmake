if (onnxruntime_USE_CUTLASS)
  include(FetchContent)
  FetchContent_Declare(
    cutlass
    URL ${DEP_URL_cutlass}
    URL_HASH SHA1=${DEP_SHA1_cutlass}
  )

  FetchContent_GetProperties(cutlass)
  if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
  endif()
endif()
