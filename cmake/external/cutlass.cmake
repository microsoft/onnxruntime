if (onnxruntime_USE_FLASH_ATTENTION)
  include(FetchContent)
  FetchContent_Declare(
    cutlass
    URL ${DEP_URL_cutlass}
    URL_HASH SHA1=${DEP_SHA1_cutlass}
  )

  onnxruntime_fetchcontent_makeavailable(cutlass)
  FetchContent_GetProperties(cutlass)
endif()
