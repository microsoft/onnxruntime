if (onnxruntime_USE_FLASH_ATTENTION)
  include(FetchContent)
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    GIT_TAG        8b42e751c63ba219755c8ed91af5f6ec1ecc1ee6
  )

  FetchContent_GetProperties(cutlass)
  if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
  endif()
endif()
