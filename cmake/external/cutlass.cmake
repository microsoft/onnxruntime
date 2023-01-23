if (onnxruntime_USE_FLASH_ATTENTION)
  include(FetchContent)
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    GIT_TAG        66d9cddc832c1cdc2b30a8755274f7f74640cfe6
  )

  FetchContent_GetProperties(cutlass)
  if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
  endif()
endif()
