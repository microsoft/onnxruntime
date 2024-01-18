if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND onnxruntime_target_platform STREQUAL "x86_64")
  set(USE_NEURAL_SPEED TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND onnxruntime_target_platform STREQUAL "x64")
  set(USE_NEURAL_SPEED TRUE)
endif()

if(USE_NEURAL_SPEED)
  FetchContent_Declare(
      neural_speed
      URL https://github.com/intel/neural-speed/archive/refs/tags/bestlav0.1.1.zip
      URL_HASH SHA1=65b0f7a0d04f72f0d5a8d48af70f0366f2ab3939
  )
  set(BTLA_USE_OPENMP OFF)
  FetchContent_MakeAvailable(neural_speed)
  if(NOT neural_speed_POPULATED)
    FetchContent_Populate(neural_speed)
  endif()
endif()
