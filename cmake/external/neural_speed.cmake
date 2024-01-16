set(NEURAL_SPEED_URL https://github.com/intel/neural-speed.git)
set(NEURAL_SPEED_TAG 44c05babb9bf01c8b26f8697526f380677cb6800) # kernel-only release v0.1

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND onnxruntime_target_platform STREQUAL "x86_64")
  set(USE_NEURAL_SPEED TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND onnxruntime_target_platform STREQUAL "x64")
  set(USE_NEURAL_SPEED TRUE)
endif()

if(USE_NEURAL_SPEED)
  FetchContent_Declare(
      neural_speed
      GIT_REPOSITORY ${NEURAL_SPEED_URL}
      GIT_TAG        ${NEURAL_SPEED_TAG}
  )
  set(BTLA_USE_OPENMP OFF)
  FetchContent_MakeAvailable(neural_speed)
endif()
