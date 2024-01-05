set(BTLA_URL https://github.com/intel/neural-speed.git)
set(BTLA_TAG 368ccbd2823e7ecef862d09e7b2385e6b2553081) # bestla v0.1

set(USE_NEURAL_SPEED FALSE)
if (onnxruntime_USE_NEURAL_SPEED)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND onnxruntime_target_platform STREQUAL "x86_64")
    set(USE_NEURAL_SPEED TRUE)
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND onnxruntime_target_platform STREQUAL "x64")
    set(USE_NEURAL_SPEED TRUE)
  endif()
  if(USE_NEURAL_SPEED)
    FetchContent_Declare(
        bestla
        GIT_REPOSITORY ${BTLA_URL}
        GIT_TAG        ${BTLA_TAG}
    )
    FetchContent_MakeAvailable(bestla)
    add_compile_definitions(MLAS_NEURAL_SPEED)
  endif()
endif()
