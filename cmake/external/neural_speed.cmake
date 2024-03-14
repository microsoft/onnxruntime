if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND onnxruntime_target_platform STREQUAL "x86_64")
  set(USE_NEURAL_SPEED TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND onnxruntime_target_platform STREQUAL "x64")
  set(USE_NEURAL_SPEED TRUE)
endif()

if(USE_NEURAL_SPEED)
  FetchContent_Declare(
      neural_speed
      URL ${DEP_URL_neural_speed}
      URL_HASH SHA1=${DEP_SHA1_neural_speed}
      PATCH_COMMAND ${Patch_EXECUTABLE} -p1 < ${PROJECT_SOURCE_DIR}/patches/neural_speed/65b0f7a0d04f72f0d5a8d48af70f0366f2ab3939.patch
  )
  set(BTLA_USE_OPENMP OFF)
  onnxruntime_fetchcontent_makeavailable(neural_speed)
endif()
