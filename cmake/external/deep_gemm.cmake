# DeepGEMM is consumed as a header-only dependency. Keep it independent of the
# project's Torch/pybind wrapper and pin the exact source revision used to select
# the AOT DSV4 configurations.
include(FetchContent)
onnxruntime_fetchcontent_declare(
  deep_gemm
  URL ${DEP_URL_deep_gemm}
  URL_HASH SHA1=${DEP_SHA1_deep_gemm}
  EXCLUDE_FROM_ALL
)

FetchContent_GetProperties(deep_gemm)
if(NOT deep_gemm_POPULATED)
  if(POLICY CMP0169)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0169 OLD)
    FetchContent_Populate(deep_gemm)
    cmake_policy(POP)
  else()
    FetchContent_Populate(deep_gemm)
  endif()
endif()