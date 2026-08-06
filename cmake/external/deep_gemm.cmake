# DeepGEMM is consumed as a header-only dependency. Keep it independent of the
# project's Torch/pybind wrapper and pin the exact source revision used to select
# the AOT DSV4 configurations.
include(FetchContent)
FetchContent_Declare(
  deep_gemm
  GIT_REPOSITORY https://github.com/deepseek-ai/DeepGEMM.git
  GIT_TAG 559d79fb6994a58b8a15b4b93bf13ccc16edf247
  GIT_SHALLOW FALSE
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