set(PATCH ${PROJECT_SOURCE_DIR}/patches/cutlass/fmha_o_strideM.patch)

include(FetchContent)
FetchContent_Declare(cutlass
  GIT_REPOSITORY https://github.com/nvidia/cutlass.git
  GIT_TAG        8b42e751c63ba219755c8ed91af5f6ec1ecc1ee6
  PATCH_COMMAND  git apply --reverse --check ${PATCH} || git apply --ignore-whitespace --whitespace=nowarn ${PATCH}
)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_Populate(cutlass)
endif()
