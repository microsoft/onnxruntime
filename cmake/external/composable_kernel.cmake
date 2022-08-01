set(composable_kernel_URL https://github.com/ROCmSoftwarePlatform/composable_kernel.git)
set(composable_kernel_TAG 85978e0201bb94bf6e59b325e1f5f19266845d08) # 2022-07-22 11:52:10 -0700

set(BUILD_DEV OFF)
set(PATCH ${PROJECT_SOURCE_DIR}/patches/composable_kernel/Fix_Clang_Build.patch)

include(FetchContent)
FetchContent_Declare(composable_kernel
  GIT_REPOSITORY ${composable_kernel_URL}
  GIT_TAG        ${composable_kernel_TAG}
  PATCH_COMMAND  git apply --reverse --check ${PATCH} || git apply --ignore-space-change --ignore-whitespace ${PATCH}
)

FetchContent_MakeAvailable(composable_kernel)

add_library(onnxruntime_composable_kernel_includes INTERFACE)
target_include_directories(onnxruntime_composable_kernel_includes INTERFACE
  ${composable_kernel_SOURCE_DIR}/include
  ${composable_kernel_SOURCE_DIR}/library/include)
