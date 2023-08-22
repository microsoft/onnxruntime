set(composable_kernel_URL https://github.com/ROCmSoftwarePlatform/composable_kernel.git)
set(composable_kernel_TAG d52ec01652b7d620386251db92455968d8d90bdc) # 2023-08-18 11:14:59 +0800

set(PATCH ${PROJECT_SOURCE_DIR}/patches/composable_kernel/Fix_Clang_Build.patch)

include(FetchContent)
FetchContent_Declare(composable_kernel
  GIT_REPOSITORY ${composable_kernel_URL}
  GIT_TAG        ${composable_kernel_TAG}
  PATCH_COMMAND  git apply --reverse --check ${PATCH} || git apply --ignore-space-change --ignore-whitespace ${PATCH}
)

FetchContent_GetProperties(composable_kernel)
if(NOT composable_kernel_POPULATED)
  FetchContent_Populate(composable_kernel)
  set(BUILD_DEV OFF CACHE BOOL "Disable -Weverything, otherwise, error: 'constexpr' specifier is incompatible with C++98 [-Werror,-Wc++98-compat]" FORCE)
  # Exclude i8 device gemm instances due to excessive long compilation time and not being used
  set(DTYPES fp32 fp16 bf16)
  set(INSTANCES_ONLY ON)
  add_subdirectory(${composable_kernel_SOURCE_DIR} ${composable_kernel_BINARY_DIR} EXCLUDE_FROM_ALL)

  add_library(onnxruntime_composable_kernel_includes INTERFACE)
  target_include_directories(onnxruntime_composable_kernel_includes INTERFACE
    ${composable_kernel_SOURCE_DIR}/include
    ${composable_kernel_SOURCE_DIR}/library/include)
  target_compile_definitions(onnxruntime_composable_kernel_includes INTERFACE __fp32__ __fp16__ __bf16__)
endif()
