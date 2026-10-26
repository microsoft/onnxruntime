# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.26)
project(onnxruntime_source_compat_cuda LANGUAGES CXX CUDA)

include(FetchContent)
find_package(CUDAToolkit 12.6 REQUIRED)

# Download headers and kernels without configuring the upstream build systems.
FetchContent_Declare(source_fa3
  URL https://codeload.github.com/sgl-project/sgl-flash-attn/tar.gz/f89bc2306632d1ec5f97b014dded4254f5b4a907
  URL_HASH SHA256=418b5681584dc3efff496a1cab5ffd58d2728d89dcfe0ea16e6985d6ef35c68c
  SOURCE_SUBDIR source_compat_headers_only)
FetchContent_Declare(source_cutlass
  URL https://codeload.github.com/NVIDIA/cutlass/tar.gz/7127592069c2fe01b041e174ba4345ef9b279671
  URL_HASH SHA256=12baa9c2bf22eaca065b5e55387c21bb4ac742992a6ecd2dd1d3b016dfd25a0a
  SOURCE_SUBDIR source_compat_headers_only)
FetchContent_Declare(source_flashinfer
  URL https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/5f3d1b3fc6e1ed8a79429986b3637802f1bd2b57
  URL_HASH SHA256=5313532c38904e64c6230ab109c8db8f52575f60ce53763784e899ef2ceec21b
  SOURCE_SUBDIR source_compat_headers_only)
FetchContent_MakeAvailable(source_fa3 source_cutlass source_flashinfer)

get_filename_component(ORT_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(ORT_HEADER_DIR "${ORT_SOURCE_ROOT}/include/onnxruntime/core/session" CACHE PATH
  "Public C/C++ headers matching the deployed ONNX Runtime")

add_library(onnxruntime_source_compat_cuda SHARED
  source_compat_cuda.cu
  "${source_fa3_SOURCE_DIR}/hopper/flash_prepare_scheduler.cu")
target_include_directories(onnxruntime_source_compat_cuda PRIVATE
  "${ORT_HEADER_DIR}"
  "${source_fa3_SOURCE_DIR}/hopper"
  "${source_cutlass_SOURCE_DIR}/include"
  "${source_flashinfer_SOURCE_DIR}/include")
target_compile_features(onnxruntime_source_compat_cuda PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(onnxruntime_source_compat_cuda PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda>")
target_compile_definitions(onnxruntime_source_compat_cuda PRIVATE
  FLASHATTENTION_DISABLE_BACKWARD FLASHATTENTION_DISABLE_DROPOUT FLASHATTENTION_DISABLE_SPLIT)
set_target_properties(onnxruntime_source_compat_cuda PROPERTIES
  CUDA_ARCHITECTURES 90a
  CUDA_RUNTIME_LIBRARY Static
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(onnxruntime_source_compat_cuda PRIVATE CUDA::cuda_driver)

install(TARGETS onnxruntime_source_compat_cuda LIBRARY DESTINATION lib)
install(FILES "${source_fa3_SOURCE_DIR}/LICENSE" DESTINATION share/source_compat_cuda
  RENAME flash-attention-LICENSE)
install(FILES "${source_cutlass_SOURCE_DIR}/LICENSE.txt" DESTINATION share/source_compat_cuda
  RENAME cutlass-LICENSE)
install(FILES "${source_flashinfer_SOURCE_DIR}/LICENSE" DESTINATION share/source_compat_cuda
  RENAME flashinfer-LICENSE)
