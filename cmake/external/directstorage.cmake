# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include_guard(GLOBAL)
onnxruntime_fetchcontent_declare(
  directstorage
  URL ${DEP_URL_directstorage}
  URL_HASH SHA1=${DEP_SHA1_directstorage}
  DOWNLOAD_NAME directstorage.zip
)
onnxruntime_fetchcontent_makeavailable(directstorage)

if(onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
  # CI deploys the SDK runtime only beside test executables, including when the SDK source is overridden.
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/directstorage-source-dir.txt" CONTENT "${directstorage_SOURCE_DIR}\n")
endif()
