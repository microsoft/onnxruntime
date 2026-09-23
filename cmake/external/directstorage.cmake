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
