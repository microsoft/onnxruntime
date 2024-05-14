# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set(WIL_BUILD_PACKAGING OFF CACHE BOOL "" FORCE)
set(WIL_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  microsoft_wil
  URL ${DEP_URL_microsoft_wil}
  URL_HASH SHA1=${DEP_SHA1_microsoft_wil}
  FIND_PACKAGE_ARGS NAMES wil
)

if(WIN32)
  onnxruntime_fetchcontent_makeavailable(microsoft_wil)
  set(WIL_TARGET "WIL::WIL")
endif()
