# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

# Pass to build
set(XNNPACK_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/googlexnnpack/Emscripten.patch)

FetchContent_Declare(
  googlexnnpack
  URL https://github.com/google/XNNPACK/archive/510b8e0f9af1891588c52bef64248cb72f04c30e.zip
  URL_HASH SHA1=6ef8eed37410c59cbba4fa9decba810422cece99
  PATCH_COMMAND ${XNNPACK_PATCH_COMMAND}
)
FetchContent_MakeAvailable(googlexnnpack)
