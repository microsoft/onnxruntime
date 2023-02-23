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
#We can not use FetchContent_MakeAvailable(microsoft_wil) at here, since their cmake file
#always executes install command without conditions.
FetchContent_Populate(microsoft_wil)
if(NOT wil_FOUND)
  add_library(WIL INTERFACE)
  add_library(WIL::WIL ALIAS WIL)

  # The interface's include directory.
  target_include_directories(WIL INTERFACE
    $<BUILD_INTERFACE:${microsoft_wil_SOURCE_DIR}/include>)
endif()