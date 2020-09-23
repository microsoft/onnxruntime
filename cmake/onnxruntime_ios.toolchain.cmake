# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(CMAKE_SYSTEM_NAME iOS)
if (NOT DEFINED CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM)
  set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED NO)
endif()

# Set Xcode property for SDKROOT as well if Xcode generator is used
if(CMAKE_GENERATOR MATCHES "Xcode")
  # If user did not specify the SDK root to use, then query xcodebuild for it.
  execute_process(COMMAND xcodebuild -version -sdk ${SDK_NAME} Path
      OUTPUT_VARIABLE CMAKE_OSX_SYSROOT_INT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT DEFINED CMAKE_OSX_SYSROOT_INT AND NOT DEFINED CMAKE_OSX_SYSROOT)
    message(SEND_ERROR "Please make sure that Xcode is installed and that the toolchain"
    "is pointing to the correct path. Please run:"
    "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    "and see if that fixes the problem for you.")
    message(FATAL_ERROR "Invalid CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT} "
    "does not exist.")
  elseif(DEFINED CMAKE_OSX_SYSROOT_INT)
    set(CMAKE_OSX_SYSROOT "${CMAKE_OSX_SYSROOT_INT}" CACHE INTERNAL "")
  endif()

  set(CMAKE_OSX_SYSROOT "${SDK_NAME}" CACHE INTERNAL "")
  if(NOT DEFINED CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM)
    set(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "123456789A" CACHE INTERNAL "")
  endif()
endif()
