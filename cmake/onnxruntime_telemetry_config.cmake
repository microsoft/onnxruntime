# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Append dependencies needed by the installed static package, before importing its targets.
if(NOT onnxruntime_USE_1DS_TELEMETRY OR onnxruntime_BUILD_SHARED_LIB)
  return()
endif()

if(onnxruntime_TELEMETRY_USES_EXTERNAL_PACKAGE)
  string(APPEND PROJECT_CONFIG_CONTENT "find_dependency(MSTelemetry CONFIG)\n")
  return()
endif()

if(APPLE)
  string(APPEND PROJECT_CONFIG_CONTENT
    "if(NOT TARGET MSTelemetry::sqlite_dependency)\n\
    add_library(MSTelemetry::sqlite_dependency INTERFACE IMPORTED)\n\
    set_property(TARGET MSTelemetry::sqlite_dependency PROPERTY INTERFACE_LINK_LIBRARIES sqlite3)\n\
    endif()\n\
    if(NOT TARGET MSTelemetry::zlib_dependency)\n\
    add_library(MSTelemetry::zlib_dependency INTERFACE IMPORTED)\n\
    set_property(TARGET MSTelemetry::zlib_dependency PROPERTY INTERFACE_LINK_LIBRARIES z)\n\
    endif()\n")
elseif(NOT WIN32)
  # The SDK publicly links Threads on Linux and Android, including the Java transport.
  string(APPEND PROJECT_CONFIG_CONTENT "find_dependency(Threads)\n")
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    if(TARGET libcurl_static)
      # mat's install interface names the SDK wrapper, not the exported curl archive.
      set(_ort_installed_curl_target onnxruntime::libcurl_static)
    else()
      string(APPEND PROJECT_CONFIG_CONTENT "find_dependency(CURL)\n")
      set(_ort_installed_curl_target CURL::libcurl)
    endif()
    string(APPEND PROJECT_CONFIG_CONTENT
      "if(NOT TARGET MSTelemetry::curl_dependency)\n\
      add_library(MSTelemetry::curl_dependency INTERFACE IMPORTED)\n\
      set_property(TARGET MSTelemetry::curl_dependency PROPERTY INTERFACE_LINK_LIBRARIES ${_ort_installed_curl_target})\n\
      endif()\n")
    unset(_ort_installed_curl_target)
  endif()
endif()
