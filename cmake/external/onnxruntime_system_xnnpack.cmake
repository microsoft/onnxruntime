# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT UNIX OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  message(FATAL_ERROR "System XNNPACK requires a Unix platform with shared libraries.")
endif()

find_path(XNNPACK_INCLUDE_DIR NAMES xnnpack.h REQUIRED)
find_path(PTHREADPOOL_INCLUDE_DIR NAMES pthreadpool.h REQUIRED)
block(SCOPE_FOR VARIABLES)
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  find_library(XNNPACK_LIBRARY NAMES XNNPACK REQUIRED)
  find_library(PTHREADPOOL_LIBRARY NAMES pthreadpool REQUIRED)
endblock()
foreach(_xnnpack_library IN ITEMS XNNPACK_LIBRARY PTHREADPOOL_LIBRARY)
  if("${${_xnnpack_library}}" MATCHES "\\.a$")
    message(FATAL_ERROR "${_xnnpack_library} must refer to a shared library for system XNNPACK.")
  endif()
endforeach()
find_package(Threads REQUIRED)

if(NOT TARGET onnxruntime::system_pthreadpool)
  add_library(onnxruntime::system_pthreadpool SHARED IMPORTED)
  set_target_properties(onnxruntime::system_pthreadpool PROPERTIES
    IMPORTED_LOCATION "${PTHREADPOOL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${PTHREADPOOL_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES Threads::Threads)
endif()
if(NOT TARGET onnxruntime::system_xnnpack)
  add_library(onnxruntime::system_xnnpack SHARED IMPORTED)
  set_target_properties(onnxruntime::system_xnnpack PROPERTIES
    IMPORTED_LOCATION "${XNNPACK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${XNNPACK_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES onnxruntime::system_pthreadpool)
endif()
