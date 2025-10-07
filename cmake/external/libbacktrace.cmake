# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Only build libbacktrace in Debug mode (for stack trace support)
if (NOT WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  option(onnxruntime_USE_LIBBACKTRACE "Build and use libbacktrace" ON)
else()
  set(onnxruntime_USE_LIBBACKTRACE OFF)
endif()

if (NOT WIN32 AND onnxruntime_USE_LIBBACKTRACE)
  message(STATUS "Configuring libbacktrace (Debug only)...")

  onnxruntime_fetchcontent_declare(
    libbacktrace
    URL ${DEP_URL_libbacktrace}
    URL_HASH SHA1=${DEP_SHA1_libbacktrace}
    EXCLUDE_FROM_ALL
  )

  FetchContent_GetProperties(libbacktrace)
  if (NOT libbacktrace_POPULATED)
    FetchContent_MakeAvailable(libbacktrace)
  endif()

  set(LIBBACKTRACE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace-install)
  set(LIBBACKTRACE_STATIC_LIB "${LIBBACKTRACE_INSTALL_DIR}/lib/libbacktrace.a")

  ExternalProject_Add(libbacktrace_ep
    SOURCE_DIR        ${libbacktrace_SOURCE_DIR}
    INSTALL_DIR       ${LIBBACKTRACE_INSTALL_DIR}
    CONFIGURE_COMMAND ${libbacktrace_SOURCE_DIR}/configure --prefix=<INSTALL_DIR>
    BUILD_COMMAND     make -j
    INSTALL_COMMAND   make install
    UPDATE_DISCONNECTED 1
    FOLDER "External"
    BUILD_BYPRODUCTS  ${LIBBACKTRACE_STATIC_LIB}     # Important for Ninja
  )

  # Imported target representing the built static library
  add_library(libbacktrace STATIC IMPORTED)
  set_target_properties(libbacktrace PROPERTIES
    IMPORTED_LOCATION "${LIBBACKTRACE_STATIC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBBACKTRACE_INSTALL_DIR}/include"
  )
  add_dependencies(libbacktrace libbacktrace_ep)
  add_library(libbacktrace::libbacktrace ALIAS libbacktrace)

  set(HAVE_LIBBACKTRACE TRUE CACHE BOOL "Whether libbacktrace is available")

else()
  message(STATUS "libbacktrace disabled (non-Debug build or Windows).")
  set(HAVE_LIBBACKTRACE FALSE CACHE BOOL "Whether libbacktrace is available")
endif()
