# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(model_package VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

option(MODEL_PACKAGE_BUILD_SHARED "Build as shared library" ON)
option(MODEL_PACKAGE_BUILD_TESTS "Build tests" OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────

# Allow the user to provide a pre-existing nlohmann_json via NLOHMANN_JSON_DIR
# (pointing to the source tree with include/nlohmann/json.hpp).
# Falls back to FetchContent if not provided.
# When built as a subdirectory (e.g., within ORT), the target may already exist.
if(TARGET nlohmann_json::nlohmann_json)
  # Already available from parent project — nothing to do.
elseif(NLOHMANN_JSON_DIR)
  # Use the provided directory directly.
  add_library(nlohmann_json_header_only INTERFACE)
  target_include_directories(nlohmann_json_header_only INTERFACE
    "${NLOHMANN_JSON_DIR}/include"
    "${NLOHMANN_JSON_DIR}/single_include"
  )
  add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json_header_only)
else()
  find_package(nlohmann_json QUIET)
  if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      nlohmann_json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
      URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757f0f1b0b77571b1c626ade15f15d6f0b7b16d31
    )
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Library target
# ─────────────────────────────────────────────────────────────────────────────

set(MODEL_PACKAGE_SOURCES
  src/api.cc
  src/parser.cc
)

if(MODEL_PACKAGE_BUILD_SHARED)
  add_library(model_package SHARED ${MODEL_PACKAGE_SOURCES})
  target_compile_definitions(model_package PRIVATE MODEL_PACKAGE_DLL_EXPORT)
else()
  add_library(model_package STATIC ${MODEL_PACKAGE_SOURCES})
endif()

target_include_directories(model_package
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(model_package PRIVATE nlohmann_json::nlohmann_json)

set_target_properties(model_package PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

# ─────────────────────────────────────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────────────────────────────────────

include(GNUInstallDirs)

install(TARGETS model_package
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES include/model_package_api.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
