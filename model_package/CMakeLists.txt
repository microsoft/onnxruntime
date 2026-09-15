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
  src/asset_hasher.cc
  src/authoring.cc
  src/commit_prune_validate.cc
  src/manifest_parser.cc
  src/model_package_impl.cc
  src/path_resolver.cc
  src/sha256.cc
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

install(FILES include/model_package_api.h include/model_package.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

if(MODEL_PACKAGE_BUILD_TESTS)
  enable_testing()
  add_executable(test_inspection tests/test_inspection.cc)
  target_link_libraries(test_inspection PRIVATE model_package nlohmann_json::nlohmann_json)
  target_include_directories(test_inspection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  add_test(NAME inspection COMMAND test_inspection)

  add_executable(test_asset_hashing tests/test_asset_hashing.cc)
  target_link_libraries(test_asset_hashing PRIVATE model_package nlohmann_json::nlohmann_json)
  target_include_directories(test_asset_hashing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  add_test(NAME asset_hashing COMMAND test_asset_hashing)

  add_executable(test_authoring tests/test_authoring.cc)
  target_link_libraries(test_authoring PRIVATE model_package nlohmann_json::nlohmann_json)
  target_include_directories(test_authoring PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  add_test(NAME authoring COMMAND test_authoring)

  add_executable(test_commit tests/test_commit.cc)
  target_link_libraries(test_commit PRIVATE model_package nlohmann_json::nlohmann_json)
  target_include_directories(test_commit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  add_test(NAME commit COMMAND test_commit)
endif()
