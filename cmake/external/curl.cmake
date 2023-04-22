# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(ExternalProject)

if (WIN32)

ExternalProject_Add(curl
    GIT_REPOSITORY https://github.com/curl/curl.git
    GIT_TAG curl-8_0_1
    PREFIX curl
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl-src
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl-build
    CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF -DCURL_ENABLE_SSL=ON -DCURL_STATICLIB=ON -DCURL_USE_SCHANNEL=ON
    INSTALL_COMMAND "")

else()

ExternalProject_Add(curl
    GIT_REPOSITORY https://github.com/curl/curl.git
    GIT_TAG curl-8_0_1
    PREFIX curl
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl-src
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl-build
    CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF -DCURL_ENABLE_SSL=OFF
    INSTALL_COMMAND "")

endif()

set(CURL_SRC ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl-src)
set(CURL_BIN ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl-build)
