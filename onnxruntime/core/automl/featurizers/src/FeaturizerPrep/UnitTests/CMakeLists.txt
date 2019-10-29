# ----------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
# ----------------------------------------------------------------------
cmake_minimum_required(VERSION 3.5.0)

project(Featurizer_UnitTests LANGUAGES CXX)

set(CMAKE_MODULE_PATH "$ENV{DEVELOPMENT_ENVIRONMENT_CMAKE_MODULE_PATH}")

if(NOT WIN32)
    string(REPLACE ":" ";" CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}")
    string(REPLACE ":" ";" _includes "$ENV{INCLUDE}")
    string(REPLACE ":" ";" _libs "$ENV{LIB}")
endif()

set(CppCommon_STATIC_CRT ON CACHE BOOL "" FORCE)
set(BoostCommon_HEADER_ONLY ON CACHE BOOL "" FORCE)

include(CppCommon)
include(BoostCommon)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

foreach(_test_name IN ITEMS
    Featurizer_UnitTest
    Traits_UnitTests
)
    add_executable(${_test_name} ${_test_name}.cpp)

    target_include_directories(${_test_name} PRIVATE ${_includes})
    target_link_directories(${_test_name} PRIVATE ${_libs})

    target_link_libraries(${_test_name} PRIVATE ${Boost_LIBRARIES})

    add_test(NAME ${_test_name} COMMAND ${_test_name} --success)
endforeach()
