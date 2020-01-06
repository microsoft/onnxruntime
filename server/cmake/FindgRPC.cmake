# ~~~
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ~~~

#[=======================================================================[.rst:
FindgRPC
--------

Locate and configure the ``gRPC`` library.

The following variables can be set and are optional:

``gRPC_DEBUG``
  Show debug messages.
``gRPC_USE_STATIC_LIBS``
  Set to ON to force the use of the static libraries.
  Default is OFF.

Defines the following variables:

``gRPC_FOUND``
  Found the gRPC library
``gRPC_VERSION``
  Version of package found.

The following ``IMPORTED`` targets are also defined:

``gRPC::grpc++``
  The gRPC C++ library.
``gRPC::grpc``
  The gRPC C core library.
``gRPC::cpp_plugin``
  The C++ plugin for the Protobuf protoc compiler.

The following cache variables are also available to set or use:

Example:

.. code-block:: cmake

  find_package(gRPC REQUIRED)
  add_executable(bar bar.cc)
  target_link_libraries(bar PRIVATE gRPC::grpc++)

#]=======================================================================]

if (gRPC_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "gRPC_USE_STATIC_LIBS = ${gRPC_USE_STATIC_LIBS}"
                   " gRPC_FOUND = ${gRPC_FOUND}")
endif ()

# gRPC always requires Thread support.
find_package(Threads REQUIRED)

# Load the module to find protobuf with proper targets. Do not use
# `find_package()` because we (have to) install this module in non-standard
# locations.
include(${CMAKE_CURRENT_LIST_DIR}/FindProtobufWithTargets.cmake)

# The gRPC::grpc_cpp_plugin target is sometimes defined, but without a
# IMPORTED_LOCATION
function (_grpc_fix_grpc_cpp_plugin_target)
    # The target may already exist, do not create it again if it does.
    if (NOT TARGET gRPC::grpc_cpp_plugin)
        add_executable(gRPC::grpc_cpp_plugin IMPORTED)
    endif ()
    get_target_property(_gRPC_CPP_PLUGIN_EXECUTABLE gRPC::grpc_cpp_plugin
                        IMPORTED_LOCATION)
    if (gRPC_DEBUG)
        message(
            STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "LOCATION=${_gRPC_CPP_PLUGIN_EXECUTABLE}")
    endif ()
    # Even if the target exists, gRPC CMake support files do not define the
    # executable for the imported target (at least they do not in v1.19.1), so
    # we need to define it ourselves.
    if (NOT _gRPC_CPP_PLUGIN_EXECUTABLE)
        find_program(_gRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin
                     DOC "The gRPC C++ plugin for protoc")
        mark_as_advanced(_gRPC_CPP_PLUGIN_EXECUTABLE)
        if (_gRPC_CPP_PLUGIN_EXECUTABLE)
            set_property(
                TARGET gRPC::grpc_cpp_plugin
                PROPERTY IMPORTED_LOCATION ${_gRPC_CPP_PLUGIN_EXECUTABLE})
        else ()
            set(gRPC_FOUND "grpc_cpp_plugin-NOTFOUND")
        endif ()
    endif ()
endfunction ()

# The gRPC::* targets sometimes lack the right definitions to compile cleanly on
# WIN32
function (_grpc_fix_grpc_target_definitions)
    # Including gRPC headers without this definition results in a build error.
    if (WIN32)
        set_property(
            TARGET gRPC::grpc
            APPEND
            PROPERTY INTERFACE_COMPILE_DEFINITIONS _WIN32_WINNT=0x600)
        set_property(
            TARGET gRPC::grpc++
            APPEND
            PROPERTY INTERFACE_COMPILE_DEFINITIONS _WIN32_WINNT=0x600)
    endif ()
endfunction ()

# First try to use the `gRPCConfig.cmake` or `grpc-config.cmake` file if it was
# installed. This is common on systems (or package managers) where gRPC was
# compiled and installed with `CMake`.
find_package(gRPC NO_MODULE QUIET)

if (gRPC_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "NO_MODULE result gRPC_FOUND = ${gRPC_FOUND}")
endif ()

if (gRPC_FOUND)
    _grpc_fix_grpc_cpp_plugin_target()
    _grpc_fix_grpc_target_definitions()
    return()
endif ()

include(SelectLibraryConfigurations)

# Internal function: search for normal library as well as a debug one if the
# debug one is specified also include debug/optimized keywords in *_LIBRARIES
# variable
function (_gRPC_find_library name filename)
    if (${name}_LIBRARY)
        # Use result recorded by a previous call.
        return()
    else ()
        find_library(${name}_LIBRARY_RELEASE NAMES ${filename})
        mark_as_advanced(${name}_LIBRARY_RELEASE)

        find_library(${name}_LIBRARY_DEBUG NAMES ${filename}d ${filename})
        mark_as_advanced(${name}_LIBRARY_DEBUG)

        select_library_configurations(${name})

        if (gRPC_DEBUG)
            message(
                STATUS
                    "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                    "${name} ${filename} RELEASE=${${name}_LIBRARY}"
                    " DEBUG=${${name}_LIBRARY_DEBUG} DEFAULT=${${name}_LIBRARY}"
            )
        endif ()

        set(${name}_LIBRARY
            "${${name}_LIBRARY}"
            PARENT_SCOPE)
    endif ()
endfunction ()

#
# Main
#

# Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
if (_gRPC_USE_STATIC_LIBS)
    set(_gRPC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if (WIN32)
        set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    else ()
        set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    endif ()
endif ()

_grpc_find_library(_gRPC_grpc grpc)
_grpc_find_library(_gRPC_grpc++ grpc++)

if (NOT _gRPC_INCLUDE_DIR)
    find_path(_gRPC_INCLUDE_DIR grpcpp/grpcpp.h)
    mark_as_advanced(_gRPC_INCLUDE_DIR)
endif ()

if (gRPC_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   " _gRPC_grpc_LIBRARY = ${_gRPC_grpc_LIBRARY}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   " _gRPC_grpc++_LIBRARY = ${_gRPC_grpc++_LIBRARY}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   " _gRPC_INCLUDE_DIR = ${_gRPC_INCLUDE_DIR}")
endif ()

if (_gRPC_grpc_LIBRARY)
    if (NOT TARGET gRPC::grpc)
        add_library(gRPC::grpc UNKNOWN IMPORTED)
        set_target_properties(
            gRPC::grpc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                  "${_gRPC_INCLUDE_DIR}")
        if (EXISTS "${_gRPC_grpc_LIBRARY}")
            set_target_properties(gRPC::grpc PROPERTIES IMPORTED_LOCATION
                                                        "${_gRPC_grpc_LIBRARY}")
        endif ()
        if (EXISTS "${_gRPC_grpc_LIBRARY_RELEASE}")
            set_property(
                TARGET gRPC::grpc
                APPEND
                PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(
                gRPC::grpc PROPERTIES IMPORTED_LOCATION_RELEASE
                                      "${_gRPC_grpc_LIBRARY_RELEASE}")
        endif ()
        if (EXISTS "${_gRPC_grpc_LIBRARY_DEBUG}")
            set_property(
                TARGET gRPC::grpc
                APPEND
                PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(
                gRPC::grpc PROPERTIES IMPORTED_LOCATION_DEBUG
                                      "${_gRPC_grpc_LIBRARY_DEBUG}")
        endif ()
        set_property(
            TARGET gRPC::grpc
            APPEND
            PROPERTY INTERFACE_LINK_LIBRARIES protobuf::libprotobuf
                     Threads::Threads)
    endif ()
endif ()

if (_gRPC_grpc++_LIBRARY)
    if (NOT TARGET gRPC::grpc++)
        add_library(gRPC::grpc++ UNKNOWN IMPORTED)
        set_target_properties(
            gRPC::grpc++ PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                    "${_gRPC++_INCLUDE_DIR}")
        if (EXISTS "${_gRPC_grpc++_LIBRARY}")
            set_target_properties(
                gRPC::grpc++ PROPERTIES IMPORTED_LOCATION
                                        "${_gRPC_grpc++_LIBRARY}")
        endif ()
        if (EXISTS "${_gRPC_grpc++_LIBRARY_RELEASE}")
            set_property(
                TARGET gRPC::grpc++
                APPEND
                PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(
                gRPC::grpc++ PROPERTIES IMPORTED_LOCATION_RELEASE
                                        "${_gRPC_grpc++_LIBRARY_RELEASE}")
        endif ()
        if (EXISTS "${_gRPC_grpc++_LIBRARY_DEBUG}")
            set_property(
                TARGET gRPC::grpc++
                APPEND
                PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(
                gRPC::grpc++ PROPERTIES IMPORTED_LOCATION_DEBUG
                                        "${_gRPC_grpc++_LIBRARY_DEBUG}")
        endif ()
        set_property(
            TARGET gRPC::grpc++
            APPEND
            PROPERTY INTERFACE_LINK_LIBRARIES gRPC::grpc protobuf::libprotobuf
                     Threads::Threads)
        if (CMAKE_VERSION VERSION_GREATER 3.8)
            # gRPC++ requires C++11, but only CMake-3.8 introduced a target
            # compiler feature to meet that requirement.
            set_property(
                TARGET gRPC::grpc++
                APPEND
                PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_11)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            # CMake 3.5 is still alive and kicking in some older distros, use
            # the compiler-specific versions in these cases.
            set_property(
                TARGET gRPC::grpc++
                APPEND
                PROPERTY INTERFACE_COMPILE_OPTIONS "-std=c++11")
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set_property(
                TARGET gRPC::grpc++
                APPEND
                PROPERTY INTERFACE_COMPILE_OPTIONS "-std=c++11")
        else ()
            message(
                WARNING
                    "gRPC::grpc++ requires C++11, but this module"
                    " (${CMAKE_CURRENT_LIST_FILE})"
                    " cannot enable it for the library target in your CMake and"
                    " compiler versions. You need to enable C++11 in the"
                    " CMakeLists.txt for your project. Consider filing a bug"
                    " so we can fix this problem.")
        endif ()
    endif ()
endif ()

# Restore original find library prefixes
if (_gRPC_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${_gRPC_ORIG_FIND_LIBRARY_PREFIXES}")
endif ()

file(
    WRITE "${CMAKE_BINARY_DIR}/get_gRPC_version.cc"
    [====[
#include <grpcpp/grpcpp.h>
#include <iostream>
int main() {
  std::cout << grpc::Version(); // no newline to simplify CMake module
  return 0;
}
        ]====])

try_run(
    _gRPC_GET_VERSION_STATUS
    _gRPC_GET_VERSION_COMPILE_STATUS
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/get_gRPC_version.cc"
    LINK_LIBRARIES
    gRPC::grpc++
    gRPC::grpc
    COMPILE_OUTPUT_VARIABLE _gRPC_GET_VERSION_COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE gRPC_VERSION)

file(REMOVE "${CMAKE_BINARY_DIR}/get_gRPC_version.cc")

_grpc_fix_grpc_cpp_plugin_target()

if (gRPC_DEBUG)
    foreach (
        _var
        _gRPC_CPP_PLUGIN_EXECUTABLE
        _gRPC_VERSION_RAW
        _gRPC_GET_VERSION_STATUS
        _gRPC_GET_VERSION_COMPILE_STATUS
        _gRPC_GET_VERSION_COMPILE_OUTPUT
        _gRPC_grpc_LIBRARY
        _gRPC_grpc++_LIBRARY
        _gRPC_INCLUDE_DIR)
        message(
            STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "${_var} = ${${_var}}")
    endforeach ()
    unset(_var)
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(gRPC REQUIRED_VARS _gRPC_grpc_LIBRARY
                                  _gRPC_INCLUDE_DIR VERSION_VAR gRPC_VERSION)
