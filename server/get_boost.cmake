# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(BOOST_REQUESTED_VERSION 1.69.0 CACHE STRING "")
set(BOOST_SHA1 8f32d4617390d1c2d16f26a27ab60d97807b35440d45891fa340fc2648b04406 CACHE STRING "")
set(BOOST_USE_STATIC_LIBS true CACHE BOOL "")

set(BOOST_COMPONENTS program_options system thread)

# These components are only needed for Windows
if(WIN32)
  list(APPEND BOOST_COMPONENTS date_time regex)
endif()

# MSVC doesn't set these variables
if(WIN32)
  set(CMAKE_STATIC_LIBRARY_PREFIX lib)
  set(CMAKE_SHARED_LIBRARY_PREFIX lib)
endif()

# Set lib prefixes and suffixes for linking
if(BOOST_USE_STATIC_LIBS)
  set(LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
  set(LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(LIBRARY_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
  set(LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

# Create list of components in Boost format
foreach(component ${BOOST_COMPONENTS})
  list(APPEND BOOST_COMPONENTS_FOR_BUILD --with-${component})
endforeach()

set(BOOST_ROOT_DIR ${CMAKE_BINARY_DIR}/boost CACHE PATH "")

# TODO: let user give their own Boost installation
macro(DOWNLOAD_BOOST)
  if(NOT BOOST_REQUESTED_VERSION)
    message(FATAL_ERROR "BOOST_REQUESTED_VERSION is not defined.")
  endif()

  string(REPLACE "." "_" BOOST_REQUESTED_VERSION_UNDERSCORE ${BOOST_REQUESTED_VERSION})

  set(BOOST_MAYBE_STATIC)
  if(BOOST_USE_STATIC_LIBS)
    set(BOOST_MAYBE_STATIC "link=static")
  endif()

  set(VARIANT "release")
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(VARIANT "debug")
  endif()

  set(WINDOWS_B2_OPTIONS)
  set(WINDOWS_LIB_NAME_SCHEME)
  if(WIN32)
    set(BOOTSTRAP_FILE_TYPE "bat")
    set(WINDOWS_B2_OPTIONS "toolset=msvc-14.1" "architecture=x86" "address-model=64")
    set(WINDOWS_LIB_NAME_SCHEME "-vc141-mt-gd-x64-1_69")
  else()
    set(BOOTSTRAP_FILE_TYPE "sh")
  endif()

  foreach(component ${BOOST_COMPONENTS})
    list(APPEND BOOST_BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${LIBRARY_PREFIX}boost_${component}${WINDOWS_LIB_NAME_SCHEME}${LIBRARY_SUFFIX})
  endforeach()

  message(STATUS "Adding Boost components")
  include(ExternalProject)
  ExternalProject_Add(
      Boost
      URL https://boostorg.jfrog.io/artifactory/main/release/${BOOST_REQUESTED_VERSION}/source/boost_${BOOST_REQUESTED_VERSION_UNDERSCORE}.tar.bz2
      URL_HASH SHA256=${BOOST_SHA1}
      DOWNLOAD_DIR ${BOOST_ROOT_DIR}
      SOURCE_DIR ${BOOST_ROOT_DIR}
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ./bootstrap.${BOOTSTRAP_FILE_TYPE} --prefix=${BOOST_ROOT_DIR}
      BUILD_COMMAND ./b2 install ${BOOST_MAYBE_STATIC} --prefix=${BOOST_ROOT_DIR} variant=${VARIANT} ${WINDOWS_B2_OPTIONS} ${BOOST_COMPONENTS_FOR_BUILD}
      BUILD_IN_SOURCE true
      BUILD_BYPRODUCTS ${BOOST_BUILD_BYPRODUCTS}
      INSTALL_COMMAND ""
      INSTALL_DIR ${BOOST_ROOT_DIR}
  )

  # Set include folders
  ExternalProject_Get_Property(Boost INSTALL_DIR)
  set(Boost_INCLUDE_DIR ${INSTALL_DIR}/include)
  if(WIN32)
    set(Boost_INCLUDE_DIR ${INSTALL_DIR}/include/boost-1_69)
  endif()

  # Set libraries to link
  macro(libraries_to_fullpath varname)
    set(${varname})
    foreach(component ${BOOST_COMPONENTS})
      list(APPEND ${varname} ${INSTALL_DIR}/lib/${LIBRARY_PREFIX}boost_${component}${WINDOWS_LIB_NAME_SCHEME}${LIBRARY_SUFFIX})
    endforeach()
  endmacro()

  libraries_to_fullpath(Boost_LIBRARIES)
  mark_as_advanced(Boost_LIBRARIES Boost_INCLUDE_DIR)
endmacro()

DOWNLOAD_BOOST()
