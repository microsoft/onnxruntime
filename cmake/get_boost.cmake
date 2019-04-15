set(BOOST_REQUESTED_VERSION 1.69.0 CACHE STRING "")
set(BOOST_SHA1 8f32d4617390d1c2d16f26a27ab60d97807b35440d45891fa340fc2648b04406 CACHE STRING "")
set(BOOST_USE_STATIC_LIBS false CACHE BOOL "")

set(BOOST_COMPONENTS program_options system thread)

if(WIN32)
  message(FATAL_ERROR "Windows not currently supported")
endif()

if(BOOST_USE_STATIC_LIBS)
  set(LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
  set(LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(LIBRARY_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
  set(LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

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

  message(STATUS "Adding Boost components")
  include(ExternalProject)
  ExternalProject_Add(
      Boost
      URL http://dl.bintray.com/boostorg/release/${BOOST_REQUESTED_VERSION}/source/boost_${BOOST_REQUESTED_VERSION_UNDERSCORE}.tar.bz2
      URL_HASH SHA256=${BOOST_SHA1}
      DOWNLOAD_DIR ${BOOST_ROOT_DIR}
      SOURCE_DIR ${BOOST_ROOT_DIR}
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ./bootstrap.sh --prefix=${BOOST_ROOT_DIR}
      BUILD_COMMAND ./b2 install ${BOOST_MAYBE_STATIC} --prefix=${BOOST_ROOT_DIR} variant=${VARIANT} ${BOOST_COMPONENTS_FOR_BUILD}
      BUILD_IN_SOURCE true
      INSTALL_COMMAND ""
      INSTALL_DIR ${BOOST_ROOT_DIR}
      LOG_BUILD ON
  )

  ExternalProject_Get_Property(Boost INSTALL_DIR)
  set(Boost_INCLUDE_DIR ${INSTALL_DIR}/include)

  macro(libraries_to_fullpath varname)
    set(${varname})
    foreach(component ${BOOST_COMPONENTS})
      list(APPEND ${varname} ${INSTALL_DIR}/lib/${LIBRARY_PREFIX}boost_${component}${LIBRARY_SUFFIX})
    endforeach()
  endmacro()

  libraries_to_fullpath(Boost_LIBRARIES)
  mark_as_advanced(Boost_LIBRARIES Boost_INCLUDE_DIR)
endmacro()

DOWNLOAD_BOOST()
