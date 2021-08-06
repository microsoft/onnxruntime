include (ExternalProject)

set(JEMALLOC_URL https://github.com/jemalloc/jemalloc/releases/download/4.1.1/jemalloc-4.1.1.tar.bz2)
set(JEMALLOC_BUILD ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc)
set(JEMALLOC_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/install)

if(NOT WIN32)
  set(JEMALLOC_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/install/lib/libjemalloc_pic.a)
else()
  message( FATAL_ERROR "Jemalloc is not supported on Windows." )
endif()

ExternalProject_Add(jemalloc
      PREFIX jemalloc
      URL ${JEMALLOC_URL}
      INSTALL_DIR ${JEMALLOC_INSTALL}
      DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
      BUILD_COMMAND $(MAKE)
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND $(MAKE) install
      CONFIGURE_COMMAND
          ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc/configure
          --prefix=${JEMALLOC_INSTALL}
  )
