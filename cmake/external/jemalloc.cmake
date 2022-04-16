include (ExternalProject)

set(JEMALLOC_URL https://github.com/jemalloc/jemalloc.git)
set(JEMALLOC_TAG 4.1.1)
set(JEMALLOC_BUILD ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc)
set(JEMALLOC_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/install)

if(NOT WIN32)
  set(JEMALLOC_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/install/lib/libjemalloc_pic.a)
else()
  message( FATAL_ERROR "Jemalloc is not supported on Windows." )
endif()

ExternalProject_Add(jemalloc
      PREFIX jemalloc
      GIT_REPOSITORY ${JEMALLOC_URL}
      GIT_TAG ${JEMALLOC_TAG}
      INSTALL_DIR ${JEMALLOC_INSTALL}
      BUILD_COMMAND $(MAKE)
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND $(MAKE) install
      CONFIGURE_COMMAND
          ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc/configure
          --prefix=${JEMALLOC_INSTALL}
  )
