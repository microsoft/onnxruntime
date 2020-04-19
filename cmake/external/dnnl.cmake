include (ExternalProject)

set(DNNL_URL https://github.com/intel/mkl-dnn.git)
# If DNNL_TAG is updated, check if MKLML_VERSION and platform.cmake.patch need to be updated.
set(DNNL_TAG v1.1.1)
set(MKLML_VERSION 2019.0.5.20190502)

if(WIN32)
  set(MKLML_OS_VERSION_STR "win")
  set(MKLML_FILE_EXTENSION "zip")
  set(DNNL_SHARED_LIB dnnl.dll)
  set(DNNL_IMPORT_LIB dnnl.lib)
  if(onnxruntime_USE_MKLML)
    # Windows-only updated MKLML binary which contains fix for thread cleanup hang.
    set(MKLML_VERSION 2020.0.20190813)
    set(MKLML_SHARED_LIB mklml.dll)
    set(MKLML_IMPORT_LIB mklml.lib)
    set(IOMP5MD_SHARED_LIB libiomp5md.dll)
    set(IOMP5MD_IMPORT_LIB libiomp5md.lib)
  endif()
else()
  set(MKLML_FILE_EXTENSION "tgz")
  if (APPLE)
    set(DNNL_SHARED_LIB libdnnl.1.dylib)
    set(MKLML_OS_VERSION_STR "mac")
  else()
    set(DNNL_SHARED_LIB libdnnl.so.1)
    set(MKLML_OS_VERSION_STR "lnx")
  endif()
  if(onnxruntime_USE_MKLML)
    set(MKLML_SHARED_LIB libmklml_intel.so)
    set(IOMP5MD_SHARED_LIB libiomp5.so)
  endif()
endif()

if (onnxruntime_USE_MKLML)
  set(MKLDNN_VERSION_SHORT v0.20)
  set(MKLML_URL https://github.com/intel/mkl-dnn/releases/download/${MKLDNN_VERSION_SHORT}/mklml_${MKLML_OS_VERSION_STR}_${MKLML_VERSION}.${MKLML_FILE_EXTENSION})

  ExternalProject_Add(project_mklml
    PREFIX mklml
    URL ${MKLML_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""  )

  set(MKML_DIR ${CMAKE_CURRENT_BINARY_DIR}/mklml/src/project_mklml)
  set(MKLML_INCLUDE_DIR "${MKML_DIR}/include")
  set(MKLML_LIB_DIR "${MKML_DIR}/lib")
  link_directories(${MKLML_LIB_DIR})
endif()

if (onnxruntime_USE_DNNL)
  set(DNNL_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/dnnl/src/dnnl/src)
  set(DNNL_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/dnnl/install)
  set(DNNL_LIB_DIR ${DNNL_INSTALL}/${CMAKE_INSTALL_LIBDIR})
  if(WIN32)
    set(DNNL_DLL_PATH ${DNNL_INSTALL}/${CMAKE_INSTALL_BINDIR}/${DNNL_SHARED_LIB})
  else()
    set(DNNL_DLL_PATH ${DNNL_LIB_DIR}/${DNNL_SHARED_LIB})
  endif()
  set(DNNL_INCLUDE_DIR ${DNNL_INSTALL}/include)
  set(DNNL_CMAKE_EXTRA_ARGS)
  # set(DNNL_PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/mkldnn/constexpr.patch)
  # discard prior changes due to patching in mkldnn source to unblock incremental builds.
  # set(MKLDNN_PATCH_DISCARD_COMMAND cd ${DNNL_SOURCE} && git checkout -- .)
  # if(NOT onnxruntime_BUILD_FOR_NATIVE_MACHINE)
    # pre-v1.0
    # list(APPEND DNNL_CMAKE_EXTRA_ARGS "-DARCH_OPT_FLAGS=")
    # v1.0
    # list(APPEND DNNL_CMAKE_EXTRA_ARGS "-DDNNL_ARCH_OPT_FLAGS=")
  # endif()
  ExternalProject_Add(project_dnnl
    PREFIX dnnl
    GIT_REPOSITORY ${DNNL_URL}
    GIT_TAG ${DNNL_TAG}
    # PATCH_COMMAND ${MKLDNN_PATCH_DISCARD_COMMAND} COMMAND ${DNNL_PATCH_COMMAND}
    SOURCE_DIR ${DNNL_SOURCE}
    CMAKE_ARGS -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${DNNL_INSTALL}
  )
  link_directories(${DNNL_LIB_DIR})
  #if (onnxruntime_USE_MKLML)
  #  add_dependencies(project_dnnl project_mklml)
  #endif()
endif()
