include (ExternalProject)

set(MKLDNN_URL https://github.com/intel/mkl-dnn.git)
set(MKLDNN_TAG v1.0.2)

if(WIN32)
  set(MKLDNN_SHARED_LIB mkldnn.dll)
  set(MKLDNN_IMPORT_LIB mkldnn.lib)
  if(onnxruntime_USE_MKLML)
    set(IOMP5MD_SHARED_LIB libiomp5md.dll)
    set(IOMP5MD_IMPORT_LIB libiomp5md.lib)
  endif()
else()
  if (APPLE)
    set(MKLDNN_SHARED_LIB libmkldnn.0.dylib)
  else()
    set(MKLDNN_SHARED_LIB libmkldnn.so.0)
  endif()
  if(onnxruntime_USE_MKLML)
    set(IOMP5MD_SHARED_LIB libiomp5.so)
  endif()
endif()

if (onnxruntime_USE_MKLML)
  set(MKLDNN_VERSION_SHORT v1.0)

  ExternalProject_Add(project_mklml
    PREFIX mklml
    URL ${MKLML_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""  )
endif()

if (onnxruntime_USE_MKLDNN)
  set(MKLDNN_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/src/mkl-dnn/src)
  set(MKLDNN_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/install)
  set(MKLDNN_LIB_DIR ${MKLDNN_INSTALL}/${CMAKE_INSTALL_LIBDIR})
  if(WIN32)
    set(MKLDNN_DLL_PATH ${MKLDNN_INSTALL}/${CMAKE_INSTALL_BINDIR}/${MKLDNN_SHARED_LIB})
  else()
    set(MKLDNN_DLL_PATH ${MKLDNN_LIB_DIR}/${MKLDNN_SHARED_LIB})
  endif()
  set(MKLDNN_INCLUDE_DIR ${MKLDNN_INSTALL}/include)
  set(MKLDNN_CMAKE_EXTRA_ARGS)
  if(NOT onnxruntime_BUILD_FOR_NATIVE_MACHINE)
    # pre-v1.0
    list(APPEND MKLDNN_CMAKE_EXTRA_ARGS "-DARCH_OPT_FLAGS=")
    # v1.0
    list(APPEND MKLDNN_CMAKE_EXTRA_ARGS "-DMKLDNN_ARCH_OPT_FLAGS=")
  endif()
  ExternalProject_Add(project_mkldnn
    PREFIX mkl-dnn
    GIT_REPOSITORY ${MKLDNN_URL}
    GIT_TAG ${MKLDNN_TAG}
    SOURCE_DIR ${MKLDNN_SOURCE}
    CMAKE_ARGS -DMKLDNN_PRODUCT_BUILD_MODE=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL} -DMKLROOT=${MKML_DIR} ${MKLDNN_CMAKE_EXTRA_ARGS}
  )
  link_directories(${MKLDNN_LIB_DIR})
  if (onnxruntime_USE_MKLML)
    add_dependencies(project_mkldnn project_mklml)
  endif()
endif()
