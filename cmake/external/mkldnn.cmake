include (ExternalProject)

set(MKLDNN_URL https://github.com/intel/mkl-dnn.git)
# If MKLDNN_TAG is updated, check if MKLML_VERSION and platform.cmake.patch need to be updated.
set(MKLDNN_TAG v0.17.1)
set(MKLML_VERSION 2019.0.1.20180928)

if(WIN32)
  set(MKLML_OS_VERSION_STR "win")
  set(MKLML_FILE_EXTENSION "zip")
  set(MKLDNN_SHARED_LIB mkldnn.dll)
  set(MKLDNN_IMPORT_LIB mkldnn.lib)
  if(onnxruntime_USE_MKLML)
    set(MKLML_SHARED_LIB mklml.dll)
    set(MKLML_IMPORT_LIB mklml.lib)
    set(IOMP5MD_SHARED_LIB libiomp5md.dll)
    set(IOMP5MD_IMPORT_LIB libiomp5md.lib)
  endif()
else()
  set(MKLML_FILE_EXTENSION "tgz")
  if (APPLE)
    set(MKLDNN_SHARED_LIB libmkldnn.0.dylib)
    set(MKLML_OS_VERSION_STR "mac")
  else()
    set(MKLDNN_SHARED_LIB libmkldnn.so.0)
    set(MKLML_OS_VERSION_STR "lnx")
  endif()
  if(onnxruntime_USE_MKLML)
    set(MKLML_SHARED_LIB libmklml_intel.so)
    set(IOMP5MD_SHARED_LIB libiomp5.so)
  endif()
endif()

if (onnxruntime_USE_MKLML)
  set(MKLML_URL https://github.com/intel/mkl-dnn/releases/download/${MKLDNN_TAG}/mklml_${MKLML_OS_VERSION_STR}_${MKLML_VERSION}.${MKLML_FILE_EXTENSION})

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

if (onnxruntime_USE_MKLDNN)
  set(MKLDNN_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/src/mkl-dnn/src)
  set(MKLDNN_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/install)
  set(MKLDNN_LIB_DIR ${MKLDNN_INSTALL}/lib)
  set(MKLDNN_INCLUDE_DIR ${MKLDNN_INSTALL}/include)

  set(MKLDNN_PATCH_COMMAND1 git apply ${CMAKE_SOURCE_DIR}/patches/mkldnn/platform.cmake.patch)
  # discard prior changes due to patching in mkldnn source to unblock incremental builds.
  set(MKLDNN_PATCH_DISCARD_COMMAND cd ${MKLDNN_SOURCE} && git checkout -- .)

  ExternalProject_Add(project_mkldnn
    PREFIX mkl-dnn
    GIT_REPOSITORY ${MKLDNN_URL}
    GIT_TAG ${MKLDNN_TAG}
    PATCH_COMMAND ${MKLDNN_PATCH_DISCARD_COMMAND} COMMAND ${MKLDNN_PATCH_COMMAND1}
    SOURCE_DIR ${MKLDNN_SOURCE}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL} -DMKLROOT=${MKML_DIR}
  )
  link_directories(${MKLDNN_LIB_DIR})
  if (onnxruntime_USE_MKLML)
    add_dependencies(project_mkldnn project_mklml)
  endif()
endif()
