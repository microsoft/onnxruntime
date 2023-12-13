include (ExternalProject)

set(DNNL_URL https://github.com/oneapi-src/onednn.git)
# If DNNL_TAG is updated, check if MKLML_VERSION and platform.cmake.patch need to be updated.
set(DNNL_TAG v3.0)

if(WIN32)
  set(DNNL_SHARED_LIB dnnl.dll)
  set(DNNL_IMPORT_LIB dnnl.lib)
else()
  if (APPLE)
    set(DNNL_SHARED_LIB libdnnl.3.dylib)
  else()
    set(DNNL_SHARED_LIB libdnnl.so.3)
  endif()  
endif()

if(onnxruntime_USE_DNNL AND onnxruntime_DNNL_GPU_RUNTIME STREQUAL "ocl" AND onnxruntime_DNNL_OPENCL_ROOT STREQUAL "")
  message(FATAL_ERROR "--dnnl_opencl_root required")
elseif(onnxruntime_USE_DNNL AND onnxruntime_DNNL_GPU_RUNTIME STREQUAL "" AND NOT (onnxruntime_DNNL_OPENCL_ROOT STREQUAL ""))
  message(FATAL_ERROR "--dnnl_gpu_runtime required")
elseif(onnxruntime_USE_DNNL AND onnxruntime_DNNL_GPU_RUNTIME STREQUAL "ocl" AND NOT (onnxruntime_DNNL_OPENCL_ROOT STREQUAL ""))
  file(TO_CMAKE_PATH ${onnxruntime_DNNL_OPENCL_ROOT} onnxruntime_DNNL_OPENCL_ROOT)
  set(DNNL_OCL_INCLUDE_DIR ${onnxruntime_DNNL_OPENCL_ROOT}/include)
  set(DNNL_GPU_CMAKE_ARGS "-DDNNL_GPU_RUNTIME=OCL " "-DOPENCLROOT=${onnxruntime_DNNL_OPENCL_ROOT}")
endif()

if(onnxruntime_USE_DNNL AND onnxruntime_DNNL_AARCH64_RUNTIME STREQUAL "acl" AND onnxruntime_DNNL_ACL_ROOT STREQUAL "")
  message(FATAL_ERROR "--dnnl_acl_root required")
elseif(onnxruntime_USE_DNNL AND onnxruntime_DNNL_AARCH64_RUNTIME STREQUAL "" AND NOT (onnxruntime_DNNL_ACL_ROOT STREQUAL ""))
  message(FATAL_ERROR "--dnnl_aarch64_runtime required")
elseif(onnxruntime_USE_DNNL AND onnxruntime_DNNL_AARCH64_RUNTIME STREQUAL "acl" AND NOT (onnxruntime_DNNL_ACL_ROOT STREQUAL ""))
  file(TO_CMAKE_PATH ${onnxruntime_DNNL_ACL_ROOT} onnxruntime_DNNL_ACL_ROOT)
  set(ACL_INCLUDE_DIR ${onnxruntime_DNNL_ACL_ROOT}/arm_compute)
  set(DNNL_AARCH64_CMAKE_ARGS "-DDNNL_AARCH64_USE_ACL=ON")
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
    CMAKE_ARGS -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_CONCURRENT_EXEC=ON -DDNNL_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${DNNL_INSTALL} ${DNNL_GPU_CMAKE_ARGS} ${DNNL_AARCH64_CMAKE_ARGS}
  )
  link_directories(${DNNL_LIB_DIR})
endif()
