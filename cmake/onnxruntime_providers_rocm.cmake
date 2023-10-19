# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_definitions(-DUSE_ROCM=1)
  include(onnxruntime_rocm_hipify.cmake)

  list(APPEND CMAKE_PREFIX_PATH ${onnxruntime_ROCM_HOME})

  find_package(HIP)
  find_package(hiprand REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(MIOpen REQUIRED)
  find_package(hipfft REQUIRED)

  # MIOpen version
  if(NOT DEFINED ENV{MIOPEN_PATH})
    set(MIOPEN_PATH ${onnxruntime_ROCM_HOME})
  else()
    set(MIOPEN_PATH $ENV{MIOPEN_PATH})
  endif()
  find_path(MIOPEN_VERSION_H_PATH
    NAMES version.h
    HINTS
    ${MIOPEN_PATH}/include/miopen
    ${MIOPEN_PATH}/miopen/include)
  if (MIOPEN_VERSION_H_PATH-NOTFOUND)
    MESSAGE(FATAL_ERROR "miopen version.h not found")
  endif()
  MESSAGE(STATUS "Found miopen version.h at ${MIOPEN_VERSION_H_PATH}")

  file(READ ${MIOPEN_VERSION_H_PATH}/version.h MIOPEN_HEADER_CONTENTS)
        string(REGEX MATCH "define MIOPEN_VERSION_MAJOR * +([0-9]+)"
                                 MIOPEN_VERSION_MAJOR "${MIOPEN_HEADER_CONTENTS}")
        string(REGEX REPLACE "define MIOPEN_VERSION_MAJOR * +([0-9]+)" "\\1"
                                 MIOPEN_VERSION_MAJOR "${MIOPEN_VERSION_MAJOR}")
        string(REGEX MATCH "define MIOPEN_VERSION_MINOR * +([0-9]+)"
                                 MIOPEN_VERSION_MINOR "${MIOPEN_HEADER_CONTENTS}")
        string(REGEX REPLACE "define MIOPEN_VERSION_MINOR * +([0-9]+)" "\\1"
                                 MIOPEN_VERSION_MINOR "${MIOPEN_VERSION_MINOR}")
        string(REGEX MATCH "define MIOPEN_VERSION_PATCH * +([0-9]+)"
                                 MIOPEN_VERSION_PATCH "${MIOPEN_HEADER_CONTENTS}")
        string(REGEX REPLACE "define MIOPEN_VERSION_PATCH * +([0-9]+)" "\\1"
                                 MIOPEN_VERSION_PATCH "${MIOPEN_VERSION_PATCH}")
  set(MIOPEN_VERSION_DEV "${MIOPEN_VERSION_MAJOR}.${MIOPEN_VERSION_MINOR}.${MIOPEN_VERSION_PATCH}")
  math(EXPR MIOPEN_VERSION_DEV_INT "(${MIOPEN_VERSION_MAJOR}*10000) + (${MIOPEN_VERSION_MINOR}*100) + ${MIOPEN_VERSION_PATCH}")
  message("MIOPEN_VERSION_DEV: ${MIOPEN_VERSION_DEV}")
  message("MIOPEN_VERSION_DEV_INT:   ${MIOPEN_VERSION_DEV_INT}")
  add_definitions(-DMIOPEN_VERSION=${MIOPEN_VERSION_DEV_INT})

  find_library(RCCL_LIB rccl REQUIRED)
  find_library(ROCTRACER_LIB roctracer64 REQUIRED)
  set(ONNXRUNTIME_ROCM_LIBS roc::rocblas MIOpen hip::hipfft ${RCCL_LIB} ${ROCTRACER_LIB})

  file(GLOB_RECURSE onnxruntime_providers_rocm_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/rocm/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/rocm/*.cc"
  )

  # The shared_library files are in a separate list since they use precompiled headers, and the above files have them disabled.
  file(GLOB_RECURSE onnxruntime_providers_rocm_shared_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  file(GLOB_RECURSE onnxruntime_providers_rocm_cu_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/rocm/*.cu"
    "${ONNXRUNTIME_ROOT}/core/providers/rocm/*.cuh"
  )

  hipify("onnxruntime/core/providers" provider_excluded_files onnxruntime_providers_rocm_generated_cc_srcs onnxruntime_providers_rocm_generated_cu_srcs)

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_rocm_cc_srcs} ${onnxruntime_providers_rocm_shared_srcs} ${onnxruntime_providers_rocm_cu_srcs})
  set(onnxruntime_providers_rocm_src ${onnxruntime_providers_rocm_cc_srcs} ${onnxruntime_providers_rocm_shared_srcs} ${onnxruntime_providers_rocm_cu_srcs})
  list(APPEND onnxruntime_providers_rocm_src ${onnxruntime_providers_rocm_generated_cc_srcs} ${onnxruntime_providers_rocm_generated_cu_srcs})

  # disable contrib ops conditionally
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    hipify("onnxruntime/contrib_ops" contrib_ops_excluded_files onnxruntime_rocm_generated_contrib_ops_cc_srcs onnxruntime_rocm_generated_contrib_ops_cu_srcs)

    # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_rocm_contrib_ops_cc_srcs} ${onnxruntime_rocm_contrib_ops_cu_srcs})
    list(APPEND onnxruntime_providers_rocm_src ${onnxruntime_rocm_contrib_ops_cc_srcs} ${onnxruntime_rocm_contrib_ops_cu_srcs})
    list(APPEND onnxruntime_providers_rocm_src ${onnxruntime_rocm_generated_contrib_ops_cc_srcs} ${onnxruntime_rocm_generated_contrib_ops_cu_srcs})
  endif()

  if (onnxruntime_ENABLE_TRAINING_OPS)
    file(GLOB_RECURSE onnxruntime_rocm_training_ops_cc_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/training_ops/rocm/*.h"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/rocm/*.cc"
    )

    file(GLOB_RECURSE onnxruntime_rocm_training_ops_cu_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/training_ops/rocm/*.cu"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/rocm/*.cuh"
    )

    hipify("orttraining/orttraining/training_ops" training_ops_excluded_files onnxruntime_rocm_generated_training_ops_cc_srcs onnxruntime_rocm_generated_training_ops_cu_srcs)

    # NCCL is not support in Windows build
    if (WIN32 OR NOT onnxruntime_USE_NCCL)
      list(REMOVE_ITEM onnxruntime_rocm_generated_training_ops_cc_srcs
      "${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining/orttraining/training_ops/rocm/collective/nccl_common.cc"
      "${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining/orttraining/training_ops/rocm/collective/nccl_kernels.cc"
      "${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining/orttraining/training_ops/rocm/collective/megatron.cc"
      )
    endif()

    source_group(TREE ${ORTTRAINING_ROOT} FILES ${onnxruntime_rocm_training_ops_cc_srcs} ${onnxruntime_rocm_training_ops_cu_srcs})
    list(APPEND onnxruntime_providers_rocm_src ${onnxruntime_rocm_training_ops_cc_srcs} ${onnxruntime_rocm_training_ops_cu_srcs})
    list(APPEND onnxruntime_providers_rocm_src ${onnxruntime_rocm_generated_training_ops_cc_srcs} ${onnxruntime_rocm_generated_training_ops_cu_srcs})
  endif()

  auto_set_source_files_hip_language(${onnxruntime_providers_rocm_src})
  onnxruntime_add_shared_library_module(onnxruntime_providers_rocm ${onnxruntime_providers_rocm_src})
  target_compile_options(onnxruntime_providers_rocm PRIVATE -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1)

  if(NOT MSVC)
    target_compile_options(onnxruntime_providers_rocm PRIVATE -Wno-sign-compare)
    target_compile_options(onnxruntime_providers_rocm PRIVATE -Wno-unused-parameter)
    target_compile_options(onnxruntime_providers_rocm PRIVATE -Wno-undefined-var-template)
  endif()

  onnxruntime_add_include_to_target(onnxruntime_providers_rocm onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  if (onnxruntime_ENABLE_TRAINING_OPS)
    onnxruntime_add_include_to_target(onnxruntime_providers_rocm onnxruntime_training)
    target_link_libraries(onnxruntime_providers_rocm PRIVATE onnxruntime_training)
    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      onnxruntime_add_include_to_target(onnxruntime_providers_rocm Python::Module)
    endif()
  endif()

  add_custom_target(generate_hipified_files DEPENDS
    ${onnxruntime_providers_rocm_generated_cc_srcs}
    ${onnxruntime_providers_rocm_generated_cu_srcs}
    ${onnxruntime_rocm_generated_contrib_ops_cc_srcs}
    ${onnxruntime_rocm_generated_contrib_ops_cu_srcs}
    ${onnxruntime_rocm_generated_training_ops_cc_srcs}
    ${onnxruntime_rocm_generated_training_ops_cu_srcs})

  add_dependencies(onnxruntime_providers_rocm generate_hipified_files onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_link_libraries(onnxruntime_providers_rocm PRIVATE ${ONNXRUNTIME_ROCM_LIBS} ${ONNXRUNTIME_PROVIDERS_SHARED} ${ABSEIL_LIBS})
  target_include_directories(onnxruntime_providers_rocm SYSTEM
    PRIVATE
      ${ONNXRUNTIME_ROOT}
      ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime
      ${eigen_INCLUDE_DIRS}
    PUBLIC
      ${onnxruntime_ROCM_HOME}/include
      ${onnxruntime_ROCM_HOME}/include/roctracer)

  set_target_properties(onnxruntime_providers_rocm PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_rocm PROPERTIES FOLDER "ONNXRuntime")

  if (onnxruntime_ENABLE_TRAINING)
    target_include_directories(onnxruntime_providers_rocm PRIVATE ${ORTTRAINING_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining ${MPI_CXX_INCLUDE_DIRS})
    if(onnxruntime_USE_MPI)
      target_link_libraries(onnxruntime_providers_rocm PRIVATE ${MPI_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    endif()

    # RCCL is enabled by default for ROCM builds
    #if (onnxruntime_USE_NCCL)
    #  target_include_directories(onnxruntime_providers_rocm PRIVATE ${NCCL_INCLUDE_DIRS})
    #  target_link_libraries(onnxruntime_providers_rocm PRIVATE ${NCCL_LIBRARIES})
    #endif()
  endif()

  if (onnxruntime_USE_ROCBLAS_EXTENSION_API)
    target_compile_definitions(onnxruntime_providers_rocm PRIVATE USE_ROCBLAS_EXTENSION_API)
    target_compile_definitions(onnxruntime_providers_rocm PRIVATE ROCBLAS_NO_DEPRECATED_WARNINGS)
    target_compile_definitions(onnxruntime_providers_rocm PRIVATE ROCBLAS_BETA_FEATURES_API)
  endif()

  if (onnxruntime_USE_HIPBLASLT)
    find_package(hipblaslt REQUIRED)
    target_link_libraries(onnxruntime_providers_rocm PRIVATE roc::hipblaslt)
    target_compile_definitions(onnxruntime_providers_rocm PRIVATE USE_HIPBLASLT)
  endif()

  if (onnxruntime_USE_TRITON_KERNEL)
    # compile triton kernel, generate .a and .h files
    include(onnxruntime_compile_triton_kernel.cmake)
    compile_triton_kernel(triton_kernel_obj_file triton_kernel_header_dir)
    add_dependencies(onnxruntime_providers_rocm onnxruntime_triton_kernel)
    target_compile_definitions(onnxruntime_providers_rocm PRIVATE USE_TRITON_KERNEL)
    target_include_directories(onnxruntime_providers_rocm PRIVATE ${triton_kernel_header_dir})
    target_link_libraries(onnxruntime_providers_rocm PUBLIC -Wl,--whole-archive ${triton_kernel_obj_file} -Wl,--no-whole-archive)
  endif()

  if (onnxruntime_USE_COMPOSABLE_KERNEL)
    include(composable_kernel)
    target_link_libraries(onnxruntime_providers_rocm PRIVATE
      onnxruntime_composable_kernel_includes
      # Currently we shall not use composablekernels::device_operations, the target includes all conv dependencies, which
      # are extremely slow to compile. Instead, we only link all gemm related objects. See the following directory on
      # updating.
      # https://github.com/ROCmSoftwarePlatform/composable_kernel/tree/develop/library/src/tensor_operation_instance/gpu
      device_gemm_instance
      device_gemm_add_fastgelu_instance
      device_gemm_fastgelu_instance
      device_gemm_splitk_instance
      device_gemm_streamk_instance
      device_batched_gemm_instance
      device_softmax_instance
    )
    target_compile_definitions(onnxruntime_providers_rocm PRIVATE USE_COMPOSABLE_KERNEL)
  endif()

  if(UNIX)
    set_property(TARGET onnxruntime_providers_rocm APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/rocm/version_script.lds -Xlinker --gc-sections")
    target_link_libraries(onnxruntime_providers_rocm PRIVATE nsync::nsync_cpp)
  else()
    message(FATAL_ERROR "onnxruntime_providers_rocm unknown platform, need to specify shared library exports for it")
  endif()

  if (onnxruntime_ENABLE_ATEN)
    target_compile_definitions(onnxruntime_providers_rocm PRIVATE ENABLE_ATEN)
  endif()

  install(TARGETS onnxruntime_providers_rocm
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
