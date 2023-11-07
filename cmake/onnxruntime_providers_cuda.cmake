# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  file(GLOB_RECURSE onnxruntime_providers_cuda_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
  )
  # Remove pch files
  list(REMOVE_ITEM onnxruntime_providers_cuda_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.cc"
  )

  # The shared_library files are in a separate list since they use precompiled headers, and the above files have them disabled.
  file(GLOB_RECURSE onnxruntime_providers_cuda_shared_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )
  file(GLOB_RECURSE onnxruntime_providers_cuda_cu_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cuh"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_shared_srcs} ${onnxruntime_providers_cuda_cu_srcs})
  set(onnxruntime_providers_cuda_src ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_shared_srcs} ${onnxruntime_providers_cuda_cu_srcs})

  # disable contrib ops conditionally
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    if (NOT onnxruntime_ENABLE_ATEN)
      list(REMOVE_ITEM onnxruntime_cuda_contrib_ops_cc_srcs
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/aten_ops/aten_op.cc"
      )
    endif()
    if (NOT onnxruntime_USE_NCCL)
      list(REMOVE_ITEM onnxruntime_cuda_contrib_ops_cc_srcs
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/nccl_kernels.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/sharding_spec.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/sharding.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_matmul.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_slice.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_reshape.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_expand.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_reduce.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_unsqueeze.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_squeeze.cc"
      )
    endif()
    # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
  endif()

  if (onnxruntime_ENABLE_TRAINING_OPS)
    file(GLOB_RECURSE onnxruntime_cuda_training_ops_cc_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.h"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cc"
    )

    file(GLOB_RECURSE onnxruntime_cuda_training_ops_cu_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cu"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cuh"
    )

    source_group(TREE ${ORTTRAINING_ROOT} FILES ${onnxruntime_cuda_training_ops_cc_srcs} ${onnxruntime_cuda_training_ops_cu_srcs})
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_training_ops_cc_srcs} ${onnxruntime_cuda_training_ops_cu_srcs})

    if(NOT onnxruntime_ENABLE_TRAINING)
      file(GLOB_RECURSE onnxruntime_cuda_full_training_only_srcs
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/*.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/*.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/communication/*.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/communication/*.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/record.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/record.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/wait.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/wait.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/yield.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/gist/*.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/gist/*.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/gist/*.cu"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/torch/*.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/torch/*.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/triton/triton_op.cc"
      )

      list(REMOVE_ITEM onnxruntime_providers_cuda_src ${onnxruntime_cuda_full_training_only_srcs})
    elseif(WIN32 OR NOT onnxruntime_USE_NCCL)
      # NCCL is not support in Windows build
      file(GLOB_RECURSE onnxruntime_cuda_nccl_op_srcs
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/nccl_common.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/nccl_kernels.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/megatron.cc"
      )
      list(REMOVE_ITEM onnxruntime_providers_cuda_src ${onnxruntime_cuda_nccl_op_srcs})
    endif()
  endif()

  if (onnxruntime_REDUCED_OPS_BUILD)
    substitute_op_reduction_srcs(onnxruntime_providers_cuda_src)
  endif()
  if(onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
    # cuda_provider_interface.cc is removed from the object target: onnxruntime_providers_cuda_obj and
    # add to the lib onnxruntime_providers_cuda separatedly.
    # onnxruntime_providers_cuda_ut can share all the object files with onnxruntime_providers_cuda except cuda_provider_interface.cc.
    set(cuda_provider_interface_src ${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_provider_interface.cc)
    list(REMOVE_ITEM onnxruntime_providers_cuda_src ${cuda_provider_interface_src})
    onnxruntime_add_object_library(onnxruntime_providers_cuda_obj ${onnxruntime_providers_cuda_src})
    onnxruntime_add_shared_library_module(onnxruntime_providers_cuda ${cuda_provider_interface_src} $<TARGET_OBJECTS:onnxruntime_providers_cuda_obj>)
  else()
    onnxruntime_add_shared_library_module(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_src})
  endif()
  # config_cuda_provider_shared_module can be used to config onnxruntime_providers_cuda_obj, onnxruntime_providers_cuda & onnxruntime_providers_cuda_ut.
  # This function guarantees that all 3 targets have the same configurations.
  function(config_cuda_provider_shared_module target)
    if (onnxruntime_REDUCED_OPS_BUILD)
      add_op_reduction_include_dirs(${target})
    endif()

    if (HAS_GUARD_CF)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /guard:cf>")
    endif()
    if (HAS_QSPECTRE)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Qspectre>")
    endif()
    foreach(ORT_FLAG ${ORT_WARNING_FLAGS})
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler \"${ORT_FLAG}\">")
    endforeach()
    # CUDA 11.3+ supports parallel compilation
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-guiding-compiler-driver-threads
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3)
      option(onnxruntime_NVCC_THREADS "Number of threads that NVCC can use for compilation." 1)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_NVCC_THREADS}\">")
    endif()
    if (UNIX)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-reorder>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-reorder>")
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
    else()
      #mutex.cuh(91): warning C4834: discarding return value of function with 'nodiscard' attribute
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4834>")
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>")
    endif()

    onnxruntime_add_include_to_target(${target} onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers)
    if (onnxruntime_ENABLE_TRAINING_OPS)
      onnxruntime_add_include_to_target(${target} onnxruntime_training)
      if (onnxruntime_ENABLE_TRAINING)
        target_link_libraries(${target} PRIVATE onnxruntime_training)
      endif()
      if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP OR onnxruntime_ENABLE_TRITON)
        onnxruntime_add_include_to_target(${target} Python::Module)
      endif()
    endif()

    add_dependencies(${target} onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
    target_link_libraries(${target} PRIVATE cublasLt cublas cudnn curand cufft ${ABSEIL_LIBS} ${ONNXRUNTIME_PROVIDERS_SHARED} Boost::mp11 safeint_interface)
    if(onnxruntime_CUDNN_HOME)
      target_include_directories(${target} PRIVATE ${onnxruntime_CUDNN_HOME}/include)
      target_link_directories(${target} PRIVATE ${onnxruntime_CUDNN_HOME}/lib)
    endif()

    if (onnxruntime_USE_TRITON_KERNEL)
      # compile triton kernel, generate .a and .h files
      include(onnxruntime_compile_triton_kernel.cmake)
      compile_triton_kernel(triton_kernel_obj_file triton_kernel_header_dir)
      add_dependencies(${target} onnxruntime_triton_kernel)
      target_compile_definitions(${target} PRIVATE USE_TRITON_KERNEL)
      target_include_directories(${target} PRIVATE ${triton_kernel_header_dir})
      target_link_libraries(${target} PUBLIC -Wl,--whole-archive ${triton_kernel_obj_file} -Wl,--no-whole-archive)
      # lib cuda needed by cuLaunchKernel
      target_link_libraries(${target} PRIVATE cuda)
    endif()

    if (onnxruntime_USE_FLASH_ATTENTION OR onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION)
      include(cutlass)
      target_include_directories(${target} PRIVATE ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/examples)
    endif()

    target_include_directories(${target} PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR}  ${eigen_INCLUDE_DIRS} ${TVM_INCLUDES} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CUDA)
    set_target_properties(${target} PROPERTIES FOLDER "ONNXRuntime")

    if (onnxruntime_ENABLE_CUDA_PROFILING) # configure cupti for cuda profiling
      target_include_directories(${target} PRIVATE ${onnxruntime_CUDA_HOME}/extras/CUPTI/include)
      target_link_directories(${target} PRIVATE ${onnxruntime_CUDA_HOME}/extras/CUPTI/lib64)
      target_link_libraries(${target} PRIVATE cupti)
    endif()

    if (onnxruntime_ENABLE_NVTX_PROFILE AND NOT WIN32)
      target_link_libraries(${target} PRIVATE nvToolsExt)
    endif()

    if (onnxruntime_ENABLE_TRAINING_OPS)
      target_include_directories(${target} PRIVATE ${ORTTRAINING_ROOT} ${MPI_CXX_INCLUDE_DIRS})
    endif()

    if(onnxruntime_USE_MPI)
      target_link_libraries(${target} PRIVATE ${MPI_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    endif()

    if (onnxruntime_USE_NCCL)
      target_include_directories(${target} PRIVATE ${NCCL_INCLUDE_DIRS})
      target_link_libraries(${target} PRIVATE ${NCCL_LIBRARIES})
    endif()

    if (WIN32)
      # *.cu cannot use PCH
      if (NOT onnxruntime_BUILD_CACHE)
        target_precompile_headers(${target} PUBLIC
          "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.h"
          "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.cc"
        )
      endif()

      # minimize the Windows includes.
      # this avoids an issue with CUDA 11.6 where 'small' is defined in the windows and cuda headers.
      target_compile_definitions(${target} PRIVATE "WIN32_LEAN_AND_MEAN")

      # disable a warning from the CUDA headers about unreferenced local functions
      #target_compile_options(${target} PRIVATE /wd4505)
      set(onnxruntime_providers_cuda_static_library_flags
          -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
      )
      set_target_properties(${target} PROPERTIES
          STATIC_LIBRARY_FLAGS "${onnxruntime_providers_cuda_static_library_flags}")
    endif()

    if(APPLE)
      set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/cuda/exported_symbols.lst")
      target_link_libraries(${target} PRIVATE nsync::nsync_cpp)
    elseif(UNIX)
      set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/cuda/version_script.lds -Xlinker --gc-sections")
      target_link_libraries(${target} PRIVATE nsync::nsync_cpp)
    elseif(WIN32)
      set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/cuda/symbols.def")
    else()
      message(FATAL_ERROR "${target} unknown platform, need to specify shared library exports for it")
    endif()

    if (onnxruntime_ENABLE_ATEN)
      target_compile_definitions(${target} PRIVATE ENABLE_ATEN)
    endif()
  endfunction()
  if(onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
    config_cuda_provider_shared_module(onnxruntime_providers_cuda_obj)
  endif()
  config_cuda_provider_shared_module(onnxruntime_providers_cuda)

  install(TARGETS onnxruntime_providers_cuda
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
