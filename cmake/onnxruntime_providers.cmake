# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Reduced ops build helpers

# In a reduced ops build, the reduction is performed by updating source files.
# Rather than modifying the source files directly, updated versions will be
# saved to another location in the build directory: ${op_reduction_root}.
set(op_reduction_root "${CMAKE_BINARY_DIR}/op_reduction.generated")

# This helper function replaces the relevant original source files with their
# updated, reduced ops versions in `all_srcs`.
function(substitute_op_reduction_srcs all_srcs)
  # files that are potentially updated in a reduced ops build
  set(original_srcs
    "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/cpu_contrib_kernels.cc"
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/cuda_contrib_kernels.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/cpu/cpu_execution_provider.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_execution_provider.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/op_kernel_type_control_overrides.inc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/cpu_training_kernels.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/cuda_training_kernels.cc"
    )

  set(replacement_srcs)

  foreach(original_src ${original_srcs})
    string(FIND "${${all_srcs}}" "${original_src}" idx)
    if(idx EQUAL "-1")
      continue()
    endif()

    file(RELATIVE_PATH src_relative_path "${REPO_ROOT}" "${original_src}")
    set(replacement_src "${op_reduction_root}/${src_relative_path}")

    message("File '${original_src}' substituted with reduced op version '${replacement_src}'.")

    string(REPLACE "${original_src}" "${replacement_src}" ${all_srcs} "${${all_srcs}}")

    list(APPEND replacement_srcs "${replacement_src}")
  endforeach()

  if(replacement_srcs)
    source_group(TREE "${op_reduction_root}" PREFIX "op_reduction.generated" FILES ${replacement_srcs})
  endif()

  set(${all_srcs} "${${all_srcs}}" PARENT_SCOPE)
endfunction()

# This helper function adds reduced ops build-specific include directories to
# `target`.
function(add_op_reduction_include_dirs target)
  set(op_reduction_include_dirs "${op_reduction_root}/onnxruntime")
  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    list(APPEND op_reduction_include_dirs "${op_reduction_root}/orttraining")
  endif()
  # add include directories BEFORE so they are searched first, giving op reduction file paths precedence
  target_include_directories(${target} BEFORE PRIVATE ${op_reduction_include_dirs})
endfunction()


file(GLOB_RECURSE onnxruntime_providers_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.cc"
)

if(onnxruntime_DISABLE_ML_OPS)
  list(FILTER onnxruntime_providers_srcs EXCLUDE REGEX ".*/ml/.*")
endif()

file(GLOB_RECURSE onnxruntime_cpu_contrib_ops_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.cc"
)

file(GLOB_RECURSE onnxruntime_cuda_contrib_ops_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cc"
)

file(GLOB_RECURSE onnxruntime_cuda_contrib_ops_cu_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cu"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cuh"
)

file(GLOB_RECURSE onnxruntime_rocm_contrib_ops_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/rocm/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/rocm/*.cc"
)

file(GLOB_RECURSE onnxruntime_rocm_contrib_ops_cu_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/rocm/*.cu"
  "${ONNXRUNTIME_ROOT}/contrib_ops/rocm/*.cuh"
)

file(GLOB onnxruntime_providers_common_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/op_kernel_type_control_overrides.inc"
)
if(onnxruntime_USE_VITISAI)
  set(PROVIDERS_VITISAI onnxruntime_providers_vitisai)
endif()
if(onnxruntime_USE_CUDA)
  set(PROVIDERS_CUDA onnxruntime_providers_cuda)
endif()
if(onnxruntime_USE_COREML)
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(PROVIDERS_COREML onnxruntime_providers_coreml onnxruntime_coreml_proto)
  else()
    set(PROVIDERS_COREML onnxruntime_providers_coreml)
  endif()
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  set(PROVIDERS_NNAPI onnxruntime_providers_nnapi)
endif()
if(onnxruntime_USE_RKNPU)
  set(PROVIDERS_RKNPU onnxruntime_providers_rknpu)
endif()
if(onnxruntime_USE_DML)
  set(PROVIDERS_DML onnxruntime_providers_dml)
endif()
if(onnxruntime_USE_MIGRAPHX)
  set(PROVIDERS_MIGRAPHX onnxruntime_providers_migraphx)
endif()
if(onnxruntime_USE_WINML)
  set(PROVIDERS_WINML onnxruntime_providers_winml)
endif()
if(onnxruntime_USE_ACL)
  set(PROVIDERS_ACL onnxruntime_providers_acl)
endif()
if(onnxruntime_USE_ARMNN)
  set(PROVIDERS_ARMNN onnxruntime_providers_armnn)
endif()
if(onnxruntime_USE_ROCM)
  set(PROVIDERS_ROCM onnxruntime_providers_rocm)
endif()
if (onnxruntime_USE_TVM)
  set(PROVIDERS_TVM onnxruntime_providers_tvm)
endif()
if (onnxruntime_USE_XNNPACK)
  set(PROVIDERS_XNNPACK onnxruntime_providers_xnnpack)
endif()
if(onnxruntime_USE_SNPE)
    include(onnxruntime_snpe_provider.cmake)
endif()
if (onnxruntime_USE_CANN)
  set(PROVIDERS_CANN onnxruntime_providers_cann)
endif()

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})

set(onnxruntime_providers_src ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})

# disable contrib ops conditionally
if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
  if (NOT onnxruntime_ENABLE_ATEN)
    list(REMOVE_ITEM onnxruntime_cpu_contrib_ops_srcs
      "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/aten_ops/aten_op.h"
      "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/aten_ops/aten_op.cc"
      "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/aten_ops/aten_op_executor.cc"
    )
  endif()
  # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
  source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cpu_contrib_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_contrib_ops_srcs})
endif()


if (onnxruntime_ENABLE_TRAINING_OPS)
  file(GLOB_RECURSE onnxruntime_cpu_training_ops_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.cc"
  )

  source_group(TREE ${ORTTRAINING_ROOT}/ FILES ${onnxruntime_cpu_training_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_training_ops_srcs})

  file(GLOB_RECURSE onnxruntime_cpu_full_training_only_srcs
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/collective/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/collective/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/communication/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/communication/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/record.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/record.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/wait.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/wait.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/yield.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/yield.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/gist/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/gist/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/tensorboard/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/tensorboard/*.h"
  )

  list(REMOVE_ITEM onnxruntime_providers_src ${onnxruntime_cpu_full_training_only_srcs})
endif()

if (onnxruntime_ENABLE_ATEN)
  file(GLOB_RECURSE onnxruntime_providers_dlpack_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/dlpack/dlpack_converter.cc"
    "${ONNXRUNTIME_ROOT}/core/dlpack/dlpack_converter.h"
  )
  set(onnxruntime_providers_dlpack_srcs ${onnxruntime_providers_dlpack_srcs})
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_dlpack_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_providers_dlpack_srcs})
endif()

if (onnxruntime_ENABLE_TRAINING)
  file(GLOB_RECURSE onnxruntime_cpu_training_ops_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/adasum/*"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/communication/*"
  )

  # This is already built in framework.cmake
  file(GLOB_RECURSE onnxruntime_training_framework_excude_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.h"
      "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.cc"
  )

  list(REMOVE_ITEM onnxruntime_cpu_training_ops_srcs ${onnxruntime_training_framework_excude_srcs})

  source_group(TREE ${ORTTRAINING_ROOT}/ FILES ${onnxruntime_cpu_training_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_training_ops_srcs})
endif()

if (onnxruntime_REDUCED_OPS_BUILD)
  substitute_op_reduction_srcs(onnxruntime_providers_src)
endif()
onnxruntime_add_static_library(onnxruntime_providers ${onnxruntime_providers_src})
if (onnxruntime_REDUCED_OPS_BUILD)
  add_op_reduction_include_dirs(onnxruntime_providers)
endif()

if (HAS_BITWISE_INSTEAD_OF_LOGICAL)
  target_compile_options(onnxruntime_providers PRIVATE "-Wno-bitwise-instead-of-logical")
endif()

if (MSVC)
   target_compile_options(onnxruntime_providers PRIVATE "/bigobj")
   if(onnxruntime_DEV_MODE AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
      target_compile_options(onnxruntime_providers PRIVATE "/wd4244")
   endif()
endif()
onnxruntime_add_include_to_target(onnxruntime_providers onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)

if (onnxruntime_BUILD_MS_EXPERIMENTAL_OPS)
  target_compile_definitions(onnxruntime_providers PRIVATE BUILD_MS_EXPERIMENTAL_OPS=1)
endif()

if(HAS_DEPRECATED_COPY)
  #temporarily ignore this warning
  #see: https://en.wikipedia.org/wiki/Rule_of_three_(C%2B%2B_programming)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/math/matmul_integer.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/math/quantize_linear_matmul.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/qlinearconv.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/conv_integer.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/generator/random.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/onehot.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/where_op.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
endif()

target_include_directories(onnxruntime_providers PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${RE2_INCLUDE_DIR})

add_dependencies(onnxruntime_providers onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})

if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
  target_include_directories(onnxruntime_providers PRIVATE ${ORTTRAINING_ROOT})
endif()

if (onnxruntime_ENABLE_ATEN)
  target_compile_definitions(onnxruntime_providers PRIVATE ENABLE_ATEN)
  # DLPack is a header-only dependency
  set(DLPACK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/dlpack/include)
  target_include_directories(onnxruntime_providers PRIVATE ${DLPACK_INCLUDE_DIR})
endif()

if (onnxruntime_ENABLE_TRAINING)
  add_dependencies(onnxruntime_providers tensorboard)
  onnxruntime_add_include_to_target(onnxruntime_providers tensorboard)
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    onnxruntime_add_include_to_target(onnxruntime_providers Python::Module)
  endif()

  if (onnxruntime_USE_NCCL OR onnxruntime_USE_MPI)
    target_include_directories(onnxruntime_providers PUBLIC ${MPI_CXX_INCLUDE_DIRS})
  endif()
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cpu  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
set_target_properties(onnxruntime_providers PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_providers PROPERTIES FOLDER "ONNXRuntime")

if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD
                                  AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS"
                                  AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Android")
                                  AND NOT onnxruntime_BUILD_WEBASSEMBLY)
  file(GLOB onnxruntime_providers_shared_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/shared/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/shared/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_shared_cc_srcs})
  onnxruntime_add_shared_library(onnxruntime_providers_shared ${onnxruntime_providers_shared_cc_srcs} "${ONNXRUNTIME_ROOT}/core/dll/onnxruntime.rc")
  set_target_properties(onnxruntime_providers_shared PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_shared PROPERTIES LINKER_LANGUAGE CXX)

  target_compile_definitions(onnxruntime_providers_shared PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
  target_compile_definitions(onnxruntime_providers_shared PRIVATE VER_MINOR=${VERSION_MINOR_PART})
  target_compile_definitions(onnxruntime_providers_shared PRIVATE VER_BUILD=${VERSION_BUILD_PART})
  target_compile_definitions(onnxruntime_providers_shared PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
  target_compile_definitions(onnxruntime_providers_shared PRIVATE VER_STRING=\"${VERSION_STRING}\")
  target_compile_definitions(onnxruntime_providers_shared PRIVATE FILE_NAME=\"onnxruntime_providers_shared.dll\")


  # On Apple/Unix we don't directly link with this library as we load it with RTLD_GLOBAL, so this is only set to the actual library on WIN32
  set(ONNXRUNTIME_PROVIDERS_SHARED)

  if(APPLE)
  set_property(TARGET onnxruntime_providers_shared APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/shared/exported_symbols.lst")
  elseif(UNIX)
  set_property(TARGET onnxruntime_providers_shared APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/shared/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
  set_property(TARGET onnxruntime_providers_shared APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/shared/symbols.def")
  set(ONNXRUNTIME_PROVIDERS_SHARED onnxruntime_providers_shared)
  else()
  message(FATAL_ERROR "onnxruntime_providers_shared unknown platform, need to specify shared library exports for it")
  endif()

  install(TARGETS onnxruntime_providers_shared
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

if (onnxruntime_USE_CUDA)
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
    # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
  endif()

  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    file(GLOB_RECURSE onnxruntime_cuda_training_ops_cc_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.h"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/communication/*.h"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/communication/*.cc"
    )

    file(GLOB_RECURSE onnxruntime_cuda_training_ops_cu_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cu"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cuh"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/communication/*.cu"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/communication/*.cuh"
    )

    # NCCL is not support in Windows build
    if (WIN32)
      list(REMOVE_ITEM onnxruntime_cuda_training_ops_cc_srcs
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/nccl_common.cc"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/nccl_kernels.cc"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/megatron.cc"
      )
    elseif (NOT onnxruntime_USE_NCCL)
      list(REMOVE_ITEM onnxruntime_cuda_training_ops_cc_srcs
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/nccl_common.cc"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/nccl_kernels.cc"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/megatron.cc"
      )
    endif()

    source_group(TREE ${ORTTRAINING_ROOT} FILES ${onnxruntime_cuda_training_ops_cc_srcs} ${onnxruntime_cuda_training_ops_cu_srcs})
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_training_ops_cc_srcs} ${onnxruntime_cuda_training_ops_cu_srcs})
  endif()

  if (onnxruntime_REDUCED_OPS_BUILD)
    substitute_op_reduction_srcs(onnxruntime_providers_cuda_src)
  endif()
  onnxruntime_add_shared_library_module(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_src})
  if (onnxruntime_REDUCED_OPS_BUILD)
    add_op_reduction_include_dirs(onnxruntime_providers_cuda)
  endif()

  #target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler \"/analyze:stacksize 131072\">")
  if (HAS_GUARD_CF)
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /guard:cf>")
  endif()
  if (HAS_QSPECTRE)
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Qspectre>")
  endif()
  foreach(ORT_FLAG ${ORT_WARNING_FLAGS})
      target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler \"${ORT_FLAG}\">")
  endforeach()
  # CUDA 11.3+ supports parallel compilation
  # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-guiding-compiler-driver-threads
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3)
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_NVCC_THREADS}\">")
  endif()
  if (onnxruntime_DEV_MODE)
	  if (UNIX)
		target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-reorder>"
				"$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-reorder>")
		target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
				"$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
	  else()
		#mutex.cuh(91): warning C4834: discarding return value of function with 'nodiscard' attribute
		target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4834>")
		target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>")
	  endif()
  endif()
  onnxruntime_add_include_to_target(onnxruntime_providers_cuda onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)
  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    onnxruntime_add_include_to_target(onnxruntime_providers_cuda onnxruntime_training)
    target_link_libraries(onnxruntime_providers_cuda PRIVATE onnxruntime_training)
    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      onnxruntime_add_include_to_target(onnxruntime_providers_cuda Python::Module)
    endif()
  endif()

  add_dependencies(onnxruntime_providers_cuda onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_link_libraries(onnxruntime_providers_cuda PRIVATE cublasLt cublas cudnn curand cufft ${ABSEIL_LIBS} ${ONNXRUNTIME_PROVIDERS_SHARED})
  if(onnxruntime_CUDNN_HOME)
    target_include_directories(onnxruntime_providers_cuda PRIVATE ${onnxruntime_CUDNN_HOME}/include)
  endif()
  target_include_directories(onnxruntime_providers_cuda PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR}  ${eigen_INCLUDE_DIRS} ${TVM_INCLUDES} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
  set_target_properties(onnxruntime_providers_cuda PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(onnxruntime_providers_cuda PROPERTIES FOLDER "ONNXRuntime")

  if (onnxruntime_ENABLE_CUDA_PROFILING) # configure cupti for cuda profiling
    target_include_directories(onnxruntime_providers_cuda PRIVATE ${onnxruntime_CUDA_HOME}/extras/CUPTI/include)
    target_link_directories(onnxruntime_providers_cuda PRIVATE ${onnxruntime_CUDA_HOME}/extras/CUPTI/lib64)
    target_link_libraries(onnxruntime_providers_cuda PRIVATE cupti)
  endif()

  if (onnxruntime_ENABLE_NVTX_PROFILE)
    target_link_libraries(onnxruntime_providers_cuda PRIVATE nvToolsExt)
  endif()

  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
    target_include_directories(onnxruntime_providers_cuda PRIVATE ${PROJECT_SOURCE_DIR}/external/cub)
  endif()

  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_providers_cuda PRIVATE ${ORTTRAINING_ROOT} ${MPI_CXX_INCLUDE_DIRS})
    if(onnxruntime_USE_MPI)
      target_link_libraries(onnxruntime_providers_cuda PRIVATE ${MPI_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    endif()

    if (onnxruntime_USE_NCCL)
      target_include_directories(onnxruntime_providers_cuda PRIVATE ${NCCL_INCLUDE_DIRS})
      target_link_libraries(onnxruntime_providers_cuda PRIVATE ${NCCL_LIBRARIES})
    endif()
  endif()

  if (WIN32)
    # *.cu cannot use PCH
    target_precompile_headers(onnxruntime_providers_cuda PUBLIC
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.h"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.cc"
    )

    # minimize the Windows includes.
    # this avoids an issue with CUDA 11.6 where 'small' is defined in the windows and cuda headers.
    target_compile_definitions(onnxruntime_providers_cuda PRIVATE "WIN32_LEAN_AND_MEAN")

    # disable a warning from the CUDA headers about unreferenced local functions
    #target_compile_options(onnxruntime_providers_cuda PRIVATE /wd4505)
    set(onnxruntime_providers_cuda_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(onnxruntime_providers_cuda PROPERTIES
        STATIC_LIBRARY_FLAGS "${onnxruntime_providers_cuda_static_library_flags}")
  endif()

  if(APPLE)
    set_property(TARGET onnxruntime_providers_cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/cuda/exported_symbols.lst")
    target_link_libraries(onnxruntime_providers_cuda PRIVATE nsync_cpp)
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/cuda/version_script.lds -Xlinker --gc-sections")
    target_link_libraries(onnxruntime_providers_cuda PRIVATE nsync_cpp)
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_cuda APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/cuda/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_cuda unknown platform, need to specify shared library exports for it")
  endif()

  if (onnxruntime_ENABLE_ATEN)
    target_compile_definitions(onnxruntime_providers_cuda PRIVATE ENABLE_ATEN)
  endif()

  install(TARGETS onnxruntime_providers_cuda
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

endif()

if (onnxruntime_USE_DNNL)
  file(GLOB_RECURSE onnxruntime_providers_dnnl_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/dnnl/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/dnnl/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_dnnl_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_dnnl ${onnxruntime_providers_dnnl_cc_srcs})
  target_link_directories(onnxruntime_providers_dnnl PRIVATE ${DNNL_LIB_DIR})
  if (MSVC AND onnxruntime_ENABLE_STATIC_ANALYSIS)
    # dnnl_convgrad.cc(47,0): Warning C6262: Function uses '38816' bytes of stack:  exceeds /analyze:stacksize '16384'.  Consider moving some data to heap.
    target_compile_options(onnxruntime_providers_dnnl PRIVATE  "/analyze:stacksize 131072")
  endif()

  add_dependencies(onnxruntime_providers_dnnl onnxruntime_providers_shared project_dnnl ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_dnnl PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${DNNL_INCLUDE_DIR} ${DNNL_OCL_INCLUDE_DIR})
  # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
  target_link_libraries(onnxruntime_providers_dnnl PRIVATE dnnl ${ONNXRUNTIME_PROVIDERS_SHARED} Boost::mp11 ${ABSEIL_LIBS} ${GSL_TARGET})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/dnnl  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_dnnl PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_dnnl PROPERTIES LINKER_LANGUAGE CXX)

  # Needed for the provider interface, as it includes training headers when training is enabled
  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_providers_dnnl PRIVATE ${ORTTRAINING_ROOT})
  endif()

  if(APPLE)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/dnnl/exported_symbols.lst")
    set_target_properties(onnxruntime_providers_dnnl PROPERTIES
      INSTALL_RPATH "@loader_path"
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE)
    target_link_libraries(onnxruntime_providers_dnnl PRIVATE nsync_cpp)
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/dnnl/version_script.lds -Xlinker --gc-sections -Xlinker -rpath=\$ORIGIN")
    target_link_libraries(onnxruntime_providers_dnnl PRIVATE nsync_cpp)
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/dnnl/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_dnnl unknown platform, need to specify shared library exports for it")
  endif()

  install(TARGETS onnxruntime_providers_dnnl
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT=1)
  if (onnxruntime_TENSORRT_PLACEHOLDER_BUILDER)
    add_definitions(-DORT_TENSORRT_PLACEHOLDER_BUILDER)
  endif()
  set(BUILD_LIBRARY_ONLY 1)
  add_definitions("-DONNX_ML=1")
  add_definitions("-DONNX_NAMESPACE=onnx")
  include_directories(${PROJECT_SOURCE_DIR}/external/protobuf)
  set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set(TENSORRT_ROOT ${onnxruntime_TENSORRT_HOME})
  include_directories(${ONNXRUNTIME_ROOT}/../cmake/external/onnx)
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  if (WIN32)
    add_definitions(-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING=1)
    set(OLD_CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4099 /wd4551 /wd4505 /wd4515 /wd4706 /wd4456 /wd4324 /wd4701 /wd4804 /wd4702 /wd4458 /wd4703")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4805")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -include algorithm")
    set(PROTOBUF_LIBRARY libprotobuf)
    set(DISABLED_WARNINGS_FOR_TRT /wd4456)
  endif()
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
  endif()
  set(CXX_VERSION_DEFINED TRUE)

  if (onnxruntime_USE_TENSORRT_BUILTIN_PARSER)
    # Add TensorRT library
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      HINTS ${TENSORRT_ROOT}
      PATH_SUFFIXES include)
    MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
    find_library(TENSORRT_LIBRARY_INFER nvinfer
      HINTS ${TENSORRT_ROOT}
      PATH_SUFFIXES lib lib64 lib/x64)
    find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
      HINTS  ${TENSORRT_ROOT}
      PATH_SUFFIXES lib lib64 lib/x64)
    find_library(TENSORRT_LIBRARY_NVONNXPARSER nvonnxparser
      HINTS  ${TENSORRT_ROOT}
      PATH_SUFFIXES lib lib64 lib/x64)
    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_NVONNXPARSER})
    MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
  else()
    add_subdirectory(${ONNXRUNTIME_ROOT}/../cmake/external/onnx-tensorrt)
    set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
    if ( CMAKE_COMPILER_IS_GNUCC )
      set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    endif()
    if (WIN32)
      set(CMAKE_CUDA_FLAGS ${OLD_CMAKE_CUDA_FLAGS})
      unset(PROTOBUF_LIBRARY)
      unset(OLD_CMAKE_CXX_FLAGS)
      unset(OLD_CMAKE_CUDA_FLAGS)
      set_target_properties(nvonnxparser PROPERTIES LINK_FLAGS "/ignore:4199")
      target_compile_options(nvonnxparser_static PRIVATE /FIio.h /wd4100)
      target_compile_options(nvonnxparser PRIVATE /FIio.h /wd4100)
    endif()
    include_directories(${ONNXRUNTIME_ROOT}/../cmake/external/onnx-tensorrt)
    include_directories(${TENSORRT_INCLUDE_DIR})
    set(onnxparser_link_libs nvonnxparser_static)
  endif()

  set(trt_link_libs cudnn ${CMAKE_DL_LIBS} ${TENSORRT_LIBRARY})

  file(GLOB_RECURSE onnxruntime_providers_tensorrt_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_tensorrt_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_tensorrt ${onnxruntime_providers_tensorrt_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_tensorrt onnxruntime_common onnx flatbuffers)
  add_dependencies(onnxruntime_providers_tensorrt onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  if (onnxruntime_USE_TENSORRT_BUILTIN_PARSER)
    target_link_libraries(onnxruntime_providers_tensorrt PRIVATE ${trt_link_libs} cudart ${ONNXRUNTIME_PROVIDERS_SHARED} ${PROTOBUF_LIB} flatbuffers ${ABSEIL_LIBS})
    target_include_directories(onnxruntime_providers_tensorrt PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  else()
    target_link_libraries(onnxruntime_providers_tensorrt PRIVATE ${onnxparser_link_libs} ${trt_link_libs} cudart ${ONNXRUNTIME_PROVIDERS_SHARED} ${PROTOBUF_LIB} flatbuffers ${ABSEIL_LIBS})
    target_include_directories(onnxruntime_providers_tensorrt PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  endif()
  if(onnxruntime_CUDNN_HOME)
    target_include_directories(onnxruntime_providers_tensorrt PRIVATE ${onnxruntime_CUDNN_HOME}/include)
  endif()
  # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/tensorrt  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_tensorrt PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(onnxruntime_providers_tensorrt PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_tensorrt PRIVATE ONNXIFI_BUILD_LIBRARY=1)
  target_compile_options(onnxruntime_providers_tensorrt PRIVATE ${DISABLED_WARNINGS_FOR_TRT})
  if (WIN32)
    target_compile_options(onnxruntime_providers_tensorrt INTERFACE /wd4456)
  endif()

  # Needed for the provider interface, as it includes training headers when training is enabled
  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_providers_tensorrt PRIVATE ${ORTTRAINING_ROOT})
    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      onnxruntime_add_include_to_target(onnxruntime_providers_tensorrt Python::Module)
    endif()
  endif()

  if(APPLE)
    set_property(TARGET onnxruntime_providers_tensorrt APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/tensorrt/exported_symbols.lst")
    target_link_libraries(onnxruntime_providers_tensorrt PRIVATE nsync_cpp)
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_tensorrt APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
    set_property(TARGET onnxruntime_providers_tensorrt APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/tensorrt/version_script.lds -Xlinker --gc-sections")
    target_link_libraries(onnxruntime_providers_tensorrt PRIVATE nsync_cpp stdc++fs)
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_tensorrt APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/tensorrt/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_tensorrt unknown platform, need to specify shared library exports for it")
  endif()

  install(TARGETS onnxruntime_providers_tensorrt
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (onnxruntime_USE_VITISAI)
  file(GLOB_RECURSE onnxruntime_providers_vitisai_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_vitisai_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_vitisai ${onnxruntime_providers_vitisai_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_vitisai
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers
  )
  add_dependencies(onnxruntime_providers_vitisai ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_vitisai PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_vitisai PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${VITISAI_INCLUDE_DIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/vitisai
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers
  )
  set_target_properties(onnxruntime_providers_vitisai PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_vitisai
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_OPENVINO)

#  include_directories("${CMAKE_CURRENT_BINARY_DIR}/onnx")
  file(GLOB_RECURSE onnxruntime_providers_openvino_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.hpp"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.cpp"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  if (WIN32)
	  set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
  endif()

  # Header paths
  find_package(InferenceEngine REQUIRED)
  find_package(ngraph REQUIRED)

  if (OPENVINO_2022_1 OR OPENVINO_2022_2)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
  list (OV_20_LIBS openvino::frontend::onnx openvino::runtime)
  endif()

  if (WIN32)
    unset(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO)
  endif()

  if ((DEFINED ENV{OPENCL_LIBS}) AND (DEFINED ENV{OPENCL_INCS}))
    add_definitions(-DIO_BUFFER_ENABLED=1)
    list(APPEND OPENVINO_LIB_LIST $ENV{OPENCL_LIBS} ${OV_20_LIBS} ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES} ngraph::onnx_importer ${PYTHON_LIBRARIES})
  else()
    list(APPEND OPENVINO_LIB_LIST ${OV_20_LIBS} ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES} ngraph::onnx_importer ${PYTHON_LIBRARIES})
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_openvino_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_openvino ${onnxruntime_providers_openvino_cc_srcs} "${ONNXRUNTIME_ROOT}/core/dll/onnxruntime.rc")
  onnxruntime_add_include_to_target(onnxruntime_providers_openvino onnxruntime_common onnx)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/openvino  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES FOLDER "ONNXRuntime")
  if(NOT MSVC)
    target_compile_options(onnxruntime_providers_openvino PRIVATE "-Wno-parentheses")
  endif()
  add_dependencies(onnxruntime_providers_openvino onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_openvino SYSTEM PUBLIC ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS} ${OpenVINO_INCLUDE_DIR} ${OPENVINO_INCLUDE_DIR_LIST} ${PYTHON_INCLUDE_DIRS} $ENV{OPENCL_INCS})
  target_link_libraries(onnxruntime_providers_openvino ${ONNXRUNTIME_PROVIDERS_SHARED} ${OPENVINO_LIB_LIST} ${ABSEIL_LIBS})

  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_MINOR=${VERSION_MINOR_PART})
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_BUILD=${VERSION_BUILD_PART})
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE VER_STRING=\"${VERSION_STRING}\")
  target_compile_definitions(onnxruntime_providers_openvino PRIVATE FILE_NAME=\"onnxruntime_providers_openvino.dll\")

  if(MSVC)
    target_compile_options(onnxruntime_providers_openvino PUBLIC /wd4099 /wd4275 /wd4100 /wd4005 /wd4244 /wd4267)
  endif()

  # Needed for the provider interface, as it includes training headers when training is enabled
  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_providers_openvino PRIVATE ${ORTTRAINING_ROOT})
  endif()

  if(APPLE)
    set_property(TARGET onnxruntime_providers_openvino APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/openvino/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_openvino APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/openvino/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_openvino APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/openvino/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_openvino unknown platform, need to specify shared library exports for it")
  endif()

  install(TARGETS onnxruntime_providers_openvino
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (onnxruntime_USE_COREML)
  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "CoreML EP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_COREML=1)

  # Compile CoreML proto definition to ${CMAKE_CURRENT_BINARY_DIR}/coreml
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(COREML_PROTO_ROOT ${PROJECT_SOURCE_DIR}/../onnxruntime/core/providers/coreml/mlmodel_format)
    file(GLOB coreml_proto_srcs
      "${COREML_PROTO_ROOT}/*.proto"
    )
    onnxruntime_add_static_library(onnxruntime_coreml_proto ${coreml_proto_srcs})
    target_include_directories(onnxruntime_coreml_proto PUBLIC $<TARGET_PROPERTY:${PROTOBUF_LIB},INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_definitions(onnxruntime_coreml_proto PUBLIC $<TARGET_PROPERTY:${PROTOBUF_LIB},INTERFACE_COMPILE_DEFINITIONS>)
    set_target_properties(onnxruntime_coreml_proto PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    set_target_properties(onnxruntime_coreml_proto PROPERTIES COMPILE_FLAGS "-fvisibility-inlines-hidden")
    set(_src_sub_dir "coreml/")
    onnxruntime_protobuf_generate(
      APPEND_PATH
      GEN_SRC_SUB_DIR ${_src_sub_dir}
      IMPORT_DIRS ${COREML_PROTO_ROOT}
      TARGET onnxruntime_coreml_proto
    )

    if (NOT onnxruntime_BUILD_SHARED_LIB)
      install(TARGETS onnxruntime_coreml_proto
              ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
              LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
              RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
              FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR}
      )
    endif()
  endif()

  # These are shared utils,
  # TODO, move this to a separated lib when used by EPs other than NNAPI and CoreML
  file(GLOB_RECURSE onnxruntime_providers_shared_utils_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
  )

  file(GLOB
    onnxruntime_providers_coreml_cc_srcs_top CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/*.cc"
  )

  # Add builder source code
  file(GLOB_RECURSE
    onnxruntime_providers_coreml_cc_srcs_nested CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/builders/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/builders/*.cc"
  )
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    list(REMOVE_ITEM onnxruntime_providers_coreml_cc_srcs_nested
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/builders/model_builder.h"
    "${ONNXRUNTIME_ROOT}/core/providers/coreml/builders/model_builder.cc"
    )
  endif()

  # Add CoreML objective c++ source code
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    file(GLOB
      onnxruntime_providers_coreml_objcc_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/model.h"
      "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/model.mm"
      "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/host_utils.h"
      "${ONNXRUNTIME_ROOT}/core/providers/coreml/model/host_utils.mm"
    )
  endif()

  set(onnxruntime_providers_coreml_cc_srcs
    ${onnxruntime_providers_coreml_cc_srcs_top}
    ${onnxruntime_providers_coreml_cc_srcs_nested}
    ${onnxruntime_providers_shared_utils_cc_srcs}
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_coreml_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_coreml
    ${onnxruntime_providers_coreml_cc_srcs} ${onnxruntime_providers_coreml_objcc_srcs}
  )
  onnxruntime_add_include_to_target(onnxruntime_providers_coreml
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers
  )
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    onnxruntime_add_include_to_target(onnxruntime_providers_coreml onnxruntime_coreml_proto)
    target_link_libraries(onnxruntime_providers_coreml PRIVATE onnxruntime_coreml_proto "-framework Foundation" "-framework CoreML")
    add_dependencies(onnxruntime_providers_coreml onnxruntime_coreml_proto)
  endif()
  add_dependencies(onnxruntime_providers_coreml onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_coreml PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_coreml PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_coreml PRIVATE ${ONNXRUNTIME_ROOT} ${coreml_INCLUDE_DIRS})
  set_target_properties(onnxruntime_providers_coreml PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_coreml
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "NNAPI can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_NNAPI=1)

  # This is the minimum Android API Level required by ORT NNAPI EP to run
  # ORT running on any host system with Android API level less than this will fall back to CPU EP
  if(onnxruntime_NNAPI_MIN_API)
    add_compile_definitions(ORT_NNAPI_MIN_API_LEVEL=${onnxruntime_NNAPI_MIN_API})
  endif()

  # This is the maximum Android API level supported in the ort model conversion for NNAPI EP
  # Note: This is only for running NNAPI for ort format model conversion on non-Android system since we cannot
  #       get the actually Android system version.
  if(onnxruntime_NNAPI_HOST_API)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
      message(FATAL_ERROR "onnxruntime_NNAPI_HOST_API should only be set for non-Android target")
    endif()
    add_compile_definitions(ORT_NNAPI_MAX_SUPPORTED_API_LEVEL=${onnxruntime_NNAPI_HOST_API})
  endif()

  set(onnxruntime_provider_nnapi_cc_src_patterns
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/*.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/*.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/builders/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/builders/*.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/builders/impl/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/builders/impl/*.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
  )

  # On Android, use the actual NNAPI implementation.
  # Otherwise, use a stub implementation to support some unit testing.
  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    list(APPEND onnxruntime_provider_nnapi_cc_src_patterns
         "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.cc")
  else()
    list(APPEND onnxruntime_provider_nnapi_cc_src_patterns
         "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation_stub.cc")
  endif()

  # These are shared utils,
  # TODO, move this to a separated lib when used by EPs other than NNAPI and CoreML
  list(APPEND onnxruntime_provider_nnapi_cc_src_patterns
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/node_unit/node_unit.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/node_unit/node_unit.cc"
  )

  file(GLOB onnxruntime_providers_nnapi_cc_srcs CONFIGURE_DEPENDS ${onnxruntime_provider_nnapi_cc_src_patterns})

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nnapi_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_nnapi ${onnxruntime_providers_nnapi_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_nnapi
    onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers
  )
  target_link_libraries(onnxruntime_providers_nnapi)
  add_dependencies(onnxruntime_providers_nnapi onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_nnapi PRIVATE ${ONNXRUNTIME_ROOT} ${nnapi_INCLUDE_DIRS})
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES LINKER_LANGUAGE CXX)
  # ignore the warning unknown-pragmas on "pragma region"
  if(NOT MSVC)
    target_compile_options(onnxruntime_providers_nnapi PRIVATE "-Wno-unknown-pragmas")
  endif()

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_nnapi
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_RKNPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
  add_definitions(-DUSE_RKNPU=1)
  option(DNN_READ_ONNX "" ON)
  set(DNN_CUSTOM_PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
  option(DNN_CMAKE_INSTALL "" OFF)
  option(DNN_BUILD_BIN "" OFF)
  if (NOT RKNPU_DDK_PATH)
    message(FATAL_ERROR "RKNPU_DDK_PATH required for onnxruntime_USE_RKNPU")
  endif()
  set(RKNPU_DDK_INCLUDE_DIR ${RKNPU_DDK_PATH}/include)
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(RKNPU_DDK_LIB_DIR ${RKNPU_DDK_PATH}/lib64)
  else()
    set(RKNPU_DDK_LIB_DIR ${RKNPU_DDK_PATH}/lib)
  endif()
  file(GLOB_RECURSE
    onnxruntime_providers_rknpu_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/rknpu/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/rknpu/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_rknpu_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_rknpu ${onnxruntime_providers_rknpu_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_rknpu
    onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers
  )
  target_link_libraries(onnxruntime_providers_rknpu PRIVATE -lrknpu_ddk)
  add_dependencies(onnxruntime_providers_rknpu onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_rknpu PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_rknpu PRIVATE
    ${ONNXRUNTIME_ROOT} ${rknpu_INCLUDE_DIRS} ${RKNPU_DDK_INCLUDE_DIR}
  )
  link_directories(onnxruntime_providers_rknpu ${RKNPU_DDK_LIB_DIR})
  set_target_properties(onnxruntime_providers_rknpu PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_rknpu
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_DML)
  file(GLOB_RECURSE onnxruntime_providers_dml_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/dml/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/dml/*.cpp"
    "${ONNXRUNTIME_ROOT}/core/providers/dml/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_dml_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_dml ${onnxruntime_providers_dml_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_dml
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers
  )
  add_dependencies(onnxruntime_providers_dml ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_dml PRIVATE
    ${ONNXRUNTIME_ROOT} ${ONNXRUNTIME_ROOT}/../cmake/external/wil/include
  )

  add_definitions(-DDML_TARGET_VERSION_USE_LATEST=1)

  if (NOT onnxruntime_USE_CUSTOM_DIRECTML)
    foreach(file "DirectML.dll" "DirectML.pdb" "DirectML.Debug.dll" "DirectML.Debug.pdb")
      add_custom_command(TARGET onnxruntime_providers_dml
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${DML_PACKAGE_DIR}/bin/${onnxruntime_target_platform}-win/${file}" $<TARGET_FILE_DIR:onnxruntime_providers_dml>)
    endforeach()
  endif()

  function(target_add_dml target)
    if (onnxruntime_USE_CUSTOM_DIRECTML)
      if (dml_EXTERNAL_PROJECT)
        # Internal build of DirectML: link against the "DirectML" target.
        target_link_libraries(${target} PRIVATE DirectML)
      else()
        if (dml_LIB_DIR)
          target_link_libraries(${target} PRIVATE ${dml_LIB_DIR}/DirectML.lib)
        else()
          target_link_libraries(${target} PRIVATE DirectML)
        endif()
      endif()
    else()
      add_dependencies(${target} RESTORE_PACKAGES)
      target_link_libraries(${target} PRIVATE "${DML_PACKAGE_DIR}/bin/${onnxruntime_target_platform}-win/DirectML.lib")
	    target_compile_definitions(${target} PRIVATE DML_TARGET_VERSION_USE_LATEST)
    endif()
  endfunction()

  target_add_dml(onnxruntime_providers_dml)
  if (GDK_PLATFORM STREQUAL Scarlett)
    target_link_libraries(onnxruntime_providers_dml PRIVATE ${gdk_dx_libs})
  else()
    target_link_libraries(onnxruntime_providers_dml PRIVATE dxguid.lib d3d12.lib dxgi.lib)
  endif()

  target_link_libraries(onnxruntime_providers_dml PRIVATE delayimp.lib)

  if (NOT GDK_PLATFORM)
    set(onnxruntime_DELAYLOAD_FLAGS "${onnxruntime_DELAYLOAD_FLAGS} /DELAYLOAD:DirectML.dll /DELAYLOAD:d3d12.dll /DELAYLOAD:dxgi.dll /DELAYLOAD:api-ms-win-core-com-l1-1-0.dll /DELAYLOAD:shlwapi.dll /DELAYLOAD:oleaut32.dll /ignore:4199")
  endif()

  target_compile_definitions(onnxruntime_providers_dml
    PRIVATE
    ONNX_NAMESPACE=onnx ONNX_ML LOTUS_LOG_THRESHOLD=2 LOTUS_ENABLE_STDERR_LOGGING PLATFORM_WINDOWS
  )
  target_compile_definitions(onnxruntime_providers_dml PRIVATE UNICODE _UNICODE NOMINMAX)
  if (MSVC)
    target_compile_definitions(onnxruntime_providers_dml PRIVATE _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
    target_compile_options(onnxruntime_providers_dml PRIVATE "/W3")
  endif()

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/dml
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers
  )

  set_target_properties(onnxruntime_providers_dml PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_dml PROPERTIES FOLDER "ONNXRuntime")

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_dml
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_MIGRAPHX)
  add_definitions(-DUSE_MIGRAPHX=1)
  set(BUILD_LIBRARY_ONLY 1)
  add_definitions("-DONNX_ML=1")
  add_definitions("-DONNX_NAMESPACE=onnx")
  include_directories(${PROJECT_SOURCE_DIR}/external/protobuf ${PROJECT_SOURCE_DIR}/external/eigen)
  set(MIGRAPHX_ROOT ${onnxruntime_MIGRAPHX_HOME})
  include_directories(${ONNXRUNTIME_ROOT}/../cmake/external/onnx)
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
  endif()
  set(CXX_VERSION_DEFINED TRUE)
  set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
  endif()

  # Add search paths for default rocm installation
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm)

  find_package(hip)
  find_package(migraphx PATHS ${AMD_MIGRAPHX_HOME})

  set(migraphx_libs migraphx::c hip::host)

  file(GLOB_RECURSE onnxruntime_providers_migraphx_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_migraphx_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_migraphx ${onnxruntime_providers_migraphx_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_migraphx onnxruntime_common onnx flatbuffers)
  add_dependencies(onnxruntime_providers_migraphx onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_link_libraries(onnxruntime_providers_migraphx PRIVATE ${migraphx_libs} ${ONNXRUNTIME_PROVIDERS_SHARED} onnx flatbuffers)
  target_include_directories(onnxruntime_providers_migraphx PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/migraphx  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_migraphx PRIVATE ONNXIFI_BUILD_LIBRARY=1)
  target_compile_options(onnxruntime_providers_migraphx PRIVATE -Wno-error=sign-compare)
  set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
  set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/migraphx/version_script.lds -Xlinker --gc-sections")
  target_link_libraries(onnxruntime_providers_migraphx PRIVATE nsync_cpp stdc++fs)

  include(CheckLibraryExists)
  check_library_exists(migraphx::c "migraphx_program_run_async" "/opt/rocm/migraphx/lib" HAS_STREAM_SYNC)
  if(HAS_STREAM_SYNC)
      target_compile_definitions(onnxruntime_providers_migraphx PRIVATE -DMIGRAPHX_STREAM_SYNC)
      message(STATUS "MIGRAPHX GPU STREAM SYNC is ENABLED")
  else()
      message(STATUS "MIGRAPHX GPU STREAM SYNC is DISABLED")
  endif()

  install(TARGETS onnxruntime_providers_migraphx
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

if (onnxruntime_USE_ACL)
  add_definitions(-DUSE_ACL=1)
  file(GLOB_RECURSE onnxruntime_providers_acl_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/acl/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/acl/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_acl_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_acl ${onnxruntime_providers_acl_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_acl
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers
  )
  target_link_libraries(onnxruntime_providers_acl -L$ENV{LD_LIBRARY_PATH})
  add_dependencies(onnxruntime_providers_acl ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_acl PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_acl
    PRIVATE
    ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${onnxruntime_ACL_HOME} ${onnxruntime_ACL_HOME}/include
  )
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/acl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers
  )
  set_target_properties(onnxruntime_providers_acl PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_acl
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_ARMNN)
  add_definitions(-DUSE_ARMNN=1)
  file(GLOB_RECURSE onnxruntime_providers_armnn_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/armnn/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/armnn/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_armnn_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_armnn ${onnxruntime_providers_armnn_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_armnn
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers
  )
  add_dependencies(onnxruntime_providers_armnn ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_armnn PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_armnn PRIVATE
    ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${onnxruntime_ARMNN_HOME} ${onnxruntime_ARMNN_HOME}/include
    ${onnxruntime_ACL_HOME} ${onnxruntime_ACL_HOME}/include
  )
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/armnn
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers
  )
  set_target_properties(onnxruntime_providers_armnn PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_armnn
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_ROCM)
  add_definitions(-DUSE_ROCM=1)
  include(onnxruntime_rocm_hipify.cmake)

  # Add search paths for default hip installation
  list(APPEND CMAKE_PREFIX_PATH ${onnxruntime_ROCM_HOME} ${onnxruntime_ROCM_HOME}/hip ${onnxruntime_ROCM_HOME}/hcc ${onnxruntime_ROCM_HOME}/miopen ${onnxruntime_ROCM_HOME}/hiprand ${onnxruntime_ROCM_HOME}/rocrand)

  set(CMAKE_MODULE_PATH "${onnxruntime_ROCM_HOME}/hip/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP)
  find_package(hiprand REQUIRED)
  find_library(HIP_LIB amdhip64 REQUIRED)
  find_library(ROC_BLAS rocblas REQUIRED)
  find_library(MIOPEN_LIB MIOpen REQUIRED)
  find_library(RCCL_LIB rccl REQUIRED)
  find_library(ROCTRACER_LIB roctracer64 REQUIRED)
  set(ONNXRUNTIME_ROCM_LIBS ${HIP_LIB} ${ROC_BLAS} ${MIOPEN_LIB} ${RCCL_LIB} ${ROCTRACER_LIB})

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
    if (NOT onnxruntime_ENABLE_ATEN)
      list(REMOVE_ITEM onnxruntime_rocm_contrib_ops_cc_srcs
        "${ONNXRUNTIME_ROOT}/contrib_ops/rocm/aten_ops/aten_op.cc"
      )
    endif()

    hipify("onnxruntime/contrib_ops" contrib_ops_excluded_files onnxruntime_rocm_generated_contrib_ops_cc_srcs onnxruntime_rocm_generated_contrib_ops_cu_srcs)

    # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_rocm_contrib_ops_cc_srcs} ${onnxruntime_rocm_contrib_ops_cu_srcs})
    list(APPEND onnxruntime_providers_rocm_src ${onnxruntime_rocm_contrib_ops_cc_srcs} ${onnxruntime_rocm_contrib_ops_cu_srcs})
    list(APPEND onnxruntime_providers_rocm_src ${onnxruntime_rocm_generated_contrib_ops_cc_srcs} ${onnxruntime_rocm_generated_contrib_ops_cu_srcs})
  endif()

  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
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

  set(HIP_CXX_FLAGS -fPIC)
  list(APPEND HIP_CXX_FLAGS -std=c++17)

  if(CMAKE_BUILD_TYPE MATCHES Debug)
      list(APPEND HIP_CXX_FLAGS -g)
      #list(APPEND HIP_CXX_FLAGS -O0)
  endif(CMAKE_BUILD_TYPE MATCHES Debug)

  list(APPEND HIP_CLANG_FLAGS ${HIP_CXX_FLAGS})
  list(APPEND HIP_CLANG_FLAGS ${CMAKE_HIP_FLAGS})

  # Generate GPU code during compilation
  list(APPEND HIP_CLANG_FLAGS -fno-gpu-rdc)

  # Generate GPU code
  foreach(HIP_ARCH ${CMAKE_HIP_ARCHITECTURES})
    list(APPEND HIP_CLANG_FLAGS --offload-arch=${HIP_ARCH})
  endforeach()

  auto_set_source_files_hip_language(${onnxruntime_providers_rocm_src})
  onnxruntime_add_shared_library_module(onnxruntime_providers_rocm ${onnxruntime_providers_rocm_src})

  if(NOT MSVC)
    target_compile_options(onnxruntime_providers_rocm PRIVATE -Wno-sign-compare -D__HIP_PLATFORM_HCC__=1)
    target_compile_options(onnxruntime_providers_rocm PRIVATE -Wno-unused-parameter)
    target_compile_options(onnxruntime_providers_rocm PRIVATE -Wno-undefined-var-template)
  endif()

  onnxruntime_add_include_to_target(onnxruntime_providers_rocm onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)
  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
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
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/rocm  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_rocm PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_rocm PROPERTIES FOLDER "ONNXRuntime")

  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
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

  include(composable_kernel)
  target_link_libraries(onnxruntime_providers_rocm PRIVATE onnxruntime_composable_kernel_includes device_gemm_instance)

  if(UNIX)
    set_property(TARGET onnxruntime_providers_rocm APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/rocm/version_script.lds -Xlinker --gc-sections")
    target_link_libraries(onnxruntime_providers_rocm PRIVATE nsync_cpp)
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

endif()

if (onnxruntime_USE_TVM)
  add_definitions(-DUSE_TVM=1)
  if (onnxruntime_TVM_USE_HASH)
    add_definitions(-DUSE_TVM_HASH=1)
  endif()

  if (onnxruntime_TVM_USE_HASH)
    file (GLOB_RECURSE onnxruntime_providers_tvm_cc_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/providers/tvm/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/tvm/*.cc"
    )
  else()
    file (GLOB onnxruntime_providers_tvm_cc_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/providers/tvm/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/tvm/*.cc"
    )
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_tvm_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_tvm ${onnxruntime_providers_tvm_cc_srcs})

  if ( CMAKE_COMPILER_IS_GNUCC )
    target_compile_options(onnxruntime_providers_tvm PRIVATE -Wno-unused-parameter -Wno-missing-field-initializers)
  endif()

  target_include_directories(onnxruntime_providers_tvm PRIVATE
          ${TVM_INCLUDES}
          ${PYTHON_INLCUDE_DIRS})
  onnxruntime_add_include_to_target(onnxruntime_providers_tvm onnxruntime_common onnx tvm)

  add_dependencies(onnxruntime_providers_tvm ${onnxruntime_EXTERNAL_DEPENDENCIES})

  target_link_libraries(onnxruntime_providers_tvm PRIVATE
      onnx
      tvm
      onnxruntime_common
      onnxruntime_framework
  )
  if (onnxruntime_TVM_USE_HASH)
    add_dependencies(onnxruntime_providers_tvm ippcp_s)
    target_include_directories(onnxruntime_providers_tvm PRIVATE ${IPP_CRYPTO_INCLUDE_DIR})
    target_link_libraries(onnxruntime_providers_tvm PRIVATE ippcp_s)
  endif()

  set_target_properties(onnxruntime_providers_tvm PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_tvm PROPERTIES LINKER_LANGUAGE CXX)

  if (WIN32 AND MSVC)
    # wd4100: identifier' : unreferenced formal parameter
    # wd4127: conditional expression is constant
    # wd4244: conversion from 'int' to 'char', possible loss of data
    target_compile_options(onnxruntime_providers_tvm PRIVATE "/wd4100" "/wd4127" "/wd4244")
  else()
    target_compile_options(onnxruntime_providers_tvm PRIVATE "-Wno-error=type-limits")
  endif()
  target_compile_definitions(onnxruntime_providers_tvm PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/tvm  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_tvm
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_XNNPACK)
  add_compile_definitions(USE_XNNPACK=1)

  file(GLOB_RECURSE onnxruntime_providers_xnnpack_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/providers/xnnpack/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/xnnpack/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/xnnpack/*.cc"
    # utils for handling QDQ models
    "${ONNXRUNTIME_ROOT}/core/providers/shared/node_unit/node_unit.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/node_unit/node_unit.cc"
  )

  source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_providers_xnnpack_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_xnnpack ${onnxruntime_providers_xnnpack_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_xnnpack
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} XNNPACK pthreadpool
  )

  add_dependencies(onnxruntime_providers_xnnpack onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_xnnpack PROPERTIES FOLDER "ONNXRuntime")

  install(DIRECTORY ${ONNXRUNTIME_INCLUDE_DIR}/core/providers/xnnpack
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers
  )

  set_target_properties(onnxruntime_providers_xnnpack PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_xnnpack
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if (onnxruntime_USE_CANN)
  add_definitions(-DUSE_CANN=1)
  file(GLOB_RECURSE onnxruntime_providers_cann_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cann/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cann/*.cc"
  )

  # The shared_library files are in a separate list since they use precompiled headers, and the above files have them disabled.
  file(GLOB_RECURSE onnxruntime_providers_cann_shared_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_cann_cc_srcs} ${onnxruntime_providers_cann_shared_srcs})
  set(onnxruntime_providers_cann_src ${onnxruntime_providers_cann_cc_srcs} ${onnxruntime_providers_cann_shared_srcs})

  onnxruntime_add_shared_library_module(onnxruntime_providers_cann ${onnxruntime_providers_cann_src})
  onnxruntime_add_include_to_target(onnxruntime_providers_cann onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)

  add_dependencies(onnxruntime_providers_cann onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_link_libraries(onnxruntime_providers_cann PRIVATE ascendcl acl_op_compiler nsync_cpp ${ABSEIL_LIBS} onnxruntime_providers_shared)
  target_link_directories(onnxruntime_providers_cann PRIVATE ${onnxruntime_CANN_HOME}/lib64)
  target_include_directories(onnxruntime_providers_cann PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS} ${onnxruntime_CANN_HOME} ${onnxruntime_CANN_HOME}/include)
  set_target_properties(onnxruntime_providers_cann PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_cann PROPERTIES FOLDER "ONNXRuntime")

  install(TARGETS onnxruntime_providers_cann
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
