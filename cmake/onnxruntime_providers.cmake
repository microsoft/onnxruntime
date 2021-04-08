# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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

file(GLOB onnxruntime_cpu_featurizers_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/featurizers_ops/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/featurizers_ops/cpu/*.cc"
)

file(GLOB onnxruntime_providers_common_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/*.cc"
)

if(onnxruntime_USE_NGRAPH)
  set(PROVIDERS_NGRAPH onnxruntime_providers_ngraph)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES ngraph)
endif()
if(onnxruntime_USE_NUPHAR)
  set(PROVIDERS_NUPHAR onnxruntime_providers_nuphar)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES nuphar)
endif()
if(onnxruntime_USE_VITISAI)
  set(PROVIDERS_VITISAI onnxruntime_providers_vitisai)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES vitisai)
endif()
if(onnxruntime_USE_CUDA)
  set(PROVIDERS_CUDA onnxruntime_providers_cuda)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES cuda)
endif()
if(onnxruntime_USE_TENSORRT)
  set(PROVIDERS_TENSORRT onnxruntime_providers_tensorrt)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES tensorrt)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  set(PROVIDERS_NNAPI onnxruntime_providers_nnapi)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES nnapi)
endif()
if(onnxruntime_USE_RKNPU)
  set(PROVIDERS_RKNPU onnxruntime_providers_rknpu)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES rknpu)
endif()
if(onnxruntime_USE_DML)
  set(PROVIDERS_DML onnxruntime_providers_dml)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES dml)
endif()
if(onnxruntime_USE_MIGRAPHX)
  set(PROVIDERS_MIGRAPHX onnxruntime_providers_migraphx)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES migraphx)
endif()
if(onnxruntime_USE_OPENVINO)
  set(PROVIDERS_OPENVINO onnxruntime_providers_openvino)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES openvino)
endif()
if(onnxruntime_USE_WINML)
  set(PROVIDERS_WINML onnxruntime_providers_winml)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES winml)
endif()
if(onnxruntime_USE_ACL)
  set(PROVIDERS_ACL onnxruntime_providers_acl)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES acl)
endif()
if(onnxruntime_USE_ARMNN)
  set(PROVIDERS_ARMNN onnxruntime_providers_armnn)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES armnn)
endif()
source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})

set(onnxruntime_providers_src ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})

# disable contrib ops conditionally
if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
  # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
  source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cpu_contrib_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_contrib_ops_srcs})
endif()

if (onnxruntime_USE_FEATURIZERS)
  source_group(TREE ${ONNXRUNTIME_ROOT}/ FILES ${onnxruntime_cpu_featurizers_cc_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_featurizers_cc_srcs})
endif()

if (onnxruntime_ENABLE_TRAINING)
  file(GLOB_RECURSE onnxruntime_cpu_training_ops_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.cc"
  )

  if (NOT onnxruntime_USE_HOROVOD)
    list(REMOVE_ITEM onnxruntime_cpu_training_ops_srcs
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/collective/horovod_kernels.cc"
    )
  endif()

  source_group(TREE ${ORTTRAINING_ROOT}/ FILES ${onnxruntime_cpu_training_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_training_ops_srcs})
endif()

add_library(onnxruntime_providers ${onnxruntime_providers_src})
if (MSVC)
   target_compile_options(onnxruntime_providers PRIVATE "/bigobj")
   if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
      target_compile_options(onnxruntime_providers PRIVATE "/wd4244")
   endif()
endif()
onnxruntime_add_include_to_target(onnxruntime_providers onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)

if (onnxruntime_USE_FEATURIZERS)
  add_dependencies(onnxruntime_providers onnxruntime_featurizers)
  onnxruntime_add_include_to_target(onnxruntime_providers onnxruntime_featurizers)
  target_link_libraries(onnxruntime_providers onnxruntime_featurizers)
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

target_include_directories(onnxruntime_providers PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${gemmlowp_src} ${RE2_INCLUDE_DIR})

add_dependencies(onnxruntime_providers onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})

if (onnxruntime_ENABLE_TRAINING)
  target_include_directories(onnxruntime_providers PRIVATE ${ORTTRAINING_ROOT})
  add_dependencies(onnxruntime_providers tensorboard)
  onnxruntime_add_include_to_target(onnxruntime_providers tensorboard)

  if (onnxruntime_USE_HOROVOD)
    target_include_directories(onnxruntime_providers PRIVATE ${HOROVOD_INCLUDE_DIRS})
  endif()
  if (onnxruntime_USE_NCCL OR onnxruntime_USE_HOROVOD)
    target_include_directories(onnxruntime_providers PUBLIC ${MPI_INCLUDE_DIRS})
  endif()
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cpu  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
set_target_properties(onnxruntime_providers PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_providers PROPERTIES FOLDER "ONNXRuntime")

if (onnxruntime_USE_CUDA)
  file(GLOB_RECURSE onnxruntime_providers_cuda_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
  )
  file(GLOB_RECURSE onnxruntime_providers_cuda_cu_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cuh"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs})
  set(onnxruntime_providers_cuda_src ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs})

  # disable contrib ops conditionally
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
  endif()

  if (onnxruntime_ENABLE_TRAINING)
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

    if (NOT onnxruntime_USE_HOROVOD)
      list(REMOVE_ITEM onnxruntime_cuda_training_ops_cc_srcs
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/horovod_kernels.cc"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/ready_event.cc"
      )
    endif()

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

  add_library(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_src})

  # Build GPU execution provider with Pytorch's C++ APIs.
  if (onnxruntime_USE_TORCH)
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>")
    target_include_directories(onnxruntime_providers_cuda PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(onnxruntime_providers_cuda PRIVATE onnxruntime_training ${TORCH_LIBRARIES})
  endif()

  if (UNIX)
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-reorder>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-reorder>")
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
  else()
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>")
  endif()
  onnxruntime_add_include_to_target(onnxruntime_providers_cuda onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
  if (onnxruntime_ENABLE_TRAINING)
    onnxruntime_add_include_to_target(onnxruntime_providers_cuda onnxruntime_training)
    target_link_libraries(onnxruntime_providers_cuda PRIVATE onnxruntime_training)
  endif()
  add_dependencies(onnxruntime_providers_cuda ${onnxruntime_EXTERNAL_DEPENDENCIES} ${onnxruntime_tvm_dependencies})
  target_include_directories(onnxruntime_providers_cuda PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_CUDNN_HOME}/include ${eigen_INCLUDE_DIRS} ${TVM_INCLUDES} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cuda  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_cuda PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(onnxruntime_providers_cuda PROPERTIES FOLDER "ONNXRuntime")

  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
    target_include_directories(onnxruntime_providers_cuda PRIVATE ${PROJECT_SOURCE_DIR}/external/cub)
  endif()

  if (onnxruntime_ENABLE_TRAINING)
    target_include_directories(onnxruntime_providers_cuda PRIVATE ${ORTTRAINING_ROOT} ${MPI_INCLUDE_DIRS})

    if (onnxruntime_USE_HOROVOD)
      target_include_directories(onnxruntime_providers_cuda PRIVATE ${HOROVOD_INCLUDE_DIRS})
    endif()

    if (onnxruntime_USE_NCCL)
      target_include_directories(onnxruntime_providers_cuda PRIVATE ${NCCL_INCLUDE_DIRS})
    endif()
  endif()

  if (WIN32)
    # *.cu cannot use PCH
    foreach(src_file ${onnxruntime_providers_cuda_cc_srcs})
      set_source_files_properties(${src_file}
        PROPERTIES
        COMPILE_FLAGS "/Yucuda_pch.h /FIcuda_pch.h")
    endforeach()
    if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
      foreach(src_file ${onnxruntime_cuda_contrib_ops_cc_srcs})
        set_source_files_properties(${src_file}
          PROPERTIES
          COMPILE_FLAGS "/Yucuda_pch.h /FIcuda_pch.h")
      endforeach()
    endif()
    set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.cc"
      PROPERTIES
      COMPILE_FLAGS "/Yccuda_pch.h"
    )
    # disable a warning from the CUDA headers about unreferenced local functions
    #target_compile_options(onnxruntime_providers_cuda PRIVATE /wd4505)
    if (onnxruntime_USE_TVM)
      target_compile_options(onnxruntime_providers_cuda PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
    endif()
    set(onnxruntime_providers_cuda_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(onnxruntime_providers_cuda PROPERTIES
        STATIC_LIBRARY_FLAGS "${onnxruntime_providers_cuda_static_library_flags}")
  endif()
endif()

if (onnxruntime_USE_TENSORRT OR onnxruntime_USE_DNNL)
  file(GLOB_RECURSE onnxruntime_providers_shared_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_shared_cc_srcs})
  add_library(onnxruntime_providers_shared SHARED ${onnxruntime_providers_shared_cc_srcs})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/shared  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_shared PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_shared PROPERTIES LINKER_LANGUAGE CXX)

  if(APPLE)
    set_property(TARGET onnxruntime_providers_shared APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/shared/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_shared APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/shared/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_shared APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/shared/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_shared unknown platform, need to specify shared library exports for it")
  endif()

endif()

if (onnxruntime_USE_DNNL)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES dnnl)

  file(GLOB_RECURSE onnxruntime_providers_dnnl_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/dnnl/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/dnnl/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_dnnl_cc_srcs})
  add_library(onnxruntime_providers_dnnl SHARED ${onnxruntime_providers_dnnl_cc_srcs})
  target_link_directories(onnxruntime_providers_dnnl PRIVATE ${DNNL_LIB_DIR})
  onnxruntime_add_include_to_target(onnxruntime_providers_dnnl onnxruntime_common onnx) # onnx needed for stl_backports.h
  add_dependencies(onnxruntime_providers_dnnl onnxruntime_providers_shared project_dnnl ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_dnnl PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS} ${DNNL_INCLUDE_DIR})
  # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
  target_link_libraries(onnxruntime_providers_dnnl PRIVATE dnnl onnxruntime_providers_shared)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/dnnl  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_dnnl PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_dnnl PROPERTIES LINKER_LANGUAGE CXX)

  if(APPLE)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/dnnl/exported_symbols.lst")
    target_link_libraries(onnxruntime_providers_dnnl PRIVATE nsync_cpp)
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/dnnl/version_script.lds -Xlinker --gc-sections")
    target_link_libraries(onnxruntime_providers_dnnl PRIVATE nsync_cpp)
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/dnnl/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_dnnl unknown platform, need to specify shared library exports for it")
  endif()

endif()

if (onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT=1)
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4244 /wd4267 /wd4099 /wd4551 /wd4505 /wd4515 /wd4706 /wd4456 /wd4324 /wd4701 /wd4804 /wd4702")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4805")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -include algorithm")
    set(PROTOBUF_LIBRARY libprotobuf)
    set(DISABLED_WARNINGS_FOR_TRT /wd4267 /wd4244 /wd4996 /wd4456)
  endif()
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
  endif()
  set(CXX_VERSION_DEFINED TRUE)
  add_subdirectory(${ONNXRUNTIME_ROOT}/../cmake/external/onnx-tensorrt)
  set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
  if (WIN32)
    set(CMAKE_CUDA_FLAGS ${OLD_CMAKE_CUDA_FLAGS})
    unset(PROTOBUF_LIBRARY)
    unset(OLD_CMAKE_CXX_FLAGS)
    unset(OLD_CMAKE_CUDA_FLAGS)
    set_target_properties(nvonnxparser PROPERTIES LINK_FLAGS "/ignore:4199")
    target_sources(onnx2trt PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/getopt.cc)
    target_sources(getSupportedAPITest PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/getopt.cc)
    target_include_directories(onnx2trt PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/include)
    target_include_directories(getSupportedAPITest PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/include)
    target_compile_options(nvonnxparser_static PRIVATE /FIio.h /wd4100)
    target_compile_options(nvonnxparser PRIVATE /FIio.h /wd4100)
    target_compile_options(onnx2trt PRIVATE /FIio.h /wd4100)
    target_compile_options(getSupportedAPITest PRIVATE /FIio.h /wd4100)
  endif()
  include_directories(${ONNXRUNTIME_ROOT}/../cmake/external/onnx-tensorrt)
  include_directories(${TENSORRT_INCLUDE_DIR})
  set(trt_link_libs cudnn ${CMAKE_DL_LIBS} ${TENSORRT_LIBRARY})
  set(onnxparser_link_libs nvonnxparser_static)

  file(GLOB_RECURSE onnxruntime_providers_tensorrt_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_tensorrt_cc_srcs})
  add_library(onnxruntime_providers_tensorrt SHARED ${onnxruntime_providers_tensorrt_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_tensorrt onnxruntime_common onnx )
  add_dependencies(onnxruntime_providers_tensorrt onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  if(WIN32)
    target_link_directories(onnxruntime_providers_tensorrt PRIVATE ${onnxruntime_CUDA_HOME}/x64/lib64)
  else()
    target_link_directories(onnxruntime_providers_tensorrt PRIVATE ${onnxruntime_CUDA_HOME}/lib64)
  endif()
  target_link_libraries(onnxruntime_providers_tensorrt PRIVATE ${onnxparser_link_libs} ${trt_link_libs} cudart onnxruntime_providers_shared protobuf::libprotobuf)
  target_include_directories(onnxruntime_providers_tensorrt PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${onnxruntime_CUDNN_HOME}/include ${eigen_INCLUDE_DIRS} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/tensorrt  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_tensorrt PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_tensorrt PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_tensorrt PRIVATE ONNXIFI_BUILD_LIBRARY=1)
  target_compile_options(onnxruntime_providers_tensorrt PRIVATE ${DISABLED_WARNINGS_FOR_TRT})
  if (WIN32)
    target_compile_options(onnxruntime_providers_tensorrt INTERFACE /wd4267 /wd4244 /wd4996 /wd4456)
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
endif()

if (onnxruntime_USE_NGRAPH)
  include_directories("${CMAKE_CURRENT_BINARY_DIR}/onnx")
  file(GLOB_RECURSE onnxruntime_providers_ngraph_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/ngraph/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/ngraph/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_ngraph_cc_srcs})
  add_library(onnxruntime_providers_ngraph ${onnxruntime_providers_ngraph_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_ngraph onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
  add_dependencies(onnxruntime_providers_ngraph project_ngraph onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_ngraph PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_ngraph PRIVATE ${ONNXRUNTIME_ROOT} ${ngraph_INCLUDE_DIRS})
  set_target_properties(onnxruntime_providers_ngraph PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT MSVC)
    target_compile_options(onnxruntime_providers_ngraph PRIVATE "SHELL:-Wformat" "SHELL:-Wformat-security" "SHELL:-fstack-protector-strong" "SHELL:-D_FORTIFY_SOURCE=2")
    target_link_options(onnxruntime_providers_ngraph PRIVATE "LINKER:-z, noexecstack " "LINKER:-z relro" "LINKER:-z now" "LINKER:-pie")
  endif()
endif()

if (onnxruntime_USE_NUPHAR)
  add_definitions(-DUSE_NUPHAR=1)

  if (NOT onnxruntime_USE_TVM)
    message(FATAL_ERROR "onnxruntime_USE_TVM required for onnxruntime_USE_NUPHAR")
  endif()

  if (NOT onnxruntime_USE_LLVM)
    message(FATAL_ERROR "onnxruntime_USE_LLVM required for onnxruntime_USE_NUPHAR")
  endif()

  include(onnxruntime_nuphar_extern.cmake)

  file(GLOB_RECURSE onnxruntime_providers_nuphar_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/nuphar/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/nuphar/*.cc"
  )

  # following files required different build flag for AVX2 in separate onnxruntime_nuphar_extern.cmake file
  list (REMOVE_ITEM onnxruntime_providers_nuphar_cc_srcs "${ONNXRUNTIME_ROOT}/core/providers/nuphar/extern/igemv_avx2.cc")
  list (REMOVE_ITEM onnxruntime_providers_nuphar_cc_srcs "${ONNXRUNTIME_ROOT}/core/providers/nuphar/extern/igemv_avx2.h")

  if (onnxruntime_USE_MKLML)
    add_definitions(-DNUPHAR_USE_MKL)
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nuphar_cc_srcs})
  add_library(onnxruntime_providers_nuphar ${onnxruntime_providers_nuphar_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_nuphar onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
  set_target_properties(onnxruntime_providers_nuphar PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_nuphar PRIVATE ${ONNXRUNTIME_ROOT} ${TVM_INCLUDES} ${eigen_INCLUDE_DIRS})
  set_target_properties(onnxruntime_providers_nuphar PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_options(onnxruntime_providers_nuphar PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
  add_dependencies(onnxruntime_providers_nuphar ${onnxruntime_EXTERNAL_DEPENDENCIES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/nuphar  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
endif()

if (onnxruntime_USE_VITISAI)
  file(GLOB_RECURSE onnxruntime_providers_vitisai_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_vitisai_cc_srcs})
  add_library(onnxruntime_providers_vitisai ${onnxruntime_providers_vitisai_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_vitisai onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
  add_dependencies(onnxruntime_providers_vitisai ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_vitisai PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_vitisai PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${VITISAI_INCLUDE_DIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/vitisai  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_vitisai PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (onnxruntime_USE_OPENVINO)

  include_directories("${CMAKE_CURRENT_BINARY_DIR}/onnx")
  file(GLOB_RECURSE onnxruntime_providers_openvino_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.hpp"
    "${ONNXRUNTIME_ROOT}/core/providers/openvino/*.cpp"
  )

  if (onnxruntime_USE_OPENVINO_BINARY)

    # Header paths
    list(APPEND OPENVINO_INCLUDE_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/include)
    list(APPEND OPENVINO_INCLUDE_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/tbb/include)
    list(APPEND OPENVINO_INCLUDE_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/mkltiny_lnx/include)

    # Library paths
    list(APPEND OPENVINO_LIB_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/tbb/lib)
    list(APPEND OPENVINO_LIB_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/mkltiny_lnx/lib)

    # Lib names
    if (WIN32)
      list(APPEND OPENVINO_LIB_LIST inference_engine.lib inference_engine_legacy.lib tbb.lib ${PYTHON_LIBRARIES})
    else()
      list(APPEND OPENVINO_LIB_LIST -linference_engine -linference_engine_legacy -ltbb ${PYTHON_LIBRARIES})
    endif()

    if ((OPENVINO_VERSION VERSION_GREATER_EQUAL "2020.3") OR (WIN32))
      # Link to nGraph from OpenVINO installation
      list(APPEND OPENVINO_INCLUDE_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/ngraph/include)
      list(APPEND OPENVINO_LIB_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/ngraph/lib)
      if (WIN32)
        list(APPEND OPENVINO_LIB_LIST ngraph.lib)
      else()
        list(APPEND OPENVINO_LIB_LIST -lngraph)
      endif()
      if (OPENVINO_VERSION VERSION_GREATER_EQUAL "2020.4")
        if (WIN32)
          list(APPEND OPENVINO_LIB_LIST onnx_importer.lib)
        else()
          list(APPEND OPENVINO_LIB_LIST -lonnx_importer)
        endif()
      endif()
    else ()
      # Link to locally built nGraph
      list(APPEND OPENVINO_INCLUDE_DIR_LIST ${ngraph_INCLUDE_DIRS})
    endif()

    if(WIN32)
      list(APPEND OPENVINO_LIB_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/Release)
    else()
      list(APPEND OPENVINO_LIB_DIR_LIST $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64)
    endif()

  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_openvino_cc_srcs})
  add_library(onnxruntime_providers_openvino ${onnxruntime_providers_openvino_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_openvino onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES FOLDER "ONNXRuntime")
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/openvino  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_openvino PROPERTIES LINKER_LANGUAGE CXX)
  link_directories(onnxruntime_providers_openvino ${OPENVINO_LIB_DIR_LIST})
  add_dependencies(onnxruntime_providers_openvino onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_openvino SYSTEM PUBLIC ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${OPENVINO_INCLUDE_DIR_LIST} ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(onnxruntime_providers_openvino ${OPENVINO_LIB_LIST})

  if(MSVC)
    target_compile_options(onnxruntime_providers_openvino PUBLIC /wd4275 /wd4100 /wd4005 /wd4244 /wd4267)
  endif()

endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  add_definitions(-DUSE_NNAPI=1)
  file(GLOB
    onnxruntime_providers_nnapi_cc_srcs_top CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/nnapi/*.cc"
  )
  file(GLOB_RECURSE
    onnxruntime_providers_nnapi_cc_srcs_nested CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/nnapi/nnapi_builtin/*.cc"
  )
  set(onnxruntime_providers_nnapi_cc_srcs ${onnxruntime_providers_nnapi_cc_srcs_top} ${onnxruntime_providers_nnapi_cc_srcs_nested})
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nnapi_cc_srcs})
  add_library(onnxruntime_providers_nnapi ${onnxruntime_providers_nnapi_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_nnapi onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers)
  target_link_libraries(onnxruntime_providers_nnapi)
  add_dependencies(onnxruntime_providers_nnapi onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_nnapi PRIVATE ${ONNXRUNTIME_ROOT} ${nnapi_INCLUDE_DIRS})
  set_target_properties(onnxruntime_providers_nnapi PROPERTIES LINKER_LANGUAGE CXX)
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
  add_library(onnxruntime_providers_rknpu ${onnxruntime_providers_rknpu_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_rknpu onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers)
  target_link_libraries(onnxruntime_providers_rknpu PRIVATE -lrknpu_ddk)
  add_dependencies(onnxruntime_providers_rknpu onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_rknpu PROPERTIES CXX_STANDARD 14)
  set_target_properties(onnxruntime_providers_rknpu PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_rknpu PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_rknpu PRIVATE ${ONNXRUNTIME_ROOT} ${rknpu_INCLUDE_DIRS} ${RKNPU_DDK_INCLUDE_DIR})
  link_directories(onnxruntime_providers_rknpu ${RKNPU_DDK_LIB_DIR})
  set_target_properties(onnxruntime_providers_rknpu PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (onnxruntime_USE_DML)
  file(GLOB_RECURSE onnxruntime_providers_dml_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/dml/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/dml/*.cpp"
    "${ONNXRUNTIME_ROOT}/core/providers/dml/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_dml_cc_srcs})
  add_library(onnxruntime_providers_dml ${onnxruntime_providers_dml_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_dml onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
  add_dependencies(onnxruntime_providers_dml ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_dml PRIVATE ${ONNXRUNTIME_ROOT} ${ONNXRUNTIME_ROOT}/../cmake/external/wil/include)

  add_definitions(-DDML_TARGET_VERSION_USE_LATEST=1)

  if (NOT onnxruntime_USE_CUSTOM_DIRECTML)
    if(NOT onnxruntime_target_platform STREQUAL "x86" AND NOT onnxruntime_target_platform STREQUAL "x64")
      message(FATAL_ERROR "Target platform ${onnxruntime_target_platform} is not supported by DML")
    endif()

    foreach(file "DirectML.dll" "DirectML.pdb" "DirectML.Debug.dll" "DirectML.Debug.pdb")
      add_custom_command(TARGET onnxruntime_providers_dml
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${DML_PACKAGE_DIR}/bin/${onnxruntime_target_platform}/${file}" $<TARGET_FILE_DIR:onnxruntime_providers_dml>)
    endforeach()
  endif()

  function(target_add_dml target)
    if (onnxruntime_USE_CUSTOM_DIRECTML)
      target_link_libraries(${target} PRIVATE DirectML)
    else()
      add_dependencies(${target} RESTORE_PACKAGES)
      target_link_libraries(${target} PRIVATE "${DML_PACKAGE_DIR}/bin/${onnxruntime_target_platform}/DirectML.lib")
	  target_compile_definitions(${target} PRIVATE DML_TARGET_VERSION_USE_LATEST)
    endif()
  endfunction()

  target_add_dml(onnxruntime_providers_dml)
  target_link_libraries(onnxruntime_providers_dml PRIVATE d3d12.lib dxgi.lib)

  if (WINDOWS_STORE)
    target_link_libraries(onnxruntime_providers_dml PRIVATE dloadhelper.lib)
  else()
    target_link_libraries(onnxruntime_providers_dml PRIVATE delayimp.lib)
  endif()

  set(onnxruntime_DELAYLOAD_FLAGS "${onnxruntime_DELAYLOAD_FLAGS} /DELAYLOAD:DirectML.dll /DELAYLOAD:d3d12.dll /DELAYLOAD:dxgi.dll /ignore:4199")

  # The DML EP requires C++17
  set_target_properties(onnxruntime_providers_dml PROPERTIES CXX_STANDARD 17)
  set_target_properties(onnxruntime_providers_dml PROPERTIES CXX_STANDARD_REQUIRED ON)

  target_compile_definitions(onnxruntime_providers_dml PRIVATE ONNX_NAMESPACE=onnx ONNX_ML LOTUS_LOG_THRESHOLD=2 LOTUS_ENABLE_STDERR_LOGGING PLATFORM_WINDOWS)
  target_compile_definitions(onnxruntime_providers_dml PRIVATE UNICODE _UNICODE NOMINMAX)
  if (MSVC)
    target_compile_definitions(onnxruntime_providers_dml PRIVATE _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
    target_compile_options(onnxruntime_providers_dml PRIVATE "/W3")
  endif()

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/dml  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)

  set_target_properties(onnxruntime_providers_dml PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_dml PROPERTIES FOLDER "ONNXRuntime")
endif()

if (onnxruntime_USE_MIGRAPHX)
  # Add search paths for default rocm installation
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm)

  find_package(hip)
  find_package(migraphx PATHS ${AMD_MIGRAPHX_HOME})

  set(migraphx_libs migraphx::c hip::host)

  file(GLOB_RECURSE onnxruntime_providers_migraphx_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_migraphx_cc_srcs})
  add_library(onnxruntime_providers_migraphx ${onnxruntime_providers_migraphx_cc_srcs})
  target_link_libraries(onnxruntime_providers_migraphx PRIVATE ${migraphx_libs})
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES FOLDER "ONNXRuntime")
  target_compile_options(onnxruntime_providers_migraphx PRIVATE -Wno-error=sign-compare)
  target_include_directories(onnxruntime_providers_migraphx PRIVATE ${ONNXRUNTIME_ROOT})
  onnxruntime_add_include_to_target(onnxruntime_providers_migraphx onnxruntime_common onnxruntime_framework onnx flatbuffers)
  add_dependencies(onnxruntime_providers_migraphx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/migraphx  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (onnxruntime_USE_ACL)
  add_definitions(-DUSE_ACL=1)
  file(GLOB_RECURSE onnxruntime_providers_acl_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/acl/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/acl/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_acl_cc_srcs})
  add_library(onnxruntime_providers_acl ${onnxruntime_providers_acl_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_acl onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
  target_link_libraries(onnxruntime_providers_acl -L$ENV{LD_LIBRARY_PATH})
  add_dependencies(onnxruntime_providers_acl ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_acl PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_acl PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${ACL_INCLUDE_DIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/acl  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_acl PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (onnxruntime_USE_ARMNN)
  add_definitions(-DUSE_ARMNN=1)
  file(GLOB_RECURSE onnxruntime_providers_armnn_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/armnn/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/armnn/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_armnn_cc_srcs})
  add_library(onnxruntime_providers_armnn ${onnxruntime_providers_armnn_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_armnn onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
  add_dependencies(onnxruntime_providers_armnn ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_armnn PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_armnn PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${ARMNN_INCLUDE_DIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/armnn  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_armnn PROPERTIES LINKER_LANGUAGE CXX)
endif()
