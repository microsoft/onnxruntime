# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(onnxruntime_language_interop_torch.cmake)

set (CXXOPTS ${PROJECT_SOURCE_DIR}/external/cxxopts/include)

# training lib
file(GLOB_RECURSE onnxruntime_training_srcs
    "${ORTTRAINING_SOURCE_DIR}/core/framework/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/tensorboard/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/tensorboard/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/adasum/*"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/communication/*"
    "${ORTTRAINING_SOURCE_DIR}/core/session/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/session/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/core/agent/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/agent/*.cc"
    )

add_library(onnxruntime_training ${onnxruntime_training_srcs})
add_dependencies(onnxruntime_training onnx tensorboard ${onnxruntime_EXTERNAL_DEPENDENCIES})
onnxruntime_add_include_to_target(onnxruntime_training onnxruntime_common onnx onnx_proto tensorboard protobuf::libprotobuf flatbuffers)

# fix event_writer.cc 4100 warning
if(WIN32)
  target_compile_options(onnxruntime_training PRIVATE /wd4100)
endif()

target_include_directories(onnxruntime_training PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${eigen_INCLUDE_DIRS} ${RE2_INCLUDE_DIR} PUBLIC ${onnxruntime_graph_header} ${MPI_INCLUDE_DIRS})

if (onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime_training PRIVATE ${onnxruntime_CUDNN_HOME}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_training PRIVATE ${NCCL_INCLUDE_DIRS})
endif()

set_target_properties(onnxruntime_training PROPERTIES FOLDER "ONNXRuntime")
source_group(TREE ${ORTTRAINING_ROOT} FILES ${onnxruntime_training_srcs})

# training runner lib
file(GLOB_RECURSE onnxruntime_training_runner_srcs
    "${ORTTRAINING_SOURCE_DIR}/models/runner/*.h"
    "${ORTTRAINING_SOURCE_DIR}/models/runner/*.cc"
)

# perf test utils
set(onnxruntime_perf_test_src_dir ${TEST_SRC_DIR}/perftest)
set(onnxruntime_perf_test_src
"${onnxruntime_perf_test_src_dir}/utils.h")

if(WIN32)
  list(APPEND onnxruntime_perf_test_src
    "${onnxruntime_perf_test_src_dir}/windows/utils.cc")
else ()
  list(APPEND onnxruntime_perf_test_src
    "${onnxruntime_perf_test_src_dir}/posix/utils.cc")
endif()

add_library(onnxruntime_training_runner ${onnxruntime_training_runner_srcs} ${onnxruntime_perf_test_src})
add_dependencies(onnxruntime_training_runner ${onnxruntime_EXTERNAL_DEPENDENCIES} onnx onnxruntime_providers)

onnxruntime_add_include_to_target(onnxruntime_training_runner onnxruntime_training onnxruntime_framework onnxruntime_common onnx onnx_proto protobuf::libprotobuf onnxruntime_training flatbuffers)

target_include_directories(onnxruntime_training_runner PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${eigen_INCLUDE_DIRS} PUBLIC ${onnxruntime_graph_header})
target_link_libraries(onnxruntime_training_runner PRIVATE nlohmann_json::nlohmann_json)
if (onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime_training_runner PUBLIC ${onnxruntime_CUDNN_HOME}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_training_runner PRIVATE ${NCCL_INCLUDE_DIRS})
endif()

if (onnxruntime_USE_ROCM)
  add_definitions(-DUSE_ROCM=1)
  target_include_directories(onnxruntime_training_runner PUBLIC ${onnxruntime_ROCM_HOME}/include)
endif()

check_cxx_compiler_flag(-Wno-maybe-uninitialized HAS_NO_MAYBE_UNINITIALIZED)
if(UNIX AND NOT APPLE)
  if (HAS_NO_MAYBE_UNINITIALIZED)
    target_compile_options(onnxruntime_training_runner PUBLIC "-Wno-maybe-uninitialized")
  endif()
endif()

if (onnxruntime_USE_ROCM)
  target_compile_options(onnxruntime_training_runner PUBLIC -D__HIP_PLATFORM_HCC__=1)
endif()

set_target_properties(onnxruntime_training_runner PROPERTIES FOLDER "ONNXRuntimeTest")
source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_training_runner_srcs} ${onnxruntime_perf_test_src})


# MNIST
file(GLOB_RECURSE training_mnist_src
    "${ORTTRAINING_SOURCE_DIR}/models/mnist/*.h"
    "${ORTTRAINING_SOURCE_DIR}/models/mnist/mnist_data_provider.cc"
    "${ORTTRAINING_SOURCE_DIR}/models/mnist/main.cc"
)
onnxruntime_add_executable(onnxruntime_training_mnist ${training_mnist_src})
onnxruntime_add_include_to_target(onnxruntime_training_mnist onnxruntime_common onnx onnx_proto protobuf::libprotobuf onnxruntime_training flatbuffers)
target_include_directories(onnxruntime_training_mnist PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${eigen_INCLUDE_DIRS} ${CXXOPTS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx onnxruntime_training_runner)

set(ONNXRUNTIME_LIBS
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_ROCM}
    ${PROVIDERS_MKLDNN}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    onnxruntime_flatbuffers
)

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
    list(APPEND ONNXRUNTIME_LIBS onnxruntime_language_interop onnxruntime_pyop)
endif()

if(UNIX AND NOT APPLE)
  if (HAS_NO_MAYBE_UNINITIALIZED)
    target_compile_options(onnxruntime_training_mnist PUBLIC "-Wno-maybe-uninitialized")
  endif()
endif()
target_link_libraries(onnxruntime_training_mnist PRIVATE onnxruntime_training_runner onnxruntime_training ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES})
set_target_properties(onnxruntime_training_mnist PROPERTIES FOLDER "ONNXRuntimeTest")


# squeezenet
# Disabling build for squeezenet, as no one is using this
#[[
file(GLOB_RECURSE training_squeezene_src
    "${ORTTRAINING_SOURCE_DIR}/models/squeezenet/*.h"
    "${ORTTRAINING_SOURCE_DIR}/models/squeezenet/*.cc"
)
onnxruntime_add_executable(onnxruntime_training_squeezenet ${training_squeezene_src})
onnxruntime_add_include_to_target(onnxruntime_training_squeezenet onnxruntime_common onnx onnx_proto protobuf::libprotobuf onnxruntime_training flatbuffers)
target_include_directories(onnxruntime_training_squeezenet PUBLIC ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${eigen_INCLUDE_DIRS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx onnxruntime_training_runner)
if(UNIX AND NOT APPLE)
  target_compile_options(onnxruntime_training_squeezenet PUBLIC "-Wno-maybe-uninitialized")
endif()
target_link_libraries(onnxruntime_training_squeezenet PRIVATE onnxruntime_training_runner onnxruntime_training ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES})
set_target_properties(onnxruntime_training_squeezenet PROPERTIES FOLDER "ONNXRuntimeTest")
]]

# BERT
file(GLOB_RECURSE training_bert_src
    "${ORTTRAINING_SOURCE_DIR}/models/bert/*.h"
    "${ORTTRAINING_SOURCE_DIR}/models/bert/*.cc"
)
onnxruntime_add_executable(onnxruntime_training_bert ${training_bert_src})

if(UNIX AND NOT APPLE)
  if (HAS_NO_MAYBE_UNINITIALIZED)
    target_compile_options(onnxruntime_training_bert PUBLIC "-Wno-maybe-uninitialized")
  endif()
endif()

onnxruntime_add_include_to_target(onnxruntime_training_bert onnxruntime_common onnx onnx_proto protobuf::libprotobuf onnxruntime_training flatbuffers)
target_include_directories(onnxruntime_training_bert PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${MPI_INCLUDE_DIRS} ${eigen_INCLUDE_DIRS} ${CXXOPTS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx onnxruntime_training_runner)

target_link_libraries(onnxruntime_training_bert PRIVATE onnxruntime_training_runner onnxruntime_training ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES})
set_target_properties(onnxruntime_training_bert PROPERTIES FOLDER "ONNXRuntimeTest")

# Pipeline
file(GLOB_RECURSE training_pipeline_poc_src
    "${ORTTRAINING_SOURCE_DIR}/models/pipeline_poc/*.h"
    "${ORTTRAINING_SOURCE_DIR}/models/pipeline_poc/*.cc"
)
onnxruntime_add_executable(onnxruntime_training_pipeline_poc ${training_pipeline_poc_src})

if(UNIX AND NOT APPLE)
  if (HAS_NO_MAYBE_UNINITIALIZED)
    target_compile_options(onnxruntime_training_pipeline_poc PUBLIC "-Wno-maybe-uninitialized")
  endif()
endif()

onnxruntime_add_include_to_target(onnxruntime_training_pipeline_poc onnxruntime_common onnx onnx_proto protobuf::libprotobuf onnxruntime_training flatbuffers)
target_include_directories(onnxruntime_training_pipeline_poc PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${MPI_INCLUDE_DIRS} ${eigen_INCLUDE_DIRS} ${CXXOPTS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx onnxruntime_training_runner)
if (onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_training_pipeline_poc PRIVATE ${NCCL_INCLUDE_DIRS})
endif()

target_link_libraries(onnxruntime_training_pipeline_poc PRIVATE onnxruntime_training_runner onnxruntime_training ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES})
set_target_properties(onnxruntime_training_pipeline_poc PROPERTIES FOLDER "ONNXRuntimeTest")

# GPT-2
file(GLOB_RECURSE training_gpt2_src
    "${ORTTRAINING_SOURCE_DIR}/models/gpt2/*.h"
    "${ORTTRAINING_SOURCE_DIR}/models/gpt2/*.cc"
)
onnxruntime_add_executable(onnxruntime_training_gpt2 ${training_gpt2_src})
if(UNIX AND NOT APPLE)
  if (HAS_NO_MAYBE_UNINITIALIZED)
    target_compile_options(onnxruntime_training_gpt2 PUBLIC "-Wno-maybe-uninitialized")
  endif()
endif()
onnxruntime_add_include_to_target(onnxruntime_training_gpt2 onnxruntime_common onnx onnx_proto protobuf::libprotobuf onnxruntime_training flatbuffers)
target_include_directories(onnxruntime_training_gpt2 PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${MPI_INCLUDE_DIRS} ${eigen_INCLUDE_DIRS} ${CXXOPTS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx onnxruntime_training_runner)

target_link_libraries(onnxruntime_training_gpt2 PRIVATE onnxruntime_training_runner onnxruntime_training ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES})
set_target_properties(onnxruntime_training_gpt2 PROPERTIES FOLDER "ONNXRuntimeTest")
