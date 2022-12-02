# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FetchContent_Declare(
  tensorboard
  URL ${DEP_URL_tensorboard}
  URL_HASH SHA1=${DEP_SHA1_tensorboard}
)
FetchContent_MakeAvailable(tensorboard)

set(TENSORBOARD_ROOT ${tensorboard_SOURCE_DIR})

# tensorboard protos
file(GLOB_RECURSE tensorboard_proto_srcs CONFIGURE_DEPENDS
  "${TENSORBOARD_ROOT}/tensorboard/compat/proto/*.proto"
)

add_library(tensorboard STATIC ${tensorboard_proto_srcs})
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${tensorboard_SOURCE_DIR} ${protobuf_SOURCE_DIR}/src TARGET tensorboard)
onnxruntime_add_include_to_target(tensorboard ${PROTOBUF_LIB})
target_include_directories(tensorboard PRIVATE ${PROJECT_BINARY_DIR})
add_dependencies(tensorboard ${onnxruntime_EXTERNAL_DEPENDENCIES})
if(WIN32)
  target_compile_options(tensorboard PRIVATE "/wd4100" "/wd4125" "/wd4127" "/wd4267" "/wd4456" "/wd4800" "/wd6011" "/wd6387" "/wd28182")
endif()
