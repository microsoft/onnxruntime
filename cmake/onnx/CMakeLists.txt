# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_library(onnx_proto ${ONNXRUNTIME_ROOT}/core/protobuf/onnx-ml.proto ${ONNXRUNTIME_ROOT}/core/protobuf/onnx-operators-ml.proto)
target_include_directories(onnx_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}/..")
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS TARGET onnx_proto)

# Cpp Tests were added and they require googletest
# since we have our own copy, try using that
set(ONNX_SOURCE_ROOT ${PROJECT_SOURCE_DIR}/external/onnx)
file(GLOB_RECURSE onnx_src
    "${ONNX_SOURCE_ROOT}/onnx/*.h"
    "${ONNX_SOURCE_ROOT}/onnx/*.cc"
)

file(GLOB_RECURSE onnx_exclude_src
    "${ONNX_SOURCE_ROOT}/onnx/py_utils.h"
    "${ONNX_SOURCE_ROOT}/onnx/proto_utils.h"
    "${ONNX_SOURCE_ROOT}/onnx/backend/test/cpp/*"
    "${ONNX_SOURCE_ROOT}/onnx/test/*"
    "${ONNX_SOURCE_ROOT}/onnx/cpp2py_export.cc"
)

list(REMOVE_ITEM onnx_src ${onnx_exclude_src})
add_library(onnx ${onnx_src})
add_dependencies(onnx onnx_proto)
target_include_directories(onnx PUBLIC $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES> "${ONNX_SOURCE_ROOT}")
target_compile_definitions(onnx PUBLIC "ONNX_ML" "ONNX_NAMESPACE=onnx")
if (WIN32)
    target_compile_options(onnx PRIVATE
        /wd4800 # 'type' : forcing value to bool 'true' or 'false' (performance warning)
        /wd4125 # decimal digit terminates octal escape sequence
        /wd4100 # 'param' : unreferenced formal parameter
        /wd4244 # 'argument' conversion from 'google::protobuf::int64' to 'int', possible loss of data
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
    set(onnx_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(onnx PROPERTIES
        STATIC_LIBRARY_FLAGS "${onnx_static_library_flags}")
else()
  if(HAS_UNUSED_PARAMETER)
    target_compile_options(onnx PRIVATE "-Wno-unused-parameter")
    target_compile_options(onnx_proto PRIVATE "-Wno-unused-parameter")
  endif()
  if(HAS_UNUSED_BUT_SET_VARIABLE)
    target_compile_options(onnx PRIVATE "-Wno-unused-but-set-variable")
    target_compile_options(onnx_proto PRIVATE "-Wno-unused-but-set-variable")
  endif()
endif()
