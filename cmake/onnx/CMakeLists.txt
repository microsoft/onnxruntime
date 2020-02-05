# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#TODO: if protobuf is a shared lib and onnxruntime_USE_FULL_PROTOBUF is ON, then onnx_proto should be built as a shared lib instead of a static lib. Otherwise any code outside onnxruntime.dll can't use onnx protobuf definitions if they share the protobuf.dll with onnxruntime. For example, if protobuf is a shared lib and onnx_proto is a static lib then onnxruntime_perf_test won't work.

add_library(onnx_proto ${ONNXRUNTIME_ROOT}/core/protobuf/onnx-ml.proto ${ONNXRUNTIME_ROOT}/core/protobuf/onnx-operators-ml.proto)
target_include_directories(onnx_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}/..")
target_compile_definitions(onnx_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_COMPILE_DEFINITIONS>)
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${ONNXRUNTIME_ROOT}/core/protobuf TARGET onnx_proto)
if (WIN32)
  target_compile_options(onnx_proto PRIVATE "/wd4146" "/wd4125" "/wd4456" "/wd4267" "/wd4309")
else()
  if(HAS_UNUSED_VARIABLE)
    target_compile_options(onnx_proto PRIVATE "-Wno-unused-variable")
  endif()
endif()
# Cpp Tests were added and they require googletest
# since we have our own copy, try using that
set(ONNX_SOURCE_ROOT ${PROJECT_SOURCE_DIR}/external/onnx)
file(GLOB_RECURSE onnx_src CONFIGURE_DEPENDS
    "${ONNX_SOURCE_ROOT}/onnx/*.h"
    "${ONNX_SOURCE_ROOT}/onnx/*.cc"
)

file(GLOB_RECURSE onnx_exclude_src CONFIGURE_DEPENDS
    "${ONNX_SOURCE_ROOT}/onnx/py_utils.h"
    "${ONNX_SOURCE_ROOT}/onnx/proto_utils.h"
    "${ONNX_SOURCE_ROOT}/onnx/backend/test/cpp/*"
    "${ONNX_SOURCE_ROOT}/onnx/test/*"
    "${ONNX_SOURCE_ROOT}/onnx/cpp2py_export.cc"
)

if (MSVC)
  SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Gw /GL")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Gw /GL")
endif()

list(REMOVE_ITEM onnx_src ${onnx_exclude_src})
add_library(onnx ${onnx_src})
add_dependencies(onnx onnx_proto)
target_include_directories(onnx PUBLIC "${ONNX_SOURCE_ROOT}")
target_include_directories(onnx PUBLIC $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(onnx PUBLIC $<TARGET_PROPERTY:onnx_proto,INTERFACE_COMPILE_DEFINITIONS> PRIVATE "__ONNX_DISABLE_STATIC_REGISTRATION")
if (onnxruntime_USE_FULL_PROTOBUF)
  target_compile_definitions(onnx PUBLIC "ONNX_ML" "ONNX_NAMESPACE=onnx")
else()
  target_compile_definitions(onnx PUBLIC "ONNX_ML" "ONNX_NAMESPACE=onnx" "ONNX_USE_LITE_PROTO" "__ONNX_NO_DOC_STRINGS")
endif()
if (WIN32)
    target_compile_options(onnx PRIVATE
        /wd4800 # 'type' : forcing value to bool 'true' or 'false' (performance warning)
        /wd4125 # decimal digit terminates octal escape sequence
        /wd4100 # 'param' : unreferenced formal parameter
        /wd4244 # 'argument' conversion from 'google::protobuf::int64' to 'int', possible loss of data
        /EHsc   # exception handling - C++ may throw, extern "C" will not
        /wd4996 # 'argument' Using double parameter version instead of single parameter version of SetTotalBytesLimit(). The second parameter is ignored.
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


set_target_properties(onnx PROPERTIES FOLDER "External/ONNX")
set_target_properties(onnx_proto PROPERTIES FOLDER "External/ONNX")
