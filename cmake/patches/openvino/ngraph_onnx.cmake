add_definitions(-DONNX_BUILD_SHARED_LIBS=ON)
add_definitions(-DONNX_ML=ON)

set(BINARY_DIR "${prebuilt_ONNX_BINARY_DIR}/onnx")
set(ONNX_INCLUDE_DIR ${prebuilt_ONNX_BINARY_DIR})
set(ONNX_SOURCE_INCLUDE_DIR "${prebuilt_ONNX_SOURCE_DIR}")
set(ONNX_PROTO_INCLUDE_DIR ${ONNX_INCLUDE_DIR})
include_directories("${ONNX_SOURCE_INCLUDE_DIR}")
include_directories("${ONNX_PROTO_INCLUDE_DIR}")
if (WIN32)
    set(ONNX_LIBRARY ${BINARY_DIR}/${CMAKE_BUILD_TYPE}/onnx.lib)
    set(ONNX_PROTO_LIBRARY ${BINARY_DIR}/${CMAKE_BUILD_TYPE}/onnx_proto.lib)
else()
    set(ONNX_LIBRARY ${BINARY_DIR}/libonnx.a)
    set(ONNX_PROTO_LIBRARY ${BINARY_DIR}/libonnx_proto.a)
endif()
set(ONNX_LIBRARIES ${ONNX_LIBRARY} ${ONNX_PROTO_LIBRARY})

if (NOT TARGET onnx)
    add_library(onnx UNKNOWN IMPORTED)
    set_target_properties(onnx PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${ONNX_INCLUDE_DIR}
            IMPORTED_LOCATION ${ONNX_LIBRARY}
            INCLUDE_DIRECTORIES "${ONNX_SOURCE_INCLUDE_DIR}"
            INTERFACE_COMPILE_DEFINITIONS ONNX_ML=1)
endif()

if (NOT TARGET onnx_proto)
    add_library(onnx_proto UNKNOWN IMPORTED)
    set_target_properties(onnx_proto PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${ONNX_PROTO_INCLUDE_DIR}
            IMPORTED_LOCATION ${ONNX_PROTO_LIBRARY}
            INCLUDE_DIRECTORIES "${ONNX_SOURCE_INCLUDE_DIR}"
            INTERFACE_COMPILE_DEFINITIONS ONNX_ML=1)
endif()
add_library(ext_onnx UNKNOWN IMPORTED)
add_dependencies(ext_onnx onnx)
