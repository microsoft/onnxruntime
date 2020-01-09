# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_graph_src CONFIGURE_DEPENDS
  "${ONNXRUNTIME_INCLUDE_DIR}/core/graph/*.h"
  "${ONNXRUNTIME_ROOT}/core/graph/*.h"
  "${ONNXRUNTIME_ROOT}/core/graph/*.cc"
  )

if (onnxruntime_DISABLE_CONTRIB_OPS)
  list(REMOVE_ITEM onnxruntime_graph_src
    "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/*.h"
    "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/*.cc"
    )
endif()

if(NOT onnxruntime_USE_FEATURIZERS)
  file(GLOB_RECURSE featurizers_to_remove_graph_src
    "${ONNXRUNTIME_ROOT}/core/graph/featurizers_ops/*.h"
    "${ONNXRUNTIME_ROOT}/core/graph/featurizers_ops/*.cc"
    )
  foreach(I in ${featurizers_to_remove_graph_src})
    list(REMOVE_ITEM onnxruntime_graph_src ${I})
  endforeach()
endif()

if(NOT onnxruntime_USE_DML)
  list(REMOVE_ITEM onnxruntime_graph_src
    "${ONNXRUNTIME_ROOT}/core/graph/dml_ops/*.h"
    "${ONNXRUNTIME_ROOT}/core/graph/dml_ops/*.cc"
    )
endif()

file(GLOB_RECURSE onnxruntime_ir_defs_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/defs/*.cc"
)

add_library(onnxruntime_graph ${onnxruntime_graph_src} ${onnxruntime_ir_defs_src})
add_dependencies(onnxruntime_graph onnx_proto)
onnxruntime_add_include_to_target(onnxruntime_graph onnxruntime_common onnx onnx_proto protobuf::libprotobuf)

target_include_directories(onnxruntime_graph PRIVATE ${ONNXRUNTIME_ROOT})
set_target_properties(onnxruntime_graph PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_graph PROPERTIES LINKER_LANGUAGE CXX)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/graph  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_graph_src} ${onnxruntime_ir_defs_src})

if (WIN32)
    set(onnxruntime_graph_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )

    set_target_properties(onnxruntime_graph PROPERTIES
        STATIC_LIBRARY_FLAGS "${onnxruntime_graph_static_library_flags}")

    target_compile_options(onnxruntime_graph PRIVATE
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )

    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(onnxruntime_graph PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
endif()
