# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_library(onnxruntime_openenclave INTERFACE)
add_dependencies(onnxruntime_openenclave ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_openenclave INTERFACE ${ONNXRUNTIME_ROOT})

target_link_libraries(onnxruntime_openenclave INTERFACE
    openenclave-enclave
    onnxruntime_session
    onnxruntime_optimizer
    onnxruntime_providers    
    onnxruntime_util
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    onnxruntime_flatbuffers
    ${onnxruntime_EXTERNAL_LIBRARIES}
    )
