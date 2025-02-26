# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_lora_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/lora_format/*.h"
    "${ONNXRUNTIME_ROOT}/lora/*.h"
    "${ONNXRUNTIME_ROOT}/lora/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_lora_srcs})

onnxruntime_add_static_library(onnxruntime_lora ${onnxruntime_lora_srcs})
onnxruntime_add_include_to_target(onnxruntime_lora onnx flatbuffers::flatbuffers Boost::mp11 ${GSL_TARGET})
target_link_libraries(onnxruntime_lora onnxruntime_framework)

if(onnxruntime_ENABLE_INSTRUMENT)
  target_compile_definitions(onnxruntime_lora PUBLIC ONNXRUNTIME_ENABLE_INSTRUMENT)
endif()

target_include_directories(onnxruntime_lora PRIVATE ${ONNXRUNTIME_ROOT})
add_dependencies(onnxruntime_lora ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_lora PROPERTIES FOLDER "ONNXRuntime")

if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_lora
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
