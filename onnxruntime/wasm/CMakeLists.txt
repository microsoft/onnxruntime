file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        )
add_library(onnxruntime_wasm STATIC)
target_sources(onnxruntime_wasm PRIVATE ${SOURCES})
