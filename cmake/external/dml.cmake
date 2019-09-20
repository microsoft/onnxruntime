
if (onnxruntime_BUILD_x86)
    set(dml_LIB_DIR "${PROJECT_SOURCE_DIR}/external/dml/x86")
else ()
    set(dml_LIB_DIR "${PROJECT_SOURCE_DIR}/external/dml/x64")
endif()

set(dml_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/dml")

include_directories(${dml_INCLUDE_DIR})
link_directories(${dml_LIB_DIR})
