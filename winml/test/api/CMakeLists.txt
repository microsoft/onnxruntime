# Compiler options
add_definitions(/bigobj)

# Specify the usage of a precompiled header
target_precompiled_header(winml_test_api precomp.h)

# Includes
target_include_directories(winml_test_api PRIVATE ${CMAKE_BINARY_DIR}/engine/lotus)                             # windows machine learning component headers
target_include_directories(winml_test_api PRIVATE ${CMAKE_BINARY_DIR}/engine/lotus/winml_api)                   # windows machine learning component headers
target_include_directories(winml_test_api PRIVATE ${CMAKE_BINARY_DIR}/engine/lotus/winml/sdk/cppwinrt/include)  # sdk cppwinrt headers
target_include_directories(winml_test_api PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_include_directories(winml_test_api PRIVATE ${CMAKE_SOURCE_DIR}/winml/tests/api)
target_include_directories(winml_test_api PRIVATE ${CMAKE_SOURCE_DIR}/winml/tests/common)
target_include_directories(winml_test_api PRIVATE ${ML_LOTUS_DIR}/winml/lib/api.image/inc)

# Properties
set_target_properties(
    winml_test_api
    PROPERTIES
    OUTPUT_NAME winmlapitests)

set_target_properties(winml_test_api
    PROPERTIES
    LINK_FLAGS
    "/DELAYLOAD:dxgi.dll /DELAYLOAD:d3d11.dll")

set_target_properties(winml_test_api
    PROPERTIES
    FOLDER
    "WinML")

# Any project that links in debug_alloc.obj needs this lib.
# unresolved external symbol __imp_SymSetOptions
# ...                        __imp_SymGetLineFromAddr64
# ...                        __imp_SymInitialize
# ...                        __imp_SymFromAddr
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(DBGHELP dbghelp.lib)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

# Link libraries
target_link_libraries(winml_test_api PRIVATE DirectML)
target_link_libraries(winml_test_api PRIVATE libprotobuf)
target_link_libraries(winml_test_api PRIVATE onnx)
target_link_libraries(winml_test_api PRIVATE onnxruntime_common)
target_link_libraries(winml_test_api PRIVATE onnxruntime_graph)
target_link_libraries(winml_test_api PRIVATE onnxruntime_framework)
target_link_libraries(winml_test_api PRIVATE onnxruntime_mlas)
target_link_libraries(winml_test_api PRIVATE onnxruntime_providers)
target_link_libraries(winml_test_api PRIVATE onnxruntime_providers_dml)
target_link_libraries(winml_test_api PRIVATE onnxruntime_session)
target_link_libraries(winml_test_api PRIVATE onnx_proto)
target_link_libraries(winml_test_api PRIVATE wil)
target_link_libraries(winml_test_api PRIVATE windowsapp.lib)
target_link_libraries(winml_test_api PRIVATE winml_lib_image)
target_link_libraries(winml_test_api PRIVATE winml_dll)
target_link_libraries(winml_test_api PRIVATE winml_test_common)
target_link_libraries(winml_test_api PRIVATE ${DBGHELP})

install(
    FILES
        $<TARGET_FILE:winml_test_api>
        $<TARGET_PDB_FILE:winml_test_api>
    DESTINATION bin
)

install(DIRECTORY models/ DESTINATION bin)
