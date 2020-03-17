# Minimum CMake required
cmake_minimum_required(VERSION 3.10)

include(${ML_BUILD_TOOLS_DIR}/cppwinrt_command.cmake)

winml_add_library(winml_test_image SHARED 
    imageTestHelper.h
    imageTestHelper.cpp
    imagetests.cpp
    imagetests.h
    precomp.h)

# Add packages
target_add_packages(winml_test_image MLRuntime.Gsl)
target_add_packages(winml_test_image Taef.Native)
target_add_packages(winml_test_image Mock10)

target_universal_test(winml_test_image)

# Compiler options
add_definitions(/bigobj)

# Specify the usage of a precompiled header
target_precompiled_header(winml_test_image precomp.h)

# Includes
target_include_directories(winml_test_image PRIVATE ${CMAKE_BINARY_DIR}/engine/lotus)                             # windows machine learning component headers
target_include_directories(winml_test_image PRIVATE ${CMAKE_BINARY_DIR}/engine/lotus/winml_api)                   # windows machine learning component headers
target_include_directories(winml_test_image PRIVATE ${CMAKE_BINARY_DIR}/engine/lotus/winml_api/comp_generated)    # windows machine learning component headers
target_include_directories(winml_test_image PRIVATE ${CMAKE_BINARY_DIR}/engine/lotus/winml/sdk/cppwinrt/include)  # sdk cppwinrt headers
target_include_directories(winml_test_image PRIVATE ${CMAKE_SOURCE_DIR}/common/inc/copied_from_windows_git)
target_include_directories(winml_test_image PRIVATE ${ML_LOTUS_DIR}/winml/test/common)
target_include_directories(winml_test_image PRIVATE ${CMAKE_SOURCE_DIR}/winml/tests/imageTest)

# Properties
set_target_properties(winml_test_image
    PROPERTIES
    OUTPUT_NAME winmlimagetests
)

set_target_properties(winml_test_image
    PROPERTIES
    FOLDER "WinML")

# Any project that links in debug_alloc.obj needs this lib.
# unresolved external symbol __imp_SymSetOptions
# ...                        __imp_SymGetLineFromAddr64
# ...                        __imp_SymInitialize
# ...                        __imp_SymFromAddr
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(DBGHELP dbghelp.lib)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

# Link libraries
target_link_libraries(winml_test_image PRIVATE wil)
target_link_libraries(winml_test_image PRIVATE windowsapp.lib)
target_link_libraries(winml_test_image PRIVATE winml_test_common)
target_link_libraries(winml_test_image PRIVATE winml_dll)
target_link_libraries(winml_test_image PRIVATE ${DBGHELP})

install(
    FILES 
        $<TARGET_FILE:winml_test_image>
        $<TARGET_PDB_FILE:winml_test_image>
        ImageTestsData.xml
        BatchTestData.xml
        MnistTestsData.xml
    DESTINATION bin
)

install(DIRECTORY images/ models/ groundTruth batchGroundTruth DESTINATION bin)