# Global settings
cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)

# CMake project
project(onnxruntime_test C CXX)

# Variables
string(TOLOWER ${CMAKE_GENERATOR_PLATFORM} PLATFORM)

include(scripts/cmake/download_nuget.cmake)


string(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
# Only apply /O2 for Release builds to avoid conflict with /RTC1 in Debug
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/utils/crypto.c PROPERTIES COMPILE_FLAGS "$<$<CONFIG:Release>:/O2>")
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/utils/crypto.c PROPERTIES LANGUAGE "C")

# ps_onnxruntime_test project
add_executable(onnxruntime_test)

# Source files
set(ONNXRUNTIME_TEST_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
list(APPEND ONNXRUNTIME_TEST_SRC_PATTERNS
    "${ONNXRUNTIME_TEST_SRC_DIR}/*.cpp"
	"${ONNXRUNTIME_TEST_SRC_DIR}/../*.cpp"
    "${ONNXRUNTIME_TEST_SRC_DIR}/*.h"
    "${ONNXRUNTIME_TEST_SRC_DIR}/utils/*.cpp"
    "${ONNXRUNTIME_TEST_SRC_DIR}/utils/*.h")
file(GLOB ONNXRUNTIME_TEST_SRC CONFIGURE_DEPENDS ${ONNXRUNTIME_TEST_SRC_PATTERNS})

# Pass execution provider to compiler as #define EP={QNN|OV|VAI}.
if(DEFINED EP)
    if("${EP}" STREQUAL "qnn")
        target_compile_definitions(onnxruntime_test PRIVATE USE_QNN)
    elseif("${EP}" STREQUAL "ov")
        target_compile_definitions(onnxruntime_test PRIVATE USE_OV)
    elseif("${EP}" STREQUAL "vitisai")
        target_compile_definitions(onnxruntime_test PRIVATE USE_VAI)
    else()
        message(FATAL_ERROR "Execution provider '${EP}' is not supported.")
    endif()
    message(STATUS "Building for EP=${EP}.")
else()
    set(EP "cpu")
    message(STATUS "Building for CPU.")
endif()

# Check onnxruntime dirs
if (NOT DEFINED ONNXRUNTIME_NUGET_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_NUGET_DIR not specified.")
endif()
set (ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_NUGET_DIR}/build/native/include")
set (ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_NUGET_DIR}/build/native/lib")
set (ONNXRUNTIME_RUNTIMES_DIR "${ONNXRUNTIME_NUGET_DIR}/runtimes/win-${PLATFORM}/native")

# Libnpy dirs
set(NUMPY_DIR "${CMAKE_CURRENT_LIST_DIR}/../libnpy")
set(NUMPY_INCLUDE_DIR "${NUMPY_DIR}/include")
add_subdirectory("${NUMPY_DIR}/src" "${NUMPY_DIR}/build")
target_include_directories(npy PRIVATE ${NUMPY_INCLUDE_DIR})

# nlohmann/json dir
set(JSON_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../json/single_include")

# Include dirs
target_include_directories(onnxruntime_test PRIVATE
    ${ONNXRUNTIME_TEST_SRC_DIR}
    ${ONNXRUNTIME_TEST_SRC_DIR}/utils
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

# Use release versions of VCRT in all flavors, because the debug versions are non-redistributable.
target_compile_options(onnxruntime_test PRIVATE /MD)
target_compile_options(npy PRIVATE /MD)

# Linked libs
target_link_directories(onnxruntime_test PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(onnxruntime_test PRIVATE onnxruntime npy)

# Compilation feature flags - USE_WINML_FEATURES enables WinML, otherwise PSORT is used
option(USE_WINML_FEATURES "Enable WinML-specific compilation (default is PSORT)" OFF)

if(USE_WINML_FEATURES)
    message(STATUS "Compiling with USE_WINML_FEATURES enabled (WinML build)")
    target_compile_definitions(onnxruntime_test PRIVATE USE_WINML_FEATURES)
else()
    message(STATUS "Compiling with PSORT features (WinML features disabled)")
endif()

# Binplace binaries
install(TARGETS onnxruntime_test DESTINATION bin)
install(DIRECTORY "${ONNXRUNTIME_RUNTIMES_DIR}/" DESTINATION bin)
