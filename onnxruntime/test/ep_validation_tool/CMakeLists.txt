# Global settings
cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)

# CMake project
project(ps_onnxruntime_test C CXX)

# Variables
string(TOLOWER ${CMAKE_GENERATOR_PLATFORM} PLATFORM)

include(scripts/cmake/download_nuget.cmake)

include(scripts/cmake/parse_encryption_key_packages.cmake)
set(ENC_KEY_DIR ${CMAKE_CURRENT_BINARY_DIR}/ps_enc_keys)
parse_encryption_key_packages(${CMAKE_CURRENT_LIST_DIR}/EncryptionKeyPackages.json ${ENC_KEY_DIR} SYMBOL_ENCRYPTION_KEY_MANAGER_INCLUDES SYMBOL_KEY_ID_TO_KEY_MAPPING)
# include the encryption key directory
include_directories(${ENC_KEY_DIR})

# Embedding symbols defined automatically in source files
configure_file(${CMAKE_CURRENT_LIST_DIR}/src/utils/encryption_key_manager.cpp.in ${CMAKE_BINARY_DIR}/encryption_key_manager.cpp)

string(REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
# Only apply /O2 for Release builds to avoid conflict with /RTC1 in Debug
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/utils/crypto.c PROPERTIES COMPILE_FLAGS "$<$<CONFIG:Release>:/O2>")
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/utils/crypto.c PROPERTIES LANGUAGE "C")

# ps_onnxruntime_test project
add_executable(ps_onnxruntime_test)

# Source files
set(PS_ONNXRUNTIME_TEST_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
list(APPEND PS_ONNXRUNTIME_TEST_SRC_PATTERNS
    "${PS_ONNXRUNTIME_TEST_SRC_DIR}/*.cpp"
	"${PS_ONNXRUNTIME_TEST_SRC_DIR}/../*.cpp"
    "${PS_ONNXRUNTIME_TEST_SRC_DIR}/*.h"
    "${PS_ONNXRUNTIME_TEST_SRC_DIR}/utils/*.cpp"
    "${PS_ONNXRUNTIME_TEST_SRC_DIR}/utils/*.h")
file(GLOB PS_ONNXRUNTIME_TEST_SRC CONFIGURE_DEPENDS ${PS_ONNXRUNTIME_TEST_SRC_PATTERNS})
target_sources(ps_onnxruntime_test PRIVATE ${PS_ONNXRUNTIME_TEST_SRC} ${CMAKE_BINARY_DIR}/encryption_key_manager.cpp ${CMAKE_CURRENT_LIST_DIR}/src/utils/crypto.c)

# Pass execution provider to compiler as #define EP={QNN|OV|VAI}.
if(DEFINED EP)
    if("${EP}" STREQUAL "qnn")
        target_compile_definitions(ps_onnxruntime_test PRIVATE USE_QNN)
    elseif("${EP}" STREQUAL "ov")
        target_compile_definitions(ps_onnxruntime_test PRIVATE USE_OV)
    elseif("${EP}" STREQUAL "vitisai")
        target_compile_definitions(ps_onnxruntime_test PRIVATE USE_VAI)
    else()
        message(FATAL_ERROR "Execution provider '${EP}' is not supported.")
    endif()
    message(STATUS "Building for EP=${EP}.")
else()
    set(EP "cpu")
    message(STATUS "Building for CPU.")
endif()

# Download ps-onnxruntime nuget.
if (NOT DEFINED ONNXRUNTIME_NUGET_DIR)
    # Set ps-onnxruntime package and version.
    set(NUGET_PREFIX "ps-onnxruntime.")
    set(NUGET_SUFFIX "-windows-v143-${PLATFORM}-Release")
    string(CONCAT ONNXRUNTIME_NUGET_NAME ${NUGET_PREFIX} ${EP} ${NUGET_SUFFIX})
    message(STATUS "ONNXRUNTIME_NUGET_DIR not defined, downloading ${ONNXRUNTIME_NUGET_NAME}.")

    if(NOT DEFINED ONNXRUNTIME_NUGET_VERSION)
        set(ONNXRUNTIME_NUGET_VERSION "2.0.5-main")
        message(STATUS "ONNXRUNTIME_NUGET_VERSION not defined, using ${ONNXRUNTIME_NUGET_VERSION}.")
    endif()

    nugetDL(${ONNXRUNTIME_NUGET_NAME} ${ONNXRUNTIME_NUGET_VERSION} "${CMAKE_CURRENT_LIST_DIR}/Nuget.config" "${CMAKE_CURRENT_LIST_DIR}/packages")
    set(ONNXRUNTIME_NUGET_DIR ${NUGET_DIR})
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
target_include_directories(ps_onnxruntime_test PRIVATE
    ${PS_ONNXRUNTIME_TEST_SRC_DIR}
    ${PS_ONNXRUNTIME_TEST_SRC_DIR}/utils
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

# Use release versions of VCRT in all flavors, because the debug versions are non-redistributable.
target_compile_options(ps_onnxruntime_test PRIVATE /MD)
target_compile_options(npy PRIVATE /MD)

# Linked libs
target_link_directories(ps_onnxruntime_test PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(ps_onnxruntime_test PRIVATE ps-onnxruntime npy)

# Compilation feature flags - USE_WINML_FEATURES enables WinML, otherwise PSORT is used
option(USE_WINML_FEATURES "Enable WinML-specific compilation (default is PSORT)" OFF)

if(USE_WINML_FEATURES)
    message(STATUS "Compiling with USE_WINML_FEATURES enabled (WinML build)")
    target_compile_definitions(ps_onnxruntime_test PRIVATE USE_WINML_FEATURES)
else()
    message(STATUS "Compiling with PSORT features (WinML features disabled)")
endif()

# Binplace binaries
install(TARGETS ps_onnxruntime_test DESTINATION bin)
install(DIRECTORY "${ONNXRUNTIME_RUNTIMES_DIR}/" DESTINATION bin)
