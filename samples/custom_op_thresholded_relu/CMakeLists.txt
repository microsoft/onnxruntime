cmake_minimum_required(VERSION 3.18)
project(ThresholdedReluCustomOp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime
# You may need to adjust these paths based on your ONNX Runtime installation
set(ONNXRUNTIME_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Look for ONNX Runtime headers
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
        ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session
        ${ONNXRUNTIME_ROOT_PATH}/include
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
    PATH_SUFFIXES core/session
)

# Look for ONNX Runtime library
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT_PATH}/lib
        ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release
        ${ONNXRUNTIME_ROOT_PATH}/build/Windows/Release/Release
        ${ONNXRUNTIME_ROOT_PATH}/build/RelWithDebInfo
        /usr/local/lib
        /usr/lib
)

# Output findings
message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(WARNING "ONNX Runtime not found. You may need to:")
    message(WARNING "1. Build ONNX Runtime first")
    message(WARNING "2. Set ONNXRUNTIME_ROOT_PATH to the correct path")
    message(WARNING "3. Install ONNX Runtime system-wide")
endif()

# Custom operator library
add_library(thresholded_relu_op SHARED
    thresholded_relu_op.cpp
    thresholded_relu_op.h
)

target_include_directories(thresholded_relu_op PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_ROOT_PATH}/include
    ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
    ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session
)

target_link_libraries(thresholded_relu_op
    ${ONNXRUNTIME_LIBRARY}
)

# Set proper export/import macros for Windows
if(WIN32)
    target_compile_definitions(thresholded_relu_op PRIVATE WIN32_LEAN_AND_MEAN)
    # Ensure proper export of the RegisterCustomOps function
    set_target_properties(thresholded_relu_op PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
    )
endif()

# Test application
add_executable(test_thresholded_relu
    test_thresholded_relu.cpp
)

target_include_directories(test_thresholded_relu PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_ROOT_PATH}/include
    ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
    ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session
)

target_link_libraries(test_thresholded_relu
    ${ONNXRUNTIME_LIBRARY}
)

# Python model creation script
find_program(PYTHON_EXECUTABLE python3 python)

if(PYTHON_EXECUTABLE)
    add_custom_target(create_model
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/create_model.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Creating ONNX model"
        SOURCES create_model.py
    )
    
    # Make test depend on model creation
    add_dependencies(test_thresholded_relu create_model)
else()
    message(WARNING "Python not found. You'll need to run create_model.py manually.")
endif()

# Installation
install(TARGETS thresholded_relu_op
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS test_thresholded_relu
    RUNTIME DESTINATION bin
)

install(FILES 
    README.md
    create_model.py
    thresholded_relu_op.h
    DESTINATION share/onnxruntime/examples/custom_op_thresholded_relu
)

# Print build instructions
message(STATUS "")
message(STATUS "=== Build Instructions ===")
message(STATUS "1. Ensure ONNX Runtime is built or installed")
message(STATUS "2. Create build directory: mkdir build && cd build")
message(STATUS "3. Configure: cmake ..")
message(STATUS "4. Build: cmake --build .")
message(STATUS "5. Run: ./test_thresholded_relu")
message(STATUS "")
message(STATUS "If ONNX Runtime is not found, you may need to:")
message(STATUS "- Set ONNXRUNTIME_ROOT_PATH to the ONNX Runtime installation directory")
message(STATUS "- Or build ONNX Runtime from source first")
message(STATUS "")