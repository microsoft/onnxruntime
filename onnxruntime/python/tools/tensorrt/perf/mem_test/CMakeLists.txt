project(alprdaemon)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required (VERSION 2.6)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories( 
	/code/onnxruntime/include/onnxruntime/core/session/
	/code/onnxruntime/include/onnxruntime/core/providers/tensorrt/
)

# Note for this to work with CUDA run the binary with this env var set: ASAN_OPTIONS=protect_shadow_gap=0
set(CMAKE_CXX_FLAGS "-fsanitize=address  -fsanitize=leak -g ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-fsanitize=address  -fsanitize=leak -g ${CMAKE_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-fsanitize=address  -fsanitize=leak ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "-fsanitize=address  -fsanitize=leak ${CMAKE_MODULE_LINKER_FLAGS}")
        
ADD_EXECUTABLE(onnx_memtest  main.cpp)
target_link_libraries(onnx_memtest onnxruntime)
