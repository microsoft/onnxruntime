cmake_minimum_required(VERSION 3.17)

project(test LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Debug)

include_directories( 
  /home/azureuser/repos/ort/include/onnxruntime/core/session
)
        
ADD_EXECUTABLE(ft_custom_op_test ft_custom_op_test.cc)
target_link_libraries(
  ft_custom_op_test 
  /mnt/test_build_cuda/Release/libonnxruntime.so 
)
