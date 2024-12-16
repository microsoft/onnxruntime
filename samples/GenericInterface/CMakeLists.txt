# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug
# cmake --build ./
# NOTE: For Windows, copy onnxruntime.dll and onnxruntime.pdb into the same folder of TestOutTreeEp.exe, otherwise, during runtime,
# it will search the default system path (C:\Windows\System32) for onnxruntime.dll
cmake_minimum_required(VERSION 3.26)
project(GenericOrtEpInterface)
add_executable(GenericOrtEpInterface test.cpp)

target_include_directories(GenericOrtEpInterface PUBLIC "../../include/onnxruntime")
target_link_libraries(GenericOrtEpInterface PUBLIC "C:/Users/leca/source/onnxruntime3/samples/GenericInterface/build/Debug/onnxruntime.lib")

