# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug
# cmake --build ./
# NOTE: For Windows, copy onnxruntime.dll and onnxruntime.pdb into the same folder of TestOutTreeEp.exe, otherwise, during runtime,
# it will search the default system path (C:\Windows\System32) for onnxruntime.dll
cmake_minimum_required(VERSION 3.26)
project(TestOutTreeEp)
add_executable(TestOutTreeEp test.cpp)

target_include_directories(TestOutTreeEp PUBLIC "../../include/onnxruntime")
if (WIN32)
#find_library(ORT_LIB NAMES onnxruntime PATHS "C:/Users/leca/source/onnxruntime/build/Windows/Debug/Debug" NO_DEFAULT_PATH)
#target_link_libraries(TestOutTreeEp PUBLIC ${ORT_LIB})

target_link_libraries(TestOutTreeEp PUBLIC "C:/Users/leca/source/onnxruntime/build/Windows/Debug/Debug/onnxruntime.lib")

#add_library(onnxruntime SHARED IMPORTED)
#set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "C:/Users/leca/source/onnxruntime/build/Windows/Debug/Debug/onnxruntime.dll")
#target_link_libraries(TestOutTreeEp PUBLIC onnxruntime)

#link_directories("C:/Users/leca/source/onnxruntime/build/Windows/Debug/Debug")
#target_link_libraries(TestOutTreeEp PUBLIC onnxruntime.lib)

elseif (LINUX)
target_link_libraries(TestOutTreeEp PUBLIC "/home/leca/code/onnxruntime/build/tensorrt/Debug/libonnxruntime.so")
endif()
