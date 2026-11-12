cmake_minimum_required(VERSION 3.26)
project(ORTCudaPluginSmoke LANGUAGES CXX)

if(NOT ORT_HOME)
  message(FATAL_ERROR "ORT_HOME must point to the source-built ONNX Runtime package")
endif()

add_executable(ort-runtime-smoke ort_runtime_smoke.cpp)
target_compile_features(ort-runtime-smoke PRIVATE cxx_std_17)
target_include_directories(ort-runtime-smoke PRIVATE "${ORT_HOME}/include")
target_link_libraries(ort-runtime-smoke PRIVATE "${ORT_HOME}/lib/onnxruntime.lib")

foreach(runtime IN ITEMS onnxruntime.dll onnxruntime_providers_cuda.dll)
  add_custom_command(TARGET ort-runtime-smoke POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${ORT_HOME}/lib/${runtime}" "$<TARGET_FILE_DIR:ort-runtime-smoke>"
    VERBATIM)
endforeach()