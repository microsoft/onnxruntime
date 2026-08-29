cmake_minimum_required(VERSION 3.13)
project(OnnxruntimeReactNativePolicyTests LANGUAGES CXX)

enable_testing()

add_executable(
  onnxruntime_react_native_policy_test
  EpContextDataReadPolicyTest.cpp
  ../EpContextDataReadPolicy.cpp
  ../LoadModelArgumentPolicy.cpp)

target_include_directories(onnxruntime_react_native_policy_test PRIVATE ..)
target_compile_features(onnxruntime_react_native_policy_test PRIVATE cxx_std_17)

add_test(
  NAME onnxruntime_react_native_policy_test
  COMMAND onnxruntime_react_native_policy_test)
