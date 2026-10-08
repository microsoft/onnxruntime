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

option(ORT_RN_BUILD_ENCRYPTION_PLUGIN "Build the test-only encrypted-context plugin" OFF)
if(ORT_RN_BUILD_ENCRYPTION_PLUGIN)
  get_filename_component(ORT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
  include(FetchContent)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  file(STRINGS "${ORT_REPO_ROOT}/cmake/deps.txt" gsl_dependency REGEX "^microsoft_gsl;")
  string(REPLACE "\\;" ";" gsl_dependency "${gsl_dependency}")
  list(GET gsl_dependency 1 gsl_url)
  list(GET gsl_dependency 2 gsl_sha1)
  set(GSL_TEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(gsl URL "${gsl_url}" URL_HASH "SHA1=${gsl_sha1}")
  FetchContent_MakeAvailable(gsl)

  set(plugin_sources "${ORT_REPO_ROOT}/onnxruntime/test/autoep/library/example_plugin_ep")
  file(GLOB plugin_source_files CONFIGURE_DEPENDS "${plugin_sources}/*.cc")
  add_library(example_plugin_ep SHARED ${plugin_source_files})
  target_include_directories(example_plugin_ep PRIVATE
    "${ORT_REPO_ROOT}/include/onnxruntime"
    "${ORT_REPO_ROOT}/include/onnxruntime/core/session")
  target_compile_features(example_plugin_ep PRIVATE cxx_std_17)
  target_link_libraries(example_plugin_ep PRIVATE Microsoft.GSL::GSL)
  # The plugin receives the runtime APIs in CreateEpFactories; it must not load a second ORT runtime.
  set_target_properties(example_plugin_ep PROPERTIES PREFIX "")
  if(APPLE)
    set_target_properties(example_plugin_ep PROPERTIES SUFFIX ".dylib")
  endif()
  if(WIN32)
    target_sources(example_plugin_ep PRIVATE "${plugin_sources}/example_plugin_ep_library.def")
    set_target_properties(example_plugin_ep PROPERTIES SUFFIX ".dll")
  endif()
endif()
