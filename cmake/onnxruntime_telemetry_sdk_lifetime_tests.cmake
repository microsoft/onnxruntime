# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Use the already-patched FetchContent source. Destructor observers also support
# the original SDK for negative controls, without ever accessing a retired object.
set(telemetry_lifetime_sdk_dir "${CMAKE_CURRENT_BINARY_DIR}/telemetry_lifetime_sdk")
file(MAKE_DIRECTORY "${telemetry_lifetime_sdk_dir}")
foreach(source pal/PAL.cpp api/AuthTokensController.cpp)
  set(source_path "${cpp_client_telemetry_SOURCE_DIR}/lib/${source}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_path}")
  file(READ "${source_path}" instrumented)
  if(source STREQUAL "pal/PAL.cpp")
    set(type SdkVersion)
    set(base std::string)
    set(constructor string)
    string(REPLACE "static const std::string sdkVersion("
      "static const OrtTelemetryTestSdkVersion sdkVersion(" instrumented "${instrumented}")
    if(instrumented MATCHES "const char\\* PlatformAbstractionLayer::getSdkVersion\\(\\) const")
      set(expected_type_count 0)
    else()
      set(expected_type_count 1)
    endif()
    set(access_site "PlatformAbstractionLayer::getSdkVersion() const\n    {")
  else()
    set(type AuthMutex)
    set(base std::mutex)
    set(constructor mutex)
    string(REPLACE "static std::mutex mutex;"
      "static OrtTelemetryTestAuthMutex mutex;" instrumented "${instrumented}")
    set(expected_type_count 1)
    set(access_site "AuthTokensController::GetMutex()\n    {")
  endif()
  string(REPLACE "using ${type} = ${base};" "using ${type} = OrtTelemetryTest${type};"
    instrumented "${instrumented}")
  string(REGEX MATCHALL "OrtTelemetryTest${type}" type_sites "${instrumented}")
  list(LENGTH type_sites type_count)
  if(NOT type_count EQUAL expected_type_count)
    message(FATAL_ERROR "Update the telemetry regression's SDK ${type} construction instrumentation")
  endif()
  string(REPLACE "${access_site}" "${access_site}\n        OrtTelemetry${type}Access();"
    instrumented "${instrumented}")
  string(REGEX MATCHALL "OrtTelemetry${type}Access\\(\\)" access_sites "${instrumented}")
  list(LENGTH access_sites access_count)
  if(NOT access_count EQUAL 1)
    message(FATAL_ERROR "Update the telemetry regression's SDK ${type} access instrumentation")
  endif()
  string(PREPEND instrumented
    "#include <string>\n#include <mutex>\n"
    "extern \"C\" void OrtTelemetry${type}Destroyed();\n"
    "extern \"C\" void OrtTelemetry${type}Access();\n"
    "struct OrtTelemetryTest${type} : ${base} {\n"
    "  using ${base}::${constructor};\n"
    "  ~OrtTelemetryTest${type}() { OrtTelemetry${type}Destroyed(); }\n"
    "};\n")
  get_filename_component(name "${source}" NAME)
  file(WRITE "${telemetry_lifetime_sdk_dir}/${name}" "${instrumented}")
endforeach()

add_library(onnxruntime_telemetry_lifetime_sdk_test OBJECT
  "${telemetry_lifetime_sdk_dir}/PAL.cpp" "${telemetry_lifetime_sdk_dir}/AuthTokensController.cpp")
target_include_directories(onnxruntime_telemetry_lifetime_sdk_test PRIVATE
  $<TARGET_PROPERTY:mat,INCLUDE_DIRECTORIES>
  "${cpp_client_telemetry_SOURCE_DIR}/lib"
  "${cpp_client_telemetry_SOURCE_DIR}/lib/pal"
  "${cpp_client_telemetry_SOURCE_DIR}/lib/api")
target_compile_definitions(onnxruntime_telemetry_lifetime_sdk_test PRIVATE
  $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
target_compile_options(onnxruntime_telemetry_lifetime_sdk_test PRIVATE $<TARGET_PROPERTY:mat,COMPILE_OPTIONS>)
target_link_libraries(onnxruntime_telemetry_lifetime_sdk_test PRIVATE mat)

onnxruntime_add_executable(onnxruntime_telemetry_sdk_lifetime_test
  "${TEST_SRC_DIR}/platform/telemetry_1ds/telemetry_sdk_lifetime_test.cc"
  $<TARGET_OBJECTS:onnxruntime_telemetry_lifetime_sdk_test>)
target_include_directories(onnxruntime_telemetry_sdk_lifetime_test PRIVATE
  $<TARGET_PROPERTY:mat,INCLUDE_DIRECTORIES> "${cpp_client_telemetry_SOURCE_DIR}/lib")
target_compile_definitions(onnxruntime_telemetry_sdk_lifetime_test PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
target_link_libraries(onnxruntime_telemetry_sdk_lifetime_test PRIVATE mat)
set_target_properties(onnxruntime_telemetry_lifetime_sdk_test onnxruntime_telemetry_sdk_lifetime_test
  PROPERTIES FOLDER "ONNXRuntimeTest")

# These accessors need no provider, worker dispatcher, cache, or network transport.
foreach(mode sdk-version sdk-version-threaded auth-mutex auth-mutex-threaded)
  add_test(NAME onnxruntime_telemetry_lifetime_${mode}_test
    COMMAND onnxruntime_telemetry_sdk_lifetime_test ${mode})
  set_tests_properties(onnxruntime_telemetry_lifetime_${mode}_test PROPERTIES TIMEOUT 10)
endforeach()
