# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT onnxruntime_USE_1DS_TELEMETRY OR CMAKE_CROSSCOMPILING OR ANDROID OR IOS)
  return()
endif()

# Recompile only the SDK's debug source and worker with test-only observers. Explicit
# objects resolve these symbols before libmat's originals; production targets are unchanged.
# The mutex observer makes libc++'s nontrivial mutex lifetime visible on GCC/MSVC too.
set(telemetry_test_dir "${TEST_SRC_DIR}/platform/telemetry_1ds")
set(telemetry_sdk_test_dir "${CMAKE_CURRENT_BINARY_DIR}/telemetry_shutdown_sdk")
file(MAKE_DIRECTORY "${telemetry_sdk_test_dir}")
foreach(sdk_source callbacks/DebugSource.cpp pal/WorkerThread.cpp)
  set(sdk_source_path "${cpp_client_telemetry_SOURCE_DIR}/lib/${sdk_source}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${sdk_source_path}")
  file(READ "${sdk_source_path}" sdk_test_source)
  if(sdk_source STREQUAL "callbacks/DebugSource.cpp")
    string(REGEX MATCHALL "DE_LOCKGUARD\\(stateLock\\(\\)\\)" lock_sites "${sdk_test_source}")
    list(LENGTH lock_sites lock_site_count)
    if(NOT lock_site_count EQUAL 5)
      message(FATAL_ERROR "Update the telemetry regression's SDK debug-lock instrumentation")
    endif()
    string(REPLACE "DE_LOCKGUARD(stateLock());"
      "OrtTelemetryBeforeDebugLock();\n        (void)stateLock();\n        OrtTelemetryDebugLockInitialized();\n        DE_LOCKGUARD(stateLock());"
      sdk_test_source "${sdk_test_source}")
    string(PREPEND sdk_test_source
      "extern \"C\" void OrtTelemetryBeforeDebugLock();\nextern \"C\" void OrtTelemetryDebugLockInitialized();\n")
  else()
    string(REGEX MATCHALL "threadToJoin\\.join\\(\\)" join_sites "${sdk_test_source}")
    list(LENGTH join_sites join_site_count)
    if(NOT join_site_count EQUAL 2)
      message(FATAL_ERROR "Update the telemetry regression's SDK worker-join instrumentation")
    endif()
    string(REPLACE "threadToJoin.join();"
      "threadToJoin.join();\n                    OrtTelemetryWorkerJoined(this);"
      sdk_test_source "${sdk_test_source}")
    string(PREPEND sdk_test_source "extern \"C\" void OrtTelemetryWorkerJoined(void*);\n")
  endif()
  get_filename_component(sdk_test_name "${sdk_source}" NAME)
  file(WRITE "${telemetry_sdk_test_dir}/${sdk_test_name}" "${sdk_test_source}")
endforeach()

add_library(onnxruntime_telemetry_sdk_test OBJECT
  "${telemetry_sdk_test_dir}/DebugSource.cpp"
  "${telemetry_sdk_test_dir}/WorkerThread.cpp")
target_include_directories(onnxruntime_telemetry_sdk_test PRIVATE $<TARGET_PROPERTY:mat,INCLUDE_DIRECTORIES>)
target_compile_definitions(onnxruntime_telemetry_sdk_test PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
target_compile_options(onnxruntime_telemetry_sdk_test PRIVATE $<TARGET_PROPERTY:mat,COMPILE_OPTIONS>)
target_link_libraries(onnxruntime_telemetry_sdk_test PRIVATE $<TARGET_PROPERTY:mat,LINK_LIBRARIES>)

onnxruntime_add_executable(onnxruntime_telemetry_shutdown_test
  "${telemetry_test_dir}/telemetry_shutdown_test.cc"
  $<TARGET_OBJECTS:onnxruntime_telemetry_sdk_test>)
target_include_directories(onnxruntime_telemetry_shutdown_test PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR}
  "${cpp_client_telemetry_SOURCE_DIR}/lib" "${cpp_client_telemetry_SOURCE_DIR}/lib/include"
  "${cpp_client_telemetry_SOURCE_DIR}/lib/include/mat")
target_compile_definitions(onnxruntime_telemetry_shutdown_test PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
target_link_libraries(onnxruntime_telemetry_shutdown_test PRIVATE onnxruntime_common mat ${CMAKE_DL_LIBS} Threads::Threads)
if(CPUINFO_SUPPORTED)
  target_link_libraries(onnxruntime_telemetry_shutdown_test PRIVATE cpuinfo)
endif()
set_target_properties(onnxruntime_telemetry_sdk_test onnxruntime_telemetry_shutdown_test
  PROPERTIES FOLDER "ONNXRuntimeTest")

# Ordinary test mains suppress telemetry. Each process gets an empty private cache,
# clears CI suppression, and dispatches only an internal diagnostic, never an upload event.
foreach(mode enabled threaded disabled disabled-threaded)
  set(test_home "${CMAKE_CURRENT_BINARY_DIR}/telemetry_shutdown_test_${mode}")
  file(MAKE_DIRECTORY "${test_home}")
  set(disable_telemetry 0)
  if(mode MATCHES "^disabled")
    set(disable_telemetry 1)
  endif()
  add_test(NAME onnxruntime_telemetry_shutdown_${mode}_test
    COMMAND ${CMAKE_COMMAND} -E env
      "HOME=${test_home}" "XDG_CACHE_HOME=${test_home}" "LOCALAPPDATA=${test_home}" "APPDATA=${test_home}"
      CI=0 TF_BUILD=0 GITHUB_ACTIONS=0 GITLAB_CI=0 CIRCLECI=0 TRAVIS=0
      JENKINS_URL=0 CODEBUILD_BUILD_ID=0 BUILDKITE=0 TEAMCITY_VERSION=0
      APPVEYOR=0 BITBUCKET_BUILD_NUMBER=0 SYSTEM_TEAMFOUNDATIONCOLLECTIONURI=0
      ORT_RUNNING_UNIT_TESTS=0 "ORT_DISABLE_TELEMETRY=${disable_telemetry}"
      $<TARGET_FILE:onnxruntime_telemetry_shutdown_test> ${mode})
  set_tests_properties(onnxruntime_telemetry_shutdown_${mode}_test PROPERTIES TIMEOUT 30)
endforeach()
