# Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

# This is a windows only file so we can run gradle tests via ctest
FILE(TO_NATIVE_PATH ${GRADLE_EXECUTABLE} GRADLE_NATIVE_PATH)
FILE(TO_NATIVE_PATH ${BIN_DIR} BINDIR_NATIVE_PATH)

message(STATUS "GRADLE_TEST_EP_FLAGS: ${ORT_PROVIDER_FLAGS}")
if (onnxruntime_ENABLE_TRAINING_APIS)
    message(STATUS "Running ORT Java training tests")
    execute_process(COMMAND cmd /C ${GRADLE_NATIVE_PATH} --console=plain cmakeCheck -DcmakeBuildDir=${BINDIR_NATIVE_PATH} -Dorg.gradle.daemon=false ${ORT_PROVIDER_FLAGS} -DENABLE_TRAINING_APIS=1
      WORKING_DIRECTORY ${REPO_ROOT}/java
      RESULT_VARIABLE HAD_ERROR)
else()
    execute_process(COMMAND cmd /C ${GRADLE_NATIVE_PATH} --console=plain cmakeCheck -DcmakeBuildDir=${BINDIR_NATIVE_PATH} -Dorg.gradle.daemon=false ${ORT_PROVIDER_FLAGS}
      WORKING_DIRECTORY ${REPO_ROOT}/java
      RESULT_VARIABLE HAD_ERROR)
endif()


if(HAD_ERROR)
    message(FATAL_ERROR "Java Unitests failed")
endif()
