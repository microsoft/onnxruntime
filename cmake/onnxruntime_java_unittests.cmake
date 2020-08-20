# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

# This is a windows only file so we can run gradle tests via ctest
FILE(TO_NATIVE_PATH ${GRADLE_EXECUTABLE} GRADLE_NATIVE_PATH)
FILE(TO_NATIVE_PATH ${BIN_DIR} BINDIR_NATIVE_PATH)

if (onnxruntime_USE_CUDA)
  execute_process(COMMAND cmd /C ${GRADLE_NATIVE_PATH} cmakeCheck -DcmakeBuildDir=${BINDIR_NATIVE_PATH} -Dorg.gradle.daemon=false -DUSE_CUDA=1
    WORKING_DIRECTORY ${REPO_ROOT}/java
    RESULT_VARIABLE HAD_ERROR)
else()
  execute_process(COMMAND cmd /C ${GRADLE_NATIVE_PATH} cmakeCheck -DcmakeBuildDir=${BINDIR_NATIVE_PATH} -Dorg.gradle.daemon=false
    WORKING_DIRECTORY ${REPO_ROOT}/java
    RESULT_VARIABLE HAD_ERROR)
endif()

if(HAD_ERROR)
    message(FATAL_ERROR "Java Unitests failed")
endif()
