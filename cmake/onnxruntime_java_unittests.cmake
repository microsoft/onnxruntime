# Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

# This is a helper script that enables us to run gradle tests via ctest.

FILE(TO_NATIVE_PATH ${GRADLE_EXECUTABLE} GRADLE_NATIVE_PATH)
FILE(TO_NATIVE_PATH ${BIN_DIR} BINDIR_NATIVE_PATH)

message(STATUS "gradle additional system property definitions: ${GRADLE_SYSTEM_PROPERTY_DEFINITIONS}")

set(GRADLE_TEST_ARGS
    ${GRADLE_NATIVE_PATH}
    test --rerun
    cmakeCheck
    --console=plain
    -DcmakeBuildDir=${BINDIR_NATIVE_PATH}
    -Dorg.gradle.daemon=false
    ${GRADLE_SYSTEM_PROPERTY_DEFINITIONS})

if(WIN32)
  list(PREPEND GRADLE_TEST_ARGS cmd /C)
endif()

message(STATUS "gradle test command args: ${GRADLE_TEST_ARGS}")

execute_process(COMMAND ${GRADLE_TEST_ARGS}
    WORKING_DIRECTORY ${REPO_ROOT}/java
    RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
  message(FATAL_ERROR "Java Unitests failed")
endif()
