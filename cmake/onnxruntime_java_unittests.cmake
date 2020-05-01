# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

FILE(TO_NATIVE_PATH ${GRADLE_EXECUTABLE} GRADLE_NATIVE_PATH)
# FILE(TO_NATIVE_PATH ${BIN_DIR} BINDIR_NATIVE_PATH)

execute_process(COMMAND cmd /c ${GRADLE_NATIVE_PATH}.bat cmakeCheck -DcmakeBuildDir=${BINDIR}
    WORKING_DIRECTORY ${REPO_ROOT}/java
    RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
    message(FATAL_ERROR "Java Unitests failed")
endif()
