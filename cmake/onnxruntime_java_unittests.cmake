# Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

# This is a helper script that enables us to run gradle tests via ctest.

FILE(TO_NATIVE_PATH ${GRADLE_EXECUTABLE} GRADLE_NATIVE_PATH)
FILE(TO_NATIVE_PATH ${BIN_DIR} BINDIR_NATIVE_PATH)

function(run_java_unit_test SYSTEM_PROPERTY_DEFINITION)
  set(GRADLE_TEST_ARGS
      ${GRADLE_NATIVE_PATH}
      test --rerun
      cmakeCheck
      --console=plain
      -DcmakeBuildDir=${BINDIR_NATIVE_PATH}
      -Dorg.gradle.daemon=false
      ${SYSTEM_PROPERTY_DEFINITIONS})

  if(WIN32)
  list(PREPEND GRADLE_TEST_ARGS cmd /C)
  endif()

  message(STATUS "gradle test command args: ${GRADLE_TEST_ARGS}")

  execute_process(COMMAND ${GRADLE_TEST_ARGS}
                  WORKING_DIRECTORY ${REPO_ROOT}/java
                  RESULT_VARIABLE HAD_ERROR)
endfunction()

message(STATUS "gradle additional system property definitions: ${GRADLE_SYSTEM_PROPERTY_DEFINITIONS}")

string(FIND "${GRADLE_SYSTEM_PROPERTY_DEFINITIONS}" "-DUSE_CUDA=1" INDEX_CUDA)
string(FIND "${GRADLE_SYSTEM_PROPERTY_DEFINITIONS}" "-DUSE_DML=1" INDEX_DML)


if((INDEX_CUDA GREATER -1) AND (INDEX_DML GREATER -1))
  string(REPLACE "-DUSE_CUDA=1" "" GRADLE_DML_YSTEM_PROPERTY_DEFINITIONS ${GRADLE_SYSTEM_PROPERTY_DEFINITIONS})
  run_java_unit_test(${GRADLE_DML_YSTEM_PROPERTY_DEFINITIONS})

  string(REPLACE "-DUSE_DML=1" "" GRADLE_CUDA_YSTEM_PROPERTY_DEFINITIONS ${GRADLE_SYSTEM_PROPERTY_DEFINITIONS})
  run_java_unit_test(${GRADLE_CUDA_YSTEM_PROPERTY_DEFINITIONS})
else()
  run_java_unit_test(${GRADLE_SYSTEM_PROPERTY_DEFINITIONS})
endif()

if(HAD_ERROR)
  message(FATAL_ERROR "Java Unitests failed")
endif()
