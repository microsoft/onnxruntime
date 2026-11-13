# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

function(onnxruntime_add_provider_test_shards target)
  set(test_names)
  foreach(shard_index RANGE 0 3)
    set(test_name "onnxruntime_provider_test_shard_${shard_index}")
    add_test(NAME ${test_name}
      COMMAND "${Python_EXECUTABLE}" "${REPO_ROOT}/tools/ci_build/run_provider_test_shard.py"
        --binary "$<TARGET_FILE:${target}>"
        --shard-index ${shard_index}
        --report "$<TARGET_FILE:${target}>.$<CONFIG>.shard_${shard_index}.results.xml"
        -- ${ARGN}
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:${target}>"
    )
    # Inter-op needs at least two threads to retain parallel-executor coverage.
    set_tests_properties(${test_name} PROPERTIES
      TIMEOUT 10800
      PROCESSORS 4
      ENVIRONMENT_MODIFICATION
        "ORT_INTRA_OP_NUM_THREADS=set:3;ORT_INTER_OP_NUM_THREADS=set:2;OMP_NUM_THREADS=set:3"
    )
    list(APPEND test_names ${test_name})
  endforeach()
  set(onnxruntime_provider_test_names ${test_names} PARENT_SCOPE)
endfunction()
