# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# training end-to-end tests

if (NOT IS_DIRECTORY ${onnxruntime_TRAINING_E2E_TEST_DATA_ROOT})
  message(FATAL_ERROR "Training E2E test data directory is not valid: ${onnxruntime_TRAINING_E2E_TEST_DATA_ROOT}")
endif()

find_package(Python3 3.5 REQUIRED COMPONENTS Interpreter)

# convergence test
add_test(
  NAME onnxruntime_training_bert_convergence_e2e_test
  COMMAND
    ${Python3_EXECUTABLE} ${REPO_ROOT}/orttraining/tools/ci_test/run_convergence_test.py
      --binary_dir $<TARGET_FILE_DIR:onnxruntime_training_bert>
      --training_data_root ${onnxruntime_TRAINING_E2E_TEST_DATA_ROOT}/data
      --model_root ${onnxruntime_TRAINING_E2E_TEST_DATA_ROOT}/models
  CONFIGURATIONS RelWithDebInfo)

set_property(
  TEST
    onnxruntime_training_bert_convergence_e2e_test
  PROPERTY
    LABELS training_e2e)
