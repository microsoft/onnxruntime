// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include "test/training/runner/training_runner.h"

void PrepareSqueezenetData(const std::string& data_folder,
                      onnxruntime::training::TrainingRunner::TrainingData& training_data,
                      onnxruntime::training::TrainingRunner::TestData& test_data);
