// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include "test/training/runner/training_runner.h"

void PrepareMNISTData(const std::string& data_folder,
                      onnxruntime::training::DataSet& training_data,
                      onnxruntime::training::DataSet& test_data);
