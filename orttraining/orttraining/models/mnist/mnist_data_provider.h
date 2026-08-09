// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include "orttraining/models/runner/training_runner.h"

/**
Load the dataset from a directory.
@shard_index the shard to be loaded, used when loading partial data for data-parallelism training
@total_shard total number of shards, used when loading partial data for data-parallelism training
*/
void PrepareMNISTData(const std::string& data_folder,
                      const std::vector<int64_t>& image_dims,
                      const std::vector<int64_t>& label_dims,
                      onnxruntime::training::DataSet& training_data,
                      onnxruntime::training::DataSet& test_data,
                      size_t shard_index = 0,
                      size_t total_shard = 1);
