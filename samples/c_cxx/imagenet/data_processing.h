// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>

class DataProcessing {
 public:
  virtual void operator()(const void* input_data, void* output_data) const = 0;
  virtual std::vector<int64_t> GetOutputShape(size_t batch_size) const = 0;
  virtual ~DataProcessing() = default;
};