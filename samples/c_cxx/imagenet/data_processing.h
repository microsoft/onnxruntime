// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <sal.h>

class DataProcessing {
 public:
  virtual void operator()(_In_ const void* input_data, _Out_writes_bytes_all_(output_len) void* output_data, size_t output_len) const = 0;
  virtual std::vector<int64_t> GetOutputShape(size_t batch_size) const = 0;
  virtual ~DataProcessing() = default;
};