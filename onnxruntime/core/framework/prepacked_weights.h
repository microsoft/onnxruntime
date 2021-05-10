// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/buffer_deleter.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

struct PrePackedWeights final {
  // Some weights may be associated with multiple pre-packed buffers.
  // Hence we hold them in containers. It is upto the developer implementing each PrePack()
  // method to define what gets stored in which position of the containers.

  // NOTE: Not all fields may be filled in and not all containers will have the same number of elements
  // It is upto the developer of the kernel to decide which fields to cache for re-use.

  std::vector<std::unique_ptr<void, BufferDeleter>> buffers_;  // cache pre-packed buffers associated with the kernel
  std::vector<size_t> buffer_sizes_;                           // cache sizes of pre-packed buffers (in bytes)

  // NOTE: `weights_sizes_` hold the number of elements in the weight tensor getting pre-packed
  // `buffer_sizes_` is the size of the pre-packed buffer.
  // In some rare cases, weights_size * sizeof(element) may not be equal to buffer_size of the pre-packed buffer.
  // Hence, we track both separately.
  std::vector<size_t> weights_sizes_;  // cache sizes associated with weights that are getting pre-packed
  std::vector<TensorShape> shapes_;    // cache tensor shapes associated with weights that are getting pre-packed
  std::vector<bool> flags_;            // cache some flags associated with the pre-packed buffers

  bool is_filled_ = false;  // By default, an instance of this class is "unfilled"

  // Produces a hash of the buffers stored in the given instance of this class
  uint64_t GetHash();
};

}  // namespace onnxruntime
