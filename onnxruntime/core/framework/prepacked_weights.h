// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/buffer_deleter.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

struct PrePackedWeights final {
  // Some weights may be associated with multiple pre-packed buffers (e.g.) QLinearConv.
  // Hence we hold them in container. It is upto the developer implementing each PrePack()
  // method to define what gets stored in which position of the container.

  std::vector<std::unique_ptr<void, BufferDeleter>> buffers_;  // cache pre-packed buffers associated with the kernel
  std::vector<size_t> buffer_sizes_;                           // cache sizes of pre-packed buffers (in bytes)

  // Produces a hash of the buffers stored in the given instance of this class
  uint64_t GetHash() const;
};

}  // namespace onnxruntime
