// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include <memory>
#include <string>

namespace onnxruntime {
namespace webgpu {

// Weight layout transformer for optimized weight formats
// Handles transformations like oihw->hwio transpose and blocked formats
class WeightLayoutTransformer {
 public:
  // Transform a tensor to a different layout format
  // format_descriptor: Format string (e.g., "hwio", "ABcd16a4b")
  // Returns Status::OK() on success, error status otherwise
  static Status TransformLayout(const Tensor& original_tensor,
                                const std::string& format_descriptor,
                                std::unique_ptr<Tensor>& transformed_tensor);

 private:
  // Transpose weights from oihw to hwio layout
  template <typename T>
  static void TransposeOIHWToHWIO(const T* src, T* dst,
                                  int64_t O, int64_t I, int64_t H, int64_t W);

  // Reorder weights from oihw to ABcd16a4b blocked format
  template <typename T>
  static void ReorderToBlockedFormat(const T* src, T* dst,
                                     int64_t O, int64_t I, int64_t H, int64_t W,
                                     int64_t O_blocks, int64_t I_blocks,
                                     int64_t block_o, int64_t block_i);
};

}  // namespace webgpu
}  // namespace onnxruntime
