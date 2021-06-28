// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "embed_layer_norm.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include <vector>

namespace onnxruntime {
namespace contrib {

// Quantized version of QEmbedLayerNorm.
template <typename T>
class QEmbedLayerNorm final : public EmbedLayerNormBase {
 public:
  explicit QEmbedLayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<uint8_t> word_embedding_lookup_table_;
  std::vector<uint8_t> position_embedding_lookup_table_;
  std::vector<uint8_t> segment_embedding_lookup_table_;
  std::vector<uint8_t> gamma_lookup_table_;
  std::vector<uint8_t> beta_lookup_table_;
};

}  // namespace contrib
}  // namespace onnxruntime
