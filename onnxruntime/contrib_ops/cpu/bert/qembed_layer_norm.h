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
  // TODO - better name? What about segment in here?
  std::vector<uint8_t> word_position_embedding_lookup_table_;

  bool has_cached_lookup_table_;
};

}  // namespace contrib
}  // namespace onnxruntime
