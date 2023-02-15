// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

struct BeamSearchParameters : public IGenerationParameters {
  Status Validate() const;

  int BatchBeamSize() const { return batch_size * num_beams; }

  void ParseFromAttributes(const OpKernelInfo& info);

  void ParseFromInputs(OpKernelContext* context);

  void SetSubgraphParameters(int vocab_size, int num_heads, int head_size, int num_layers);
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
