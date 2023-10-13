// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/transformers/beam_search_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

struct GreedySearchParameters : public BeamSearchParameters {
  int BatchBeamSize() const { return batch_size; }

  void ParseFromAttributes(const OpKernelInfo& info) override;

  void ParseFromInputs(OpKernelContext* context);
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
