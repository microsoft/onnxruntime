// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class EmbedLayerNormBase : public OpKernel {
 public:
  explicit EmbedLayerNormBase(const OpKernelInfo& op_kernel_info);

 protected:
  float epsilon() const;
  bool is_add_output() const;

 private:
  float epsilon_;
  bool add_output_;
};

template <typename T>
class EmbedLayerNorm : public EmbedLayerNormBase {
 public:
  explicit EmbedLayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;
};
}  // namespace contrib
}  // namespace onnxruntime
