#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/attention.h"

namespace onnxruntime {
namespace contrib {
template <typename T, typename QInput, typename QWeight>
class QAttention : public OpKernel, public AttentionBase {
 public:
  QAttention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace contrib
}  // namespace onnxruntime
