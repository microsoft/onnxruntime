// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class MoEPacked final : public OpKernel {
 public:
  explicit MoEPacked(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t k_{0};
  int64_t normalize_routing_weights_{1};
  std::string activation_type_;
  int64_t gate_type_{-1};
  int64_t up_type_{-1};
  int64_t down_type_{-1};
};

}  // namespace contrib
}  // namespace onnxruntime
