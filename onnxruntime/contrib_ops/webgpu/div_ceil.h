// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/math/binary_elementwise_ops.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class DivCeil final : public BinaryElementwise {
 public:
  DivCeil(const OpKernelInfo& info)
      : BinaryElementwise{info, "DivCeil",
                          "select(ceil(a / b), round(a / b), (a % b) == output_value_t(0.0))"} {}
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
