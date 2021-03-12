// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "matmul_integer_base.h"

namespace onnxruntime {

// Allow subclassing for test only
class QLinearMatMul : public MatMulIntegerBase {
 public:
  QLinearMatMul(const OpKernelInfo& info) : MatMulIntegerBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime