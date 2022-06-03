// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "matmul_integer_base.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace onnxruntime {

// Allow subclassing for test only
class QLinearMatMul : public MatMulIntegerBase {
 public:
  QLinearMatMul(const OpKernelInfo& info) : MatMulIntegerBase(info) {
    const CPUExecutionProvider* ep = static_cast<const CPUExecutionProvider*>(info.GetExecutionProvider());
    use_fixed_point_requant_ = ep->UseFixedPointRequantOnARM64();
  }

  Status Compute(OpKernelContext* context) const override;

  /**
   * @brief Give each input a name, should be consistent with doc spec in
   * Operators.md
  */
  enum InputTensors : int {
    IN_A = 0,
    IN_A_SCALE = 1,
    IN_A_ZERO_POINT = 2,
    IN_B = 3,
    IN_B_SCALE = 4,
    IN_B_ZERO_POINT = 5,
    IN_Y_SCALE = 6,
    IN_Y_ZERO_POINT = 7
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

 protected:
  int GetBIdx() const override {
    return IN_B;
  }

  bool use_fixed_point_requant_{false};
};

}  // namespace onnxruntime