// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "matmul_integer_base.h"

namespace onnxruntime {

// Allow subclassing for test only
class QLinearMatMul : public MatMulIntegerBase {
 public:
  QLinearMatMul(const OpKernelInfo& info) : MatMulIntegerBase(info) {}

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
  int GetBIdx() override {
    return IN_B;
  }
};

}  // namespace onnxruntime