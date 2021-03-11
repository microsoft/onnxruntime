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
  enum InputID : int {
    IN_A = 0,
    IN_Ascale = 1,
    IN_Azero = 2,
    IN_B = 3,
    IN_Bscale = 4,
    IN_Bzero = 5,
    IN_Yscale = 6,
    IN_Yzero = 7
  };

  enum OutputID : int {
    OUT_Y = 0
  };

 protected:
  int GetBIdx() override {
    return IN_B;
  }
};

}  // namespace onnxruntime