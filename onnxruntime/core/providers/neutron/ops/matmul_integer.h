// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/neutron/neutron_kernel.h"
#include "core/providers/cpu/quantization/matmul_integer_base.h"

namespace onnxruntime {
namespace neutron {

class MatMulInteger final : public MatMulIntegerBase {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulInteger);

  MatMulInteger(const OpKernelInfo& info) : MatMulIntegerBase(info) {}

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_ZERO_POINT = 2,
    IN_B_ZERO_POINT = 3
  };

  enum OutputTensors : int { OUT_Y = 0 };

 protected:
  int GetBIdx() const override { return IN_B; }

  // neutron parameters
  size_t m_handle{0};
  uint32_t* m_header{NULL};
  int8_t* m_b_neutron{NULL};
  bool m_dynamic_bias{true};
  int32_t* m_b_bias{NULL};
  int32_t* m_b_row_sum{NULL};
  uint32_t* m_b_factors{NULL};

  // pre-packing
  uint8_t m_a_zp;
  uint32_t m_b_rows;
  uint32_t m_b_cols;
};

}  // namespace neutron
}  // namespace onnxruntime
