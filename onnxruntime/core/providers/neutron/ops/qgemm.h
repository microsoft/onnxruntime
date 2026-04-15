// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/neutron/neutron_kernel.h"

namespace onnxruntime {
namespace neutron {

extern std::shared_ptr<NeutronStackAllocator> neutronAlloc;

class QGemm : public NeutronKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QGemm);

  explicit QGemm(const OpKernelInfo& info) : NeutronKernel(info) {
    m_handle = neutronAlloc->getMemoryHandle();
    m_header = (uint32_t*)neutronAlloc->Alloc(16 * sizeof(uint32_t), m_handle);
  }

  Status Compute(OpKernelContext* ctx) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  // Check if quantization parameter of B is supported.
  // It should be in one of the formats below:
  // 1. Scalar
  // 2. 1D tensor with size equal to 1 or last dimension of B_shape if B_shape is a 2D tensor
  // 3. Equal to B_shape except that the second to last is 1
  bool IsBQuantParamSupported(const TensorShape& B_quant_param_shape, const TensorShape& B_shape) const {
    int64_t B_quant_param_rank = B_quant_param_shape.NumDimensions();
    int64_t B_shape_rank = B_shape.NumDimensions();
    if (B_quant_param_rank == 0 ||                                       // scalar
        (B_quant_param_rank == 1 && B_quant_param_shape.Size() == 1)) {  // 1D tensor with size 1
      return true;
    }

    if (B_quant_param_rank == 1 &&
        B_shape_rank == 2 &&
        B_quant_param_shape[0] == B_shape[1]) {
      return true;
    }

    if (B_quant_param_rank != B_shape_rank ||
        B_quant_param_rank <= 1 ||
        B_quant_param_shape[SafeInt<size_t>(B_quant_param_rank) - 2] != 1) {
      return false;
    }

    for (int64_t rank = 0; rank < B_quant_param_rank; rank++) {
      if (rank != B_quant_param_rank - 2 &&
          B_quant_param_shape[onnxruntime::narrow<size_t>(rank)] != B_shape[onnxruntime::narrow<size_t>(rank)]) {
        return false;
      }
    }

    return true;
  }

  enum InputTensors : int {
    IN_A = 0,
    IN_A_SCALE = 1,
    IN_A_ZERO_POINT = 2,
    IN_B = 3,
    IN_B_SCALE = 4,
    IN_B_ZERO_POINT = 5,
    IN_C = 6,
    IN_Y_SCALE = 7,
    IN_Y_ZERO_POINT = 8
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  int GetAIdx() const { return IN_A; }
  int GetBIdx() const { return IN_B; }

  // neutron parameters
  size_t m_handle{0};
  uint32_t* m_header{NULL};
  int8_t* m_b_neutron{NULL};
  int32_t* m_b_bias{NULL};
  uint32_t* m_b_factors{NULL};

  // pre-packing
  float m_a_scale_data;
  float m_y_scale_data;
  uint8_t m_a_zp;
  uint8_t m_y_zp;
  uint32_t m_b_rows;
  uint32_t m_b_cols;
  const int32_t* m_c_data;
  std::vector<float> m_b_scales;
  std::vector<float> m_output_scales;
};

}  // namespace neutron
}  // namespace onnxruntime
