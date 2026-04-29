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

int test_main();
// MatMulNBits op input indices.
// These should match the inputs names specified in the op schema.
namespace InputIndex {
constexpr size_t IN_A = 0,
                 IN_B = 1,
                 SCALES = 2,
                 ZERO_POINTS = 3,
                 G_IDX = 4,
                 BIAS = 5;
};

namespace OutputIndex {
constexpr size_t OUT_Y = 0;
};

// T1 is the type of the input matrix A, scales and biases.
// Use class level template to facilitate specialization for different types.
class MatMulNBits final : public OpKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulNBits);

  MatMulNBits(const OpKernelInfo& info)
      : OpKernel(info),
        K_{narrow<size_t>(info.GetAttr<int64_t>("K"))},
        N_{narrow<size_t>(info.GetAttr<int64_t>("N"))},
        block_size_{narrow<size_t>(info.GetAttr<int64_t>("block_size"))},
        nbits_{narrow<size_t>(info.GetAttr<int64_t>("bits"))},
        blocks_per_col_{(K_ + block_size_ - 1) / block_size_},
        accuracy_level_{4},
        offline_packed_{static_cast<const NeutronExecutionProvider*>(info.GetExecutionProvider())->IsOfflinePacked()} {
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   gsl::span<const size_t> input_indices,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  const size_t K_;
  const size_t N_;
  const size_t block_size_;
  const size_t nbits_;
  const size_t blocks_per_col_;
  const int64_t accuracy_level_;
  const uint8_t* unpacked_b_{NULL};
  size_t b_size_;
  float* int8_scale_{NULL};

  // neutron parameters
  size_t m_handle{0};
  uint32_t* m_header{NULL};
  uint8_t* m_decode_input{NULL};
  uint32_t* m_b_factors{NULL};
  int32_t* m_b_bias{NULL};
  int32_t* m_b_row_sum{NULL};
  int16_t* m_decode_scale{NULL};
  int8_t* m_b_neutron{NULL};
  int8_t* m_decode_bias{NULL};

  float* float_b{NULL};
  const bool offline_packed_{false};
};

}  // namespace neutron
}  // namespace onnxruntime
