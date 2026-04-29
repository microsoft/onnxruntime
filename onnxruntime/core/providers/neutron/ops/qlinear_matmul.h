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

class QLinearMatMul : public MatMulIntegerBase {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QLinearMatMul);

  explicit QLinearMatMul(const OpKernelInfo& info) : MatMulIntegerBase(info) {}

  Status Compute(OpKernelContext* ctx) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

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

  int GetAIdx() const override { return IN_A; }
  int GetBIdx() const override { return IN_B; }

  // neutron parameters
  size_t m_handle{0};
  uint32_t* m_header{NULL};
  int8_t* m_b_neutron{NULL};
  int32_t* m_b_bias{NULL};
  uint32_t* m_b_factors{NULL};

  // pre-packing
  float m_a_scale_data;
  uint8_t m_a_zp;
  uint8_t m_y_zp;
  uint32_t m_b_rows;
  uint32_t m_b_cols;
  std::vector<float> m_b_scales;
  std::vector<float> m_output_scales;
};

class MatMulIntegerBase : public NeutronKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulIntegerBase);

  MatMulIntegerBase(const OpKernelInfo& info) : NeutronKernel(info) {}

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override {
    is_packed = false;

    // only pack Matrix B
    if (input_idx == GetBIdx()) {
      // Only handle the common case of a 2D weight matrix. Additional matrices
      // could be handled by stacking the packed buffers.
      b_shape_ = tensor.Shape();
      if (b_shape_.NumDimensions() != 2) {
        return Status::OK();
      }

      auto a_elem_type = Node().InputDefs()[GetAIdx()]->TypeAsProto()->tensor_type().elem_type();
      bool a_is_signed = ONNX_NAMESPACE::TensorProto_DataType_INT8 == a_elem_type;

      b_is_signed_ = tensor.IsDataType<int8_t>();

      size_t K = static_cast<size_t>(b_shape_[0]);
      size_t N = static_cast<size_t>(b_shape_[1]);

      const auto* b_data = static_cast<const uint8_t*>(tensor.DataRaw());

      std::optional<Tensor> b_trans_buffer;
      if (IsBTransposed()) {
        std::swap(K, N);
        b_data = quantization::TransPoseInputData(b_data, b_trans_buffer, alloc, N, K);
      }
      const size_t packed_b_size = MlasGemmPackBSize(N, K, a_is_signed, b_is_signed_, nullptr);
      if (packed_b_size == 0) {
        return Status::OK();
      }

      packed_b_ = IAllocator::MakeUniquePtr<void>(alloc, packed_b_size, true);
      // Zero-initialize to handle padding and ensure consistent hashes
      // when caching pre-packed buffers between sessions.
      memset(packed_b_.get(), 0, packed_b_size);
      MlasGemmPackB(N, K, b_data, N, a_is_signed, b_is_signed_, packed_b_.get());

      bool share_prepacked_weights = (prepacked_weights != nullptr);
      if (share_prepacked_weights) {
        prepacked_weights->buffers_.push_back(std::move(packed_b_));
        prepacked_weights->buffer_sizes_.push_back(packed_b_size);
      }

      is_packed = true;
    }
    return Status::OK();
  }

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   gsl::span<const size_t> prepacked_buffer_sizes,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override {
    ORT_UNUSED_PARAMETER(prepacked_buffer_sizes);
    used_shared_buffers = false;
    if (input_idx == GetBIdx()) {
      used_shared_buffers = true;
      packed_b_ = std::move(prepacked_buffers[0]);
    }

    return Status::OK();
  }

 protected:
  /// @return input index of Matrix A (activation/input data)
  virtual int GetAIdx() const { return 0; }

  /// @return input index of Matrix B (weight tensor)
  virtual int GetBIdx() const = 0;

  virtual bool IsBTransposed() const {
    return false;
  }

  /// Check if B's quantization parameter (scale/zp) shape is supported.
  /// Supported formats:
  /// 1. Scalar (per-tensor quantization)
  /// 2. 1D tensor with size 1 (per-tensor, same as scalar)
  /// 3. 1D tensor with size equal to B's last dimension (per-channel)
  /// 4. Same rank as B, with second-to-last dimension = 1
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

  bool b_is_signed_{true};
  TensorShape b_shape_;
  IAllocatorUniquePtr<void> packed_b_;
};

}  // namespace neutron
}  // namespace onnxruntime
