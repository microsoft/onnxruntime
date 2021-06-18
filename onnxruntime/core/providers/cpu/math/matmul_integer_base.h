// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"

namespace onnxruntime {

class MatMulIntegerBase : public OpKernel {
 public:
  MatMulIntegerBase(const OpKernelInfo& info) : OpKernel(info) {}

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

      b_is_signed_ = tensor.IsDataType<int8_t>();

      const size_t K = static_cast<size_t>(b_shape_[0]);
      const size_t N = static_cast<size_t>(b_shape_[1]);

      const auto* b_data = static_cast<const uint8_t*>(tensor.DataRaw());

      const size_t packed_b_size = MlasGemmPackBSize(N, K, b_is_signed_);
      if (packed_b_size == 0) {
        return Status::OK();
      }

      auto* packed_b_data = alloc->Alloc(packed_b_size);

      // Initialize memory to 0 as there could be some padding associated with pre-packed
      // buffer memory and we don not want it uninitialized and generate different hashes
      // if and when we try to cache this pre-packed buffer for sharing between sessions.
      memset(packed_b_data, 0, packed_b_size);

      packed_b_ = BufferUniquePtr(packed_b_data, BufferDeleter(alloc));
      MlasGemmPackB(N, K, b_data, N, b_is_signed_, packed_b_data);

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
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override {
    used_shared_buffers = false;

    if (input_idx == GetBIdx()) {
      used_shared_buffers = true;
      packed_b_ = std::move(prepacked_buffers[0]);
    }

    return Status::OK();
  }

 protected:
  /**
   * @return input index of Matrix B, the weight tensor 
  */
  virtual int GetBIdx() = 0;

  // Check if quantization parameter of B is supported.
  // It should be in one of the formats below:
  // 1. Scalar
  // 2. 1D tensor with size equal to 1 or last dimension of B_shape if B_shape is a 2D tensor
  // 3. Equal to B_shape except that the second to last is 1
  bool IsBQuantParamSupported(const TensorShape& B_quant_param_shape, const TensorShape& B_shape) const {
    int64_t B_quant_param_rank = B_quant_param_shape.NumDimensions();
    int64_t B_shape_rank = B_shape.NumDimensions();
    if (B_quant_param_rank == 0 ||                                     //scalar
        B_quant_param_rank == 1 && B_quant_param_shape.Size() == 1) {  // 1D tensor with size 1
      return true;
    }

    if (B_quant_param_rank == 1 &&
        B_shape_rank == 2 &&
        B_quant_param_shape[B_quant_param_rank - 1] == B_shape[B_shape_rank - 1]) {
      return true;
    }

    if (B_quant_param_rank != B_shape_rank ||
        B_quant_param_rank <= 1 ||
        B_quant_param_shape[B_quant_param_rank - 2] != 1) {
      return false;
    }

    for (int64_t rank = 0; rank < B_quant_param_rank; rank++) {
      if (rank != B_quant_param_rank - 2 &&
          B_quant_param_shape[rank] != B_shape[rank]) {
        return false;
      }
    }

    return true;
  }

  bool b_is_signed_{true};
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;
};

}  // namespace onnxruntime
