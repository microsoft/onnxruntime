// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"

namespace onnxruntime {

class MatMulIntegerBase : public OpKernel {
 public:
  MatMulIntegerBase(const OpKernelInfo& info) : OpKernel(info) {}

  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed,
                 /*in_out*/ PackedWeight& cached_prepacked_tensor,
                 /*out*/ bool& read_from_cache,
                 AllocatorPtr alloc_for_caching) override {
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

      // Cached pre-packed weight
      if (cached_prepacked_tensor.has_cached_) {
        read_from_cache = true;
        is_packed = true;
        b_shape_ = cached_prepacked_tensor.shapes_[0];
        packed_b_ = BufferUniquePtr(cached_prepacked_tensor.buffers_[0].get(), BufferDeleter(nullptr));
        return Status::OK();
      }

      const size_t K = static_cast<size_t>(b_shape_[0]);
      const size_t N = static_cast<size_t>(b_shape_[1]);

      const auto* b_data = static_cast<const uint8_t*>(tensor.DataRaw());

      const size_t packed_b_size = MlasGemmPackBSize(N, K, b_is_signed_);
      if (packed_b_size == 0) {
        return Status::OK();
      }

      bool kernel_owns_prepacked_buffer = (alloc_for_caching == nullptr);
      AllocatorPtr alloc = kernel_owns_prepacked_buffer ? Info().GetAllocator(0, OrtMemTypeDefault) : alloc_for_caching;

      auto* packed_b_data = alloc->Alloc(packed_b_size);
      packed_b_ = BufferUniquePtr(packed_b_data, BufferDeleter(alloc));
      MlasGemmPackB(N, K, b_data, N, b_is_signed_, packed_b_data);

      if (!kernel_owns_prepacked_buffer) {
        cached_prepacked_tensor.buffers_.push_back(std::move(packed_b_));
        cached_prepacked_tensor.shapes_.push_back(b_shape_);
        cached_prepacked_tensor.has_cached_ = true;
        packed_b_ = BufferUniquePtr(cached_prepacked_tensor.buffers_[0].get(), BufferDeleter(nullptr));
      }

      is_packed = true;
    }
    return Status::OK();
  }

 protected:
  /**
   * @return input index of Matrix B, the weight tensor 
  */
  virtual int GetBIdx() = 0;

  bool b_is_signed_{true};
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;
};

}  // namespace onnxruntime
